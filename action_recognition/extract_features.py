#!/usr/bin/env python3
"""
第二步: 冻结 Qwen2.5-VL Vision Tower 逐帧特征提取 (GPU)

输入: extract_pose.py 输出的 skeleton JSON (COCO17 3D)
处理: 30 帧滑窗 (默认 stride=1), 每窗按 3D 包围盒归一化后渲染成双视角骨架图
      (左半正面 x,-y / 右半侧面 z,-y, 448x448), 逐帧过冻结 ViT 并 mean-pool 成 1280 维
输出: npz {"feats": [N,30,1280] float16, "starts": [N] int64}
      可用 --out 指定路径; infer.py 通过 --feats-cache 直接复用, 跳过本步

用法:
  python extract_features.py --skeleton in_pose.json --out in_feats.npz \
      --qwen Qwen/Qwen2.5-VL-3B-Instruct [--stride 1] [--chunk 48] [--device cuda:0]

显存: chunk=48 时约 14GB (24G 卡安全); 更大显存可调高 --chunk 提速
"""
import os
import json
import time
import argparse
import numpy as np
import cv2
import torch
from PIL import Image

WINDOW = 30
IMG_SIZE = 448
PANEL_W = IMG_SIZE // 2
MIN_AXIS_SPAN = 0.6   # 每轴归一化跨度下限 (米), 防止深度轴噪声放大

LEFT_JOINTS = {0, 1, 3, 5, 7, 9, 11, 13, 15}
RIGHT_JOINTS = {2, 4, 6, 8, 10, 12, 14, 16}
LEFT_PAIRS = [(0, 1), (1, 3), (5, 7), (7, 9), (5, 11), (11, 13), (13, 15)]
RIGHT_PAIRS = [(0, 2), (2, 4), (6, 8), (8, 10), (6, 12), (12, 14), (14, 16)]
CENTER_PAIRS = [(5, 6), (11, 12)]
COL_LEFT = (200, 60, 40)
COL_RIGHT = (40, 60, 200)
COL_CENTER = (160, 160, 160)
COL_PANEL_LINE = (80, 80, 80)


def load_skeleton_json(path):
    """MediaPipe skeleton JSON -> dict frame_id -> COCO17 3D [17,3] (无效帧缺失)"""
    with open(path) as f:
        data = json.load(f)
    frs = data.get("person_0", [])
    out = {}
    for fr in frs:
        arr = np.asarray(fr["joints3d"], dtype=np.float32)
        if arr.shape != (17, 3) or np.abs(arr).sum() == 0:
            continue
        out[int(fr["frame_id"])] = arr
    return out


def norm_box3d_of(kps_list):
    """窗口级 3D 归一化框: (x_min, y_min, z_min, x_span, y_span, z_span)"""
    all_kps = np.concatenate(kps_list, axis=0)
    valid = np.abs(all_kps).sum(axis=1) > 0
    if valid.sum() == 0:
        return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    pts = all_kps[valid]
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(maxs - mins, MIN_AXIS_SPAN)
    margin = 0.08
    mins = mins - margin * span
    span = span * (1 + 2 * margin)
    return (float(mins[0]), float(mins[1]), float(mins[2]),
            float(span[0]), float(span[1]), float(span[2]))


def render_frame_3d(kps17, box3d, img_size=IMG_SIZE):
    """双视角骨架图: 左半正面(x,-y), 右半侧面(z,-y)。kps17: [17,3], 无效关节全0。"""
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    if kps17 is None or np.abs(kps17).sum() == 0:
        return canvas
    x0, y0, z0, xs, ys, zs = box3d
    k = kps17.astype(np.float32).copy()
    front = np.stack([(k[:, 0] - x0) / xs, (y0 + ys - k[:, 1]) / ys], axis=1)
    side = np.stack([(k[:, 2] - z0) / zs, (y0 + ys - k[:, 1]) / ys], axis=1)

    for view, x_off in ((front, 0), (side, PANEL_W)):
        for pairs, col, th in ((LEFT_PAIRS, COL_LEFT, 3),
                               (RIGHT_PAIRS, COL_RIGHT, 3),
                               (CENTER_PAIRS, COL_CENTER, 2)):
            for p1, p2 in pairs:
                a, b = view[p1], view[p2]
                if np.abs(kps17[p1]).sum() == 0 or np.abs(kps17[p2]).sum() == 0:
                    continue
                pa = (int(a[0] * (PANEL_W - 12) + 6 + x_off), int(a[1] * (img_size - 16) + 8))
                pb = (int(b[0] * (PANEL_W - 12) + 6 + x_off), int(b[1] * (img_size - 16) + 8))
                cv2.line(canvas, pa, pb, col, th)
        for ji in LEFT_JOINTS | RIGHT_JOINTS:
            if np.abs(kps17[ji]).sum() == 0:
                continue
            col = COL_LEFT if ji in LEFT_JOINTS else COL_RIGHT
            p = view[ji]
            px, py = int(p[0] * (PANEL_W - 12) + 6 + x_off), int(p[1] * (img_size - 16) + 8)
            cv2.circle(canvas, (px, py), 4, col, -1)
    cv2.line(canvas, (PANEL_W, 0), (PANEL_W, img_size), COL_PANEL_LINE, 1)
    return canvas


@torch.no_grad()
def visual_tokens_1280(visual, pixel_values, device):
    """Qwen2.5-VL ViT 前向 (transformers==4.49 内部结构), 返回 [B,1024,1280]"""
    B = pixel_values.shape[0] // 1024
    h = visual.patch_embed(pixel_values).view(B, 1024, -1)

    grid = torch.tensor([[1, 32, 32]], dtype=torch.int64)
    unit_perm = visual.get_window_index(grid.to(device))[0].to(device)
    token_perm = (unit_perm[:, None] * 4 + torch.arange(4, device=device)[None, :]).flatten()
    rotary_full = visual.rot_pos_emb(grid.to(device))
    emb = torch.cat((rotary_full, rotary_full), dim=-1)
    dt = visual.patch_embed.proj.weight.dtype
    cos, sin = emb.cos().to(dt), emb.sin().to(dt)
    cos_p, sin_p = cos[token_perm].contiguous(), sin[token_perm].contiguous()
    h = h[:, token_perm, :]

    n_heads = visual.blocks[0].attn.num_heads
    hd = h.shape[-1] // n_heads

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    for layer_num, blk in enumerate(visual.blocks):
        a = blk.attn
        y = blk.norm1(h)
        B_, L, D = y.shape
        q, k, v = a.qkv(y).reshape(B_, L, 3, n_heads, hd).permute(2, 0, 1, 3, 4).unbind(0)
        if layer_num in visual.fullatt_block_indexes:
            q = q * cos_p[None, :, None, :] + rotate_half(q) * sin_p[None, :, None, :]
            k = k * cos_p[None, :, None, :] + rotate_half(k) * sin_p[None, :, None, :]
            o = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            o = o.transpose(1, 2).reshape(B_, L, -1)
        else:
            qw = q.reshape(B_ * 16, 64, n_heads, hd)
            kw = k.reshape(B_ * 16, 64, n_heads, hd)
            vw = v.reshape(B_ * 16, 64, n_heads, hd)
            cos_w = cos_p.reshape(16, 64, hd).repeat(B_, 1, 1)
            sin_w = sin_p.reshape(16, 64, hd).repeat(B_, 1, 1)
            qw = qw * cos_w[:, :, None, :] + rotate_half(qw) * sin_w[:, :, None, :]
            kw = kw * cos_w[:, :, None, :] + rotate_half(kw) * sin_w[:, :, None, :]
            ow = torch.nn.functional.scaled_dot_product_attention(
                qw.transpose(1, 2), kw.transpose(1, 2), vw.transpose(1, 2)
            )
            o = ow.transpose(1, 2).reshape(B_, L, -1)
        h = h + a.proj(o)
        h = h + blk.mlp(blk.norm2(h))
    return h


@torch.no_grad()
def encode_frames(all_frames, qwen_path, device, chunk=48, quiet=False):
    """all_frames: [N,448,448,3] uint8 -> [N,1280] float32 (ViT mean-pool 特征)"""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(qwen_path).image_processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_path, torch_dtype=torch.bfloat16).to(device)
    visual = model.visual
    visual.eval()
    del model.model, model.lm_head
    torch.cuda.empty_cache()
    tpi = (IMG_SIZE // 14) ** 2
    out = np.zeros((len(all_frames), 1280), np.float16)
    t0 = time.time()
    for c0 in range(0, len(all_frames), chunk):
        ch = all_frames[c0:c0 + chunk]
        batch = proc(images=[Image.fromarray(im) for im in ch], return_tensors="pt")
        pv = batch["pixel_values"].to(device)
        f = visual_tokens_1280(visual, pv, device)
        f = f.float().view(len(ch), tpi, 1280).mean(1)
        out[c0:c0 + len(ch)] = f.half().cpu().numpy()
        if not quiet:
            el = time.time() - t0
            print(f"[ENC] {c0+len(ch)}/{len(all_frames)}  {el:.1f}s", flush=True)
    del visual
    torch.cuda.empty_cache()
    return out.astype(np.float32)


def build_windows(frame_map, stride):
    """全视频滑窗; 缺帧用全零占位 (渲染为黑)"""
    max_f = max(frame_map) + 1
    starts = list(range(0, max_f - WINDOW + 1, stride))
    assert starts, f"视频帧数 {max_f} < 窗口 {WINDOW}"
    return starts


def extract_for_video(skeleton_json, qwen_path, device, stride=1, chunk=48,
                      cache_path=None, quiet=False):
    """完整特征提取流程 (infer.py 在无缓存时调用本函数); 返回 (feats[N,30,1280] f32, starts)"""
    frame_map = load_skeleton_json(skeleton_json)
    assert frame_map, f"无有效骨架: {skeleton_json}"
    starts = build_windows(frame_map, stride)
    if not quiet:
        print(f"[WIN] {len(starts)} windows (stride={stride}), {max(frame_map)+1} frames",
              flush=True)
    all_frames = np.zeros((len(starts) * WINDOW, IMG_SIZE, IMG_SIZE, 3), np.uint8)
    for wi, s in enumerate(starts):
        ks = [frame_map.get(i) for i in range(s, s + WINDOW)]
        ks = [k if k is not None else np.zeros((17, 3), np.float32) for k in ks]
        box = norm_box3d_of(ks)
        all_frames[wi * WINDOW:(wi + 1) * WINDOW] = [render_frame_3d(k, box) for k in ks]
    flat = encode_frames(all_frames, qwen_path, device, chunk=chunk, quiet=quiet)
    feats = flat.reshape(len(starts), WINDOW, 1280)
    if cache_path:
        np.savez_compressed(cache_path, feats=feats.astype(np.float16),
                            starts=np.array(starts, dtype=np.int64))
        if not quiet:
            print(f"[CACHE] -> {cache_path}")
    return feats, np.array(starts, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton", required=True, help="extract_pose.py 输出的 json")
    ap.add_argument("--out", required=True, help="输出 npz 路径")
    ap.add_argument("--qwen", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                    help="Qwen2.5-VL 模型路径 (HF id 或本地目录)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=48, help="ViT 批量大小 (显存敏感)")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    extract_for_video(args.skeleton, args.qwen, args.device,
                      stride=args.stride, chunk=args.chunk, cache_path=args.out)


if __name__ == "__main__":
    main()
