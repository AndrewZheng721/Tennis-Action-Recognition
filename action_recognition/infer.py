#!/usr/bin/env python3
"""
第三步: 动作识别推理 + 标注视频渲染

流程: skeleton json -> (复用缓存或现场提取 ViT 特征) -> 时序Transformer softmax
      -> 滑窗概率按帧聚合 -> 渲染 (骨架插值防闪烁 + 人物框上方大字动作标签)
      可选 (--court-weights): 本仓库 src/court 球场关键点检测 + 右侧俯视热力图
      (球场 1.5x, JET 热力, 90 帧轨迹尾迹, 底部大字号跑动统计) + 关键点连线

用法:
  # 基础动作识别
  python infer.py --video in.mp4 --skeleton in_pose.json --out pred.mp4

  # 完整版: 动作 + 球场关键点 + 俯视热力图 (在仓库根目录运行, 复用 src/court)
  python infer.py --video in.mp4 --skeleton in_pose.json --out pred.mp4 \
      --court-weights weights/court_heatmap.pth

  # 附加精度评估 (标注 txt: 每行 "xxx 起始秒 结束秒 xxx 标签")
  python infer.py ... --gt in.txt
"""
import os
import sys
import json
import argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import load_skeleton_json, extract_for_video, WINDOW  # noqa: E402
from model import load_ensemble, predict_ensemble_probs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)          # 本目录置于仓库根时指向仓库根
DOUBLES_WIDTH, COURT_LENGTH = 10.97, 23.77
FPS = 30
DEFAULT_CKPTS = [os.path.join(HERE, "weights", f"action_transformer_3cls_fold{i}.pt")
                 for i in range(5)]

BONES = [(0, 1), (1, 3), (3, 5), (5, 7), (7, 9), (0, 2), (2, 4), (4, 6), (6, 8), (8, 10),
         (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
BASE_COL = {"ready": (255, 220, 60), "forehand": (60, 160, 255),
            "backhand": (80, 255, 120), "serve": (255, 0, 255)}
EXTRA_COL = [(255, 160, 0), (0, 255, 255), (180, 100, 255), (100, 255, 180)]
# 14 点连线: 双打外框/单打边线/发球线/中线
COURT_LINKS = [(0, 1), (1, 3), (3, 2), (2, 0),
               (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]


def label_colors(labels):
    col = {}
    for i, l in enumerate(labels):
        col[l] = BASE_COL.get(l, EXTRA_COL[i % len(EXTRA_COL)])
    return col


def load_j2d(skeleton_json):
    with open(skeleton_json) as f:
        data = json.load(f)
    return {int(fr["frame_id"]): np.array(fr["joints2d"], np.float32)
            for fr in data.get("person_0", [])}


def interp_joints(j2d_map, max_f):
    """逐关节线性插值: 漏检帧补齐, 从未见过的关节保持 nan"""
    js = np.full((max_f, 17, 2), np.nan)
    vj = np.zeros((max_f, 17), bool)
    for t, k in j2d_map.items():
        if t >= max_f:
            continue
        vis = k[:, 0] > 0
        vj[t] = vis
        js[t, vis] = k[vis]
    idx = np.arange(max_f)
    for j in range(17):
        mk = vj[:, j]
        if mk.sum() < 2:
            continue
        for c in range(2):
            js[:, j, c] = np.interp(idx, idx[mk], js[mk, j, c])
    return js


def median_smooth(x, k=5):
    n = len(x)
    out = x.copy()
    for i in range(n):
        a, b = max(0, i - k // 2), min(n, i + k // 2 + 1)
        out[i] = np.median(x[a:b], axis=0)
    return out


def frame_aggregate(starts, max_f, win_probs, window=WINDOW):
    fp = np.zeros((max_f, win_probs.shape[1]), np.float32)
    fc = np.zeros(max_f, np.float32)
    for wi, s in enumerate(starts):
        fp[s:s + window] += win_probs[wi]
        fc[s:s + window] += 1
    fp /= np.maximum(fc, 1)[:, None]
    return fp


# ---------------- 球场相关 (仅 --court-weights 时启用) ----------------

def import_court_modules():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from src.court.heatmap_court import HeatmapCourtDetector, REF_KPS  # noqa
    from src.score.geom import image_to_court  # noqa
    return HeatmapCourtDetector, REF_KPS, image_to_court


def detect_court(court_weights, video, device, cache_tag):
    HeatmapCourtDetector, REF_KPS, _ = import_court_modules()
    det = HeatmapCourtDetector(court_weights, device=device)
    cap = cv2.VideoCapture(video)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    best = None
    results = []
    for fid in range(0, n_total, 10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, img = cap.read()
        if not ok:
            continue
        kps, H, H_inv = det.predict(img)
        kps2 = kps.reshape(-1, 2)
        nvalid = int(np.isfinite(kps2[:, 0]).sum())
        results.append((fid, nvalid))
        if H_inv is not None and (best is None or nvalid > best[0]):
            best = (nvalid, fid, kps2.copy(), H, H_inv)
    cap.release()
    print(f"[COURT] 抽帧检测: {results}")
    assert best is not None, "球场检测失败"
    nvalid, fid, kps, H, H_inv = best
    print(f"[COURT] 采用 f{fid}: 有效点 {nvalid}/14")
    cache = os.path.join(HERE, "outputs", f"court_det_{cache_tag}.npz")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, keypoints=kps, H=H, H_inv=H_inv, frame_id=fid)
    return kps, H.astype(np.float32), H_inv


def meters_trajectory(j2d_map, max_f, H_inv, image_to_court, ref_to_meters):
    """脚踝均值像素 -> 球场参考系 -> 米制; 钳制+插值+中值平滑"""
    raw = []
    for t in range(max_f):
        k = j2d_map.get(t)
        ank = [k[i] for i in (15, 16) if k is not None and np.abs(k[i]).sum() > 0]
        raw.append(np.mean(ank, axis=0) if ank else None)
    pos_m = np.full((max_f, 2), np.nan)
    valid = np.array([p is not None for p in raw])
    xy_img = np.array([tuple(p) for p in raw if p is not None])
    court_pts = [image_to_court(p, H_inv) for p in xy_img]
    meters = np.array([ref_to_meters(q) for q in court_pts if q is not None])
    ok = np.array([q is not None for q in court_pts])
    vi = np.where(valid)[0][ok]
    pos_m[vi] = meters
    pos_m[:, 0] = np.clip(pos_m[:, 0], -3.0, DOUBLES_WIDTH + 3)
    pos_m[:, 1] = np.clip(pos_m[:, 1], -4.0, COURT_LENGTH + 4.2)
    idx = np.arange(max_f)
    good = ~np.isnan(pos_m[:, 0])
    for c in range(2):
        pos_m[:, c] = np.interp(idx, idx[good], pos_m[good, c])
    return median_smooth(pos_m)


class CourtPanel:
    """右侧 640x1080: 球场上半(1.5x放大) + 底部大字号跑动统计"""

    def __init__(self, w=640, h=1080):
        self.w, self.h = w, h
        self.margin_x = 30
        self.y_min, self.y_max = -5.0, 26.5
        base = min((w - 2 * self.margin_x) / DOUBLES_WIDTH,
                   660.0 / (self.y_max - self.y_min))
        self.scale = base * 1.5
        self.top = 6
        self.cw = DOUBLES_WIDTH * self.scale
        self.ch = COURT_LENGTH * self.scale
        self.ox = (w - self.cw) / 2
        self.oy = self.top
        court_px_h = int((self.y_max - self.y_min) * self.scale)
        self.heat = np.zeros((court_px_h + 1, w), np.float32)
        self.stats_y = self.oy + court_px_h + 40

    def to_px(self, xy):
        x, y = xy
        return (int(self.ox + x * self.scale),
                int(self.oy + (y - self.y_min) * self.scale))

    def add_heat(self, xy, sigma_m=0.6):
        cx, cy = self.to_px(xy)
        gy = int(self.oy)
        sig = sigma_m * self.scale
        r = int(sig * 3)
        x0, x1 = max(0, cx - r), min(self.heat.shape[1], cx + r + 1)
        y0, y1 = max(gy, cy - r), min(gy + self.heat.shape[0], cy + r + 1)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2))
        self.heat[y0 - gy:y1 - gy, x0:x1] += g.astype(np.float32)

    def draw_court(self, panel):
        W, L = DOUBLES_WIDTH, COURT_LENGTH
        lines = [
            [(0, 0), (W, 0)], [(W, 0), (W, L)], [(W, L), (0, L)], [(0, L), (0, 0)],
            [(1.37, 0), (1.37, L)], [(W - 1.37, 0), (W - 1.37, L)],
            [(1.37, L / 2 - 6.4), (W - 1.37, L / 2 - 6.4)],
            [(1.37, L / 2 + 6.4), (W - 1.37, L / 2 + 6.4)],
            [(W / 2, L / 2 - 6.4), (W / 2, L / 2 + 6.4)],
        ]
        for ln in lines:
            cv2.line(panel, self.to_px(ln[0]), self.to_px(ln[1]), (200, 200, 200), 1)
        cv2.line(panel, self.to_px((0, L / 2)), self.to_px((W, L / 2)), (255, 0, 255), 2)
        cv2.putText(panel, "FAR", (self.to_px((W / 2, 0))[0] - 16, self.oy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(panel, "NEAR", (self.to_px((W / 2, COURT_LENGTH))[0] - 24,
                                    self.oy + int((self.y_max - self.y_min) * self.scale) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    def render(self, frame_i, court_m, stats):
        panel = np.full((self.h, self.w, 3), (30, 30, 30), np.uint8)
        self.draw_court(panel)
        if self.heat.max() > 0:
            hn = self.heat / self.heat.max()
            hm = (np.clip(hn * 2.0, 0, 1) * 255).astype(np.uint8)
            hm_col = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            y0 = int(self.oy)
            roi = panel[y0:y0 + self.heat.shape[0], 0:self.heat.shape[1]]
            mask = (hn > 0.08)[..., None]
            blended = cv2.addWeighted(roi, 0.60, hm_col, 0.40, 0)
            np.copyto(roi, blended, where=mask)
        s0 = max(0, frame_i - 90)
        for k in range(s0, frame_i + 1):
            cv2.circle(panel, self.to_px(court_m[k]), 2, (255, 255, 0), -1)
        cv2.circle(panel, self.to_px(court_m[frame_i]), 6, (255, 255, 0), -1)
        cv2.circle(panel, self.to_px(court_m[frame_i]), 8, (255, 255, 255), 1)
        y = self.stats_y
        for text, col in stats:
            cv2.putText(panel, text, (self.margin_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, col, 2, cv2.LINE_AA)
            y += 34
        return panel


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--skeleton", required=True, help="extract_pose.py 输出的 json")
    ap.add_argument("--ckpts", nargs="+", default=DEFAULT_CKPTS,
                    help="5 折模型权重路径列表 (默认 weights/ 下 5 个 fold)")
    ap.add_argument("--qwen", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--feats-cache", default="", help="特征 npz; 已存在则直接复用")
    ap.add_argument("--out", required=True, help="输出标注视频路径")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=48, help="ViT 批量 (显存敏感)")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--court-weights", default="",
                    help="球场关键点权重 (本仓库 weights/court_heatmap.pth); 给出则启用俯视图")
    ap.add_argument("--gt", default="", help="可选: 标注 txt, 评估帧精度/召回")
    args = ap.parse_args()
    device = args.device
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    frame_map = load_skeleton_json(args.skeleton)
    assert frame_map, f"无有效骨架: {args.skeleton}"
    max_f = max(frame_map) + 1
    j2d_map = load_j2d(args.skeleton)
    j2s = interp_joints(j2d_map, max_f)

    # ---- 特征 (缓存优先) ----
    cache = args.feats_cache or (os.path.splitext(args.out)[0] + "_feats.npz")
    if os.path.exists(cache):
        print(f"[CACHE] load {cache}")
        d = np.load(cache)
        feats, starts = d["feats"].astype(np.float32), d["starts"]
    else:
        feats, starts = extract_for_video(args.skeleton, args.qwen, device,
                                          stride=args.stride, chunk=args.chunk,
                                          cache_path=cache)

    # ---- 推理 (5 折集成) ----
    folds, labels = load_ensemble(args.ckpts, device)
    COL = label_colors(labels)
    win_probs = predict_ensemble_probs(folds, feats, device)
    fp = frame_aggregate(starts, max_f, win_probs)
    frame_pred = fp.argmax(1)
    print("[PRED] 帧分布: " + ", ".join(
        f"{l}={int((frame_pred == i).sum())}" for i, l in enumerate(labels)))

    # ---- 球场 (可选) ----
    court = None
    if args.court_weights:
        _, REF_KPS, image_to_court = import_court_modules()
        REF_X0, REF_X1 = float(REF_KPS[0, 0]), float(REF_KPS[1, 0])
        REF_Y0, REF_Y1 = float(REF_KPS[0, 1]), float(REF_KPS[2, 1])
        NET_Y_REF = (float(REF_KPS[12, 1]) + float(REF_KPS[13, 1])) / 2.0

        def ref_to_meters(xy):
            x, y = xy
            return ((x - REF_X0) / (REF_X1 - REF_X0) * DOUBLES_WIDTH,
                    (y - REF_Y0) / (REF_Y1 - REF_Y0) * COURT_LENGTH)

        tag = os.path.splitext(os.path.basename(args.out))[0]
        kps, H, H_inv = detect_court(args.court_weights, args.video, device, tag)
        net_ref = np.array([[[423.0, NET_Y_REF]], [[1242.0, NET_Y_REF]]], np.float32)
        net_img = cv2.perspectiveTransform(net_ref, H).reshape(2, 2)
        pos_sm = meters_trajectory(j2d_map, max_f, H_inv, image_to_court, ref_to_meters)
        speed = np.zeros(max_f)
        speed[1:] = np.linalg.norm(np.diff(pos_sm, axis=0), axis=1) * FPS
        speed = np.convolve(speed, np.ones(7) / 7, mode="same")
        court = (kps, net_img, pos_sm, speed)

    # ---- 渲染 ----
    panel = CourtPanel(640, 1080) if court else None
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = W + (640 if court else 0)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, H))
    t, dist_acc = 0, 0.0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        canvas = np.full((H, out_w, 3), (20, 20, 20), np.uint8)
        if t < max_f:
            k = j2s[t]
            vis = ~np.isnan(k[:, 0])
            if vis.any():
                for a, b in BONES:
                    if vis[a] and vis[b]:
                        cv2.line(img, tuple(k[a].astype(int)), tuple(k[b].astype(int)),
                                 (0, 255, 255), 2)
                for i in range(17):
                    if vis[i]:
                        cv2.circle(img, tuple(k[i].astype(int)), 3, (0, 200, 255), -1)
                jv = k[vis]
                bx0, by0 = jv[:, 0].min() - 12, jv[:, 1].min() - 12
                bx1, by1 = jv[:, 0].max() + 12, jv[:, 1].max() + 12
                cv2.rectangle(img, (int(bx0), int(by0)), (int(bx1), int(by1)),
                              (0, 255, 255), 2)
                lab = labels[frame_pred[t]]
                (tw, th), _ = cv2.getTextSize(lab.upper(), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
                tx = int(max(10, min(bx0, W - tw - 20)))
                ty = int(by0) - 14
                if ty < th + 8:
                    ty = int(by1) + th + 18
                cv2.rectangle(img, (tx - 8, ty - th - 8), (tx + tw + 8, ty + 6), (0, 0, 0), -1)
                cv2.putText(img, lab.upper(), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            1.6, COL[lab], 4, cv2.LINE_AA)
            if court:
                kps, net_img, pos_sm, speed = court
                for a, b in COURT_LINKS:
                    if np.isfinite(kps[a]).all() and np.isfinite(kps[b]).all():
                        cv2.line(img, tuple(kps[a].astype(int)), tuple(kps[b].astype(int)),
                                 (0, 255, 0), 2)
                cv2.line(img, tuple(net_img[0].astype(int)), tuple(net_img[1].astype(int)),
                         (255, 0, 255), 3)
                for i, p in enumerate(kps):
                    if np.isfinite(p).all():
                        cv2.circle(img, tuple(p.astype(int)), 5, (0, 0, 255), -1)
        canvas[:, :W] = img
        if court:
            kps, net_img, pos_sm, speed = court
            panel.add_heat(pos_sm[t])
            dist_acc += float(np.linalg.norm(pos_sm[t] - pos_sm[t - 1])) if t else 0.0
            stats = [
                (f"Dist {dist_acc:.1f} m   Avg {speed[:t+1].mean():.1f}   "
                 f"Max {speed[:t+1].max():.1f} m/s", (255, 255, 0)),
                (f"Cover X {np.nanmin(pos_sm[:t+1, 0]):.1f}~{np.nanmax(pos_sm[:t+1, 0]):.1f} m"
                 f"   Y {np.nanmin(pos_sm[:t+1, 1]):.1f}~{np.nanmax(pos_sm[:t+1, 1]):.1f} m",
                 (230, 230, 230)),
                (f"Speed now {speed[t]:.1f} m/s", (255, 255, 255)),
            ]
            canvas[:, W:] = panel.render(t, pos_sm, stats)
        vw.write(canvas)
        t += 1
    cap.release()
    vw.release()
    print(f"[VIDEO] -> {args.out}")

    # ---- 可选评估 ----
    if args.gt:
        LABEL2IDX = {l: i for i, l in enumerate(labels)}
        ytrue = np.full(max_f, -1)
        for line in open(args.gt):
            parts = line.split()
            if len(parts) < 5:
                continue
            t0, t1, lab = float(parts[1]), float(parts[2]), parts[4].lower().strip()
            if lab not in LABEL2IDX:
                continue
            ytrue[max(int(t0 * FPS), 0):min(int(np.ceil(t1 * FPS)), max_f)] = LABEL2IDX[lab]
        mm = ytrue >= 0
        if mm.sum():
            print(f"\n[EVAL] 帧精度 = {(frame_pred[mm] == ytrue[mm]).mean():.3f}"
                  f" ({int(mm.sum())}/{max_f})")
            for i, l in enumerate(labels):
                sel = mm & (ytrue == i)
                if sel.sum():
                    print(f"   {l:>9} recall={(frame_pred[sel] == i).mean():.3f}"
                          f" (n={int(sel.sum())})")
        else:
            print("[EVAL] 标注与视频无重叠, 跳过")
    if court:
        print(f"[跑动] 总距离={dist_acc:.1f}m 最大速度={speed.max():.2f}m/s")


if __name__ == "__main__":
    main()
