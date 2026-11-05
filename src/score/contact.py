import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from .match import play_match

MODES = ("yolo26", "rtmpose", "rtmo", "vitpose")
_RTMPOSE = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)
_RTMO = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
    "rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip"
)
_VITPOSE = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/vitpose-b-coco.onnx"


def _ball_uv(item):
    box = (item or {}).get(1)
    if not box or len(box) < 4:
        return None
    return (float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5


def _players(frame):
    out = []
    for p in frame or []:
        b = p.get("bbox") or []
        if p.get("side") in ("near", "far") and len(b) >= 4:
            out.append(p)
    out.sort(key=lambda p: (float(p["bbox"][2]) - float(p["bbox"][0])) * (float(p["bbox"][3]) - float(p["bbox"][1])), reverse=True)
    return out


def _device():
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _as_people(kpts, scores):
    kpts = np.asarray(kpts)
    scores = np.asarray(scores)
    if kpts.size == 0:
        return np.zeros((0, 17, 2)), np.zeros((0, 17))
    if kpts.ndim == 2:
        kpts = kpts[None]
        scores = scores[None]
    return kpts, scores


def _match(kpts, scores, players):
    kpts, scores = _as_people(kpts, scores)
    used = set()
    out = {}
    for p in players:
        b = [float(v) for v in p["bbox"][:4]]
        h = max(1.0, b[3] - b[1])
        pcx = (b[0] + b[2]) * 0.5
        pcy = (b[1] + b[3]) * 0.5
        best_i, best_d = None, 1e18
        for i in range(len(kpts)):
            if i in used:
                continue
            vis = scores[i] >= 0.3
            if not np.any(vis):
                continue
            cx = float(np.mean(kpts[i, vis, 0]))
            cy = float(np.mean(kpts[i, vis, 1]))
            d = (cx - pcx) ** 2 + (cy - pcy) ** 2
            if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                d *= 0.01
            if d < best_d and d <= (1.5 * h) ** 2:
                best_i, best_d = i, d
        if best_i is not None:
            used.add(best_i)
            out[p["side"]] = (kpts[best_i], scores[best_i])
    return out


def _load_rtmpose():
    from rtmlib import RTMPose

    kw = dict(onnx_model=_RTMPOSE, model_input_size=(192, 256), to_openpose=False, backend="onnxruntime")
    device = _device()
    try:
        return RTMPose(device=device, **kw)
    except Exception:
        if device == "cpu":
            raise
        return RTMPose(device="cpu", **kw)


def _load_rtmo():
    from rtmlib import RTMO

    kw = dict(onnx_model=_RTMO, model_input_size=(640, 640), to_openpose=False, backend="onnxruntime", score_thr=0.3)
    device = _device()
    try:
        return RTMO(device=device, **kw)
    except Exception:
        if device == "cpu":
            raise
        return RTMO(device="cpu", **kw)


def _load_vitpose():
    from rtmlib import ViTPose

    kw = dict(onnx_model=_VITPOSE, model_input_size=(192, 256), to_openpose=False, backend="onnxruntime")
    device = _device()
    try:
        return ViTPose(device=device, **kw)
    except Exception:
        if device == "cpu":
            raise
        return ViTPose(device="cpu", **kw)


def _load_yolo():
    from ultralytics import YOLO

    path = "weights/yolo26m-pose.pt" if os.path.isfile("weights/yolo26m-pose.pt") else "yolo26m-pose.pt"
    return YOLO(path)


def _yolo_people(model, frame):
    result = model.predict(frame, verbose=False, conf=0.25, imgsz=1280)[0]
    if result.keypoints is None or result.keypoints.xy is None:
        return np.zeros((0, 17, 2)), np.zeros((0, 17))
    xy = result.keypoints.xy.detach().cpu().numpy()
    conf = result.keypoints.conf
    scores = np.ones(xy.shape[:2], dtype=np.float32) if conf is None else conf.detach().cpu().numpy()
    return xy, scores


def _take_wrist(uv, p, kpts, scores):
    h = max(1.0, float(p["bbox"][3]) - float(p["bbox"][1]))
    thr = max(24.0, 0.45 * h)
    best, best_d = None, 1e18
    for idx in (9, 10):
        if idx >= len(scores) or float(scores[idx]) < 0.3:
            continue
        x, y = float(kpts[idx][0]), float(kpts[idx][1])
        d = (x - uv[0]) ** 2 + (y - uv[1]) ** 2
        if d < best_d:
            best, best_d = (x, y), d
    if best is None:
        return None
    return [best[0], best[1], best_d <= thr * thr]


def _mark(flags, poses, mode, fid, uv, players, paired):
    poses[mode][fid] = {}
    for p in players:
        item = paired.get(p["side"])
        if not item:
            continue
        wrist = _take_wrist(uv, p, item[0], item[1])
        if wrist is None:
            continue
        poses[mode][fid][p["side"]] = wrist
        if wrist[2]:
            flags[mode][p["side"]].add(fid)


def collect_contact(cap, filled, players):
    try:
        yolo = _load_yolo()
        rtmpose = _load_rtmpose()
        rtmo = _load_rtmo()
        vitpose = _load_vitpose()
    except ImportError as e:
        raise RuntimeError("缺少 rtmlib。先执行 pip install rtmlib onnxruntime-gpu") from e
    flags = {m: {"near": set(), "far": set()} for m in MODES}
    poses = {m: {} for m in MODES}
    n = min(len(filled), len(players))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for fid in tqdm(range(n), desc="pose"):
        ok, frame = cap.read()
        if not ok:
            break
        uv = _ball_uv(filled[fid])
        pls = _players(players[fid])
        if uv is None or not pls:
            continue
        boxes = [p["bbox"][:4] for p in pls]
        yk, ys = _yolo_people(yolo, frame)
        _mark(flags, poses, "yolo26", fid, uv, pls, _match(yk, ys, pls))
        rk, rs = _as_people(*rtmpose(frame, boxes))
        _mark(flags, poses, "rtmpose", fid, uv, pls, {p["side"]: (rk[i], rs[i]) for i, p in enumerate(pls) if i < len(rk)})
        _mark(flags, poses, "rtmo", fid, uv, pls, _match(*rtmo(frame), pls))
        vk, vs = _as_people(*vitpose(frame, boxes))
        _mark(flags, poses, "vitpose", fid, uv, pls, {p["side"]: (vk[i], vs[i]) for i, p in enumerate(pls) if i < len(vk)})
    return flags, poses


def run_ablation(cap, filled, players, bounces, hits, server, confirm, n_frames, out_dir):
    if not players:
        raise RuntimeError("四组对照需要人体框，不要加 --skip-pose")
    flags, poses = collect_contact(cap, filled, players)
    rows = {}
    for mode in MODES:
        got = flags[mode]

        def played(frame_id, side, frames=got):
            return side in frames and frame_id in frames[side]

        match = play_match(
            bounces,
            hits,
            server=server,
            confirm=confirm,
            n_frames=n_frames,
            played=played,
        )
        rows[mode] = {
            "contact_frames": {s: len(got[s]) for s in ("near", "far")},
            "points": len(match["points"]),
            "final": match["final"],
            "reasons": [p["reason"] for p in match["points"]],
            "match": match,
        }
        print(
            mode,
            "near/far",
            rows[mode]["contact_frames"]["near"],
            rows[mode]["contact_frames"]["far"],
            "points",
            rows[mode]["points"],
            rows[mode]["final"],
        )
    payload = {"server": server, "confirm": confirm, "modes": rows}
    path = os.path.join(out_dir, "contact_ablation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("ablation:", path)
    return payload, poses
