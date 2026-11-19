import json
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from .match import play_match

MODES = ("yolo26", "rtmpose", "rtmo", "vitpose", "racket")
_RTMPOSE = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)
_RTMO = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
    "rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip"
)
_VITPOSE = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/vitpose-b-coco.onnx"
_CFG = os.path.join(os.path.dirname(__file__), "configs")


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


def _assign_side(center, players):
    best, best_d = None, 1e18
    for p in players:
        b = [float(v) for v in p["bbox"][:4]]
        h = max(1.0, b[3] - b[1])
        cx = (b[0] + b[2]) * 0.5
        cy = b[1] + 0.35 * h
        d = (center[0] - cx) ** 2 + (center[1] - cy) ** 2
        if d < best_d and d <= (1.2 * h) ** 2:
            best, best_d = p["side"], d
    return best


def _ckpt(name):
    dest = os.path.join("weights", "racket", name)
    if os.path.isfile(dest) and os.path.getsize(dest) > 1000:
        return os.path.abspath(dest)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError("缺少 huggingface_hub。先执行 pip install huggingface_hub") from e
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    cached = hf_hub_download("linfeng302/RacketVision-Models", f"checkpoints/{name}", repo_type="model")
    shutil.copy2(cached, dest)
    return os.path.abspath(dest)


def _load_racket():
    try:
        import torch
        from mmdet.apis import inference_detector, init_detector
        from mmpose.apis import inference_topdown, init_model
    except ImportError as e:
        raise RuntimeError(
            '球拍模型需要 mmdet 和 mmpose。先执行 pip install -U openmim huggingface_hub && '
            'mim install "mmengine>=0.10.7" "mmcv>=2.0.0rc4,<2.2.0" "mmdet>=3.0.0,<3.3.0" "mmpose>=1.1.0"'
        ) from e
    orig = torch.load

    def _load(*args, **kwargs):
        kwargs["weights_only"] = False
        return orig(*args, **kwargs)

    torch.load = _load
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        det = init_detector(
            os.path.join(_CFG, "rtmdet_m_racket_infer.py"),
            _ckpt("epoch_300.pth"),
            palette="coco",
            device=device,
        )
        pose = init_model(os.path.join(_CFG, "rtmpose_m_racket_infer.py"), _ckpt("best_PCK_epoch_90.pth"), device=device)
        det.eval()
        pose.eval()
    finally:
        torch.load = orig

    def run(img):
        from mmengine.registry import init_default_scope

        init_default_scope("mmdet")
        det_result = inference_detector(det, img)
        inst = det_result.pred_instances
        bboxes = inst.bboxes.detach().cpu().numpy()
        scores = inst.scores.detach().cpu().numpy()
        labels = inst.labels.detach().cpu().numpy()
        keep = (labels == 2) & (scores >= 0.3)
        bboxes = bboxes[keep]
        scores = scores[keep]
        if len(bboxes) == 0:
            return []
        order = np.argsort(-scores)[:4]
        bboxes = bboxes[order]
        pose_results = inference_topdown(pose, img, bboxes)
        out = []
        for i, pred in enumerate(pose_results):
            kpts = pred.pred_instances.keypoints
            ks = pred.pred_instances.keypoint_scores
            if hasattr(kpts, "detach"):
                kpts = kpts.detach().cpu().numpy()
                ks = ks.detach().cpu().numpy()
            kpts = np.asarray(kpts)
            ks = np.asarray(ks)
            if kpts.ndim == 3:
                kpts = kpts[0]
                ks = ks[0]
            out.append(
                {
                    "bbox": [float(v) for v in bboxes[i]],
                    "keypoints": kpts,
                    "scores": ks,
                }
            )
        return out

    return run


def _mark_racket(flags, poses, fid, uv, players, rackets):
    poses["racket"][fid] = {}
    by_side = {"near": [], "far": []}
    for r in rackets:
        bb = r["bbox"]
        side = _assign_side(((bb[0] + bb[2]) * 0.5, (bb[1] + bb[3]) * 0.5), players)
        if side:
            by_side[side].append(r)
    for side, rs in by_side.items():
        best, best_d = None, 1e18
        for r in rs:
            pts = []
            for i, xy in enumerate(r["keypoints"]):
                if float(r["scores"][i]) >= 0.3:
                    pts.append((float(xy[0]), float(xy[1])))
            if not pts:
                b = r["bbox"]
                pts = [((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)]
            d = min((x - uv[0]) ** 2 + (y - uv[1]) ** 2 for x, y in pts)
            if d < best_d:
                best, best_d = (pts, d), d
        if best is None:
            continue
        pts, d = best
        span = max(
            max(p[0] for p in pts) - min(p[0] for p in pts),
            max(p[1] for p in pts) - min(p[1] for p in pts),
            12.0,
        )
        near = d <= max(20.0, 0.55 * span) ** 2
        poses["racket"][fid][side] = [
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
            near,
            pts,
        ]
        if near:
            flags["racket"][side].add(fid)


def collect_contact(cap, filled, players):
    try:
        yolo = _load_yolo()
        rtmpose = _load_rtmpose()
        rtmo = _load_rtmo()
        vitpose = _load_vitpose()
    except ImportError as e:
        raise RuntimeError("缺少 rtmlib。先执行 pip install rtmlib onnxruntime-gpu") from e
    racket = _load_racket()
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
        _mark_racket(flags, poses, fid, uv, pls, racket(frame))
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
