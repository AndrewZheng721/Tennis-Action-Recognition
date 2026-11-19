import cv2
import numpy as np
from tqdm import tqdm

_VITPOSE = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/vitpose-b-coco.onnx"
LABELS = ("ready", "forehand", "backhand", "serve", "volley")
CN = {
    "ready": "准备",
    "forehand": "正手",
    "backhand": "反手",
    "serve": "发球",
    "volley": "截击",
}
_BONES = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)


def _device():
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


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


def _players(frame):
    out = []
    for p in frame or []:
        b = p.get("bbox") or []
        if p.get("side") in ("near", "far") and len(b) >= 4:
            out.append(p)
    return out


def _vis(scores, i):
    return scores is not None and i < len(scores) and float(scores[i]) >= 0.3


def _unit(v):
    n = float(np.hypot(v[0], v[1]))
    if n < 1e-3:
        return None
    return v / n


def _body(k, s):
    if not (_vis(s, 5) and _vis(s, 6)):
        return None
    ls, rs = k[5], k[6]
    mid = (ls + rs) * 0.5
    if _vis(s, 11) and _vis(s, 12):
        hip = (k[11] + k[12]) * 0.5
    else:
        hip = mid + np.array([0.0, np.linalg.norm(rs - ls) * 1.5], dtype=np.float32)
    torso = max(float(np.linalg.norm(hip - mid)), 12.0)
    hat = _unit(rs - ls)
    if hat is None:
        return None
    body = {"torso": torso, "mid": mid, "hat": hat, "sw": max(float(np.linalg.norm(rs - ls)), 8.0)}
    body["nose_y"] = float(k[0, 1]) if _vis(s, 0) else float(mid[1]) - torso * 0.4
    for name, i in (("lw", 9), ("rw", 10), ("le", 7), ("re", 8)):
        body[name] = k[i] if _vis(s, i) else None
    return body


def _wrist_pts(seq, key):
    return [b[key] for b in seq if b is not None and b.get(key) is not None]


def _span(pts):
    if len(pts) < 2:
        return 0.0
    arr = np.stack([np.asarray(p, dtype=np.float32) for p in pts])
    c = arr.mean(axis=0)
    return float(np.max(np.linalg.norm(arr - c, axis=1)) * 2.0)


def _bent(body, side):
    elbow = body["le"] if side == "l" else body["re"]
    wrist = body["lw"] if side == "l" else body["rw"]
    if elbow is None or wrist is None:
        return False
    sh = body["mid"]
    upper = float(np.linalg.norm(elbow - sh))
    fore = float(np.linalg.norm(wrist - elbow))
    direct = float(np.linalg.norm(wrist - sh))
    if upper + fore < 1e-3:
        return False
    return direct / (upper + fore) < 0.82


def _raw(seq):
    bodies = [b for b in seq if b is not None]
    if not bodies:
        return "ready"
    center = bodies[len(bodies) // 2]
    lw, rw = _span(_wrist_pts(seq, "lw")), _span(_wrist_pts(seq, "rw"))
    use_l = lw >= rw
    key = "lw" if use_l else "rw"
    pts = _wrist_pts(seq, key)
    if not pts:
        return "ready"
    best_i, best_a = 0, -1.0
    for i, b in enumerate(bodies):
        w = b.get(key)
        if w is None:
            continue
        a = abs(float(np.dot(w - b["mid"], b["hat"])))
        if a > best_a:
            best_i, best_a = i, a
    b = bodies[best_i]
    w = b.get(key)
    if w is None:
        w = center.get(key)
    if w is None:
        return "ready"
    torso = b["torso"]
    sw = b["sw"]
    side = float(np.dot(np.asarray(w) - b["mid"], b["hat"])) / torso
    height = (float(b["mid"][1]) - float(w[1])) / torso
    above_nose = float(b["nose_y"]) - float(w[1]) > 0.15 * torso
    travel = max(lw, rw)
    if height > 0.75 and above_nose:
        return "serve"
    if travel > 0.65 * sw and abs(side) > 0.15:
        return "forehand" if side > 0 else "backhand"
    if travel < 0.45 * sw and -0.35 < height < 0.55 and _bent(b, "l" if use_l else "r"):
        return "volley"
    return "ready"


def _window(series, fid, side, radius=8):
    seq = []
    for t in range(fid - radius, fid + radius + 1):
        if t < 0 or t >= len(series):
            seq.append(None)
            continue
        item = series[t].get(side)
        seq.append(_body(*item[:2]) if item is not None else None)
    return _raw(seq)


def _smooth(raw, radius=3):
    out = []
    for i in range(len(raw)):
        if raw[i] is None:
            out.append(None)
            continue
        chunk = [x for x in raw[max(0, i - radius) : i + radius + 1] if x]
        if not chunk:
            out.append("ready")
            continue
        strokes = [x for x in chunk if x != "ready"]
        if len(strokes) >= 3:
            out.append(max(set(strokes), key=strokes.count))
        else:
            out.append(max(set(chunk), key=chunk.count))
    return out


def run_actions(cap, players):
    vit = _load_vitpose()
    n = len(players)
    series = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for fid in tqdm(range(n), desc="vitpose"):
        ok, frame = cap.read()
        if not ok:
            break
        pls = _players(players[fid])
        item = {}
        if pls:
            boxes = np.asarray([p["bbox"][:4] for p in pls], dtype=np.float32)
            kpts, scores = vit(frame, boxes)
            kpts = np.asarray(kpts)
            scores = np.squeeze(np.asarray(scores))
            if kpts.ndim == 2:
                kpts = kpts[None]
            if scores.ndim == 1:
                scores = scores[None]
            for i, p in enumerate(pls):
                if i >= len(kpts):
                    break
                item[p["side"]] = (kpts[i], scores[i])
        series.append(item)
    labels = []
    for side in ("near", "far"):
        raw = [_window(series, i, side) if side in series[i] else None for i in range(len(series))]
        labels.append(_smooth(raw))
    frames = []
    for i in range(len(series)):
        row = {}
        for si, side in enumerate(("near", "far")):
            if labels[si][i]:
                row[side] = labels[si][i]
        frames.append(row)
    return frames, series


def draw_actions(img, kpts_frame, labels):
    from .cntext import put_cn

    if kpts_frame:
        for k, _s in kpts_frame.values():
            for a, b in _BONES:
                if a >= len(k) or b >= len(k):
                    continue
                if float(k[a, 0]) <= 1 or float(k[b, 0]) <= 1:
                    continue
                cv2.line(img, (int(k[a, 0]), int(k[a, 1])), (int(k[b, 0]), int(k[b, 1])), (255, 180, 0), 2)
    bits = []
    for side, name in (("near", "近端"), ("far", "远端")):
        lab = (labels or {}).get(side)
        if lab:
            bits.append(f"{name}{CN.get(lab, lab)}")
    if bits:
        put_cn(img, "  ".join(bits), (20, 190), (255, 180, 0), 0.8, 2)
    return img
