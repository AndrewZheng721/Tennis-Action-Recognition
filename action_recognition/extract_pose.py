#!/usr/bin/env python3
"""
第一步: MediaPipe BlazePose 3D 骨架提取 (纯 CPU)

逐帧静态检测 (不做 tracking, 因此切镜/多人场景不会跟丢或跟错),
输出统一 JSON schema:
  {"fps", "width", "height", "ground_y", "source",
   "person_0": [{"frame_id", "joints2d" [17,2] 像素, "joints3d" [17,3] (x,y像素取负,z米),
                 "bbox" [x0,y0,x1,y1]}, ...]}

三种模式 (按机位/球员占比选择):
  full  全幅图直接检测, 带 sanity 过滤 (bbox 高度>=--min-h 且在画面内)
        适用于球员占画面较大的近景
  crop  固定裁剪框 + 2x 上采样后检测, 坐标换算回原图
        适用于高视角且球员集中在画面某一区域 (如下半部)
  tiles 画面下半部左右两块 tile 各自 2x 上采样检测, 选 feet_y>=--feet-y-min
        中 bbox 最高的作为近端球员; 适用于高转播视角

用法:
  python extract_pose.py --video in.mp4 --out in.json --mode full
  python extract_pose.py --video in.mp4 --out in.json --mode crop --crop 400 1500 520 1080
  python extract_pose.py --video in.mp4 --out in.json --mode tiles \
      --tiles 250 1000 520 1080,950 1750 520 1080 --feet-y-min 580
"""
import os
import json
import argparse
import numpy as np
import cv2
import mediapipe as mp

MP33_TO_COCO17 = {
    0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12, 7: 13, 8: 14,
    9: 15, 10: 16, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
}
VIS_THRESHOLD = 0.3


def parse_tiles(s):
    """'x0 x1 y0 y1,x0 x1 y0 y1' -> [(x0,x1,y0,y1), ...]"""
    tiles = []
    for part in s.split(","):
        v = [float(x) for x in part.split()]
        assert len(v) == 4, f"--tiles 格式错误: {s}"
        tiles.append(tuple(v))
    return tiles


def lm_to_coco17(lm, W, H, x_off=0.0, y_off=0.0):
    """BlazePose 33 点 -> COCO17; 像素坐标 (x_off/y_off 为 tile 在原图中的偏移)"""
    j2 = np.zeros((17, 2), np.float32)
    j3 = np.zeros((17, 3), np.float32)
    for coco, mpi in MP33_TO_COCO17.items():
        p = lm[mpi]
        if p.visibility < VIS_THRESHOLD:
            continue
        j2[coco] = (x_off + p.x * W, y_off + p.y * H)
        j3[coco] = (x_off + p.x * W, -(y_off + p.y * H), p.z)
    return j2, j3


def bbox_of(j2):
    vis = j2[:, 0] > 0
    if not vis.any():
        return None
    return (float(j2[vis, 0].min()), float(j2[vis, 1].min()),
            float(j2[vis, 0].max()), float(j2[vis, 1].max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="", help="输出 json 路径, 默认 <video去后缀>_pose.json")
    ap.add_argument("--mode", choices=["full", "crop", "tiles"], default="full")
    ap.add_argument("--min-h", type=float, default=200,
                    help="full 模式: bbox 最小高度 (像素), 过滤远处小人/路人")
    ap.add_argument("--crop", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"),
                    default=None, help="crop 模式裁剪框 (原图像素)")
    ap.add_argument("--tiles", type=str, default="",
                    help="tiles 模式 tile 列表, 如 '250 1000 520 1080,950 1750 520 1080'")
    ap.add_argument("--feet-y-min", type=float, default=580,
                    help="tiles 模式: 近端球员脚底最小 y (像素)")
    args = ap.parse_args()
    out_path = args.out or os.path.splitext(args.video)[0] + "_pose.json"

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"打不开视频: {args.video}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.mode == "crop":
        assert args.crop is not None, "crop 模式需要 --crop X0 X1 Y0 Y1"
        regions = [tuple(args.crop)]
    elif args.mode == "tiles":
        assert args.tiles, "tiles 模式需要 --tiles 'x0 x1 y0 y1,x0 x1 y0 y1'"
        regions = parse_tiles(args.tiles)
    else:
        regions = [(0.0, float(W), 0.0, float(H))]

    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1,
                                  min_detection_confidence=0.3)
    frames_out = []
    fid = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        best = None  # tiles 模式: (bbox_h, j2, j3)
        for (x0, x1, y0, y1) in regions:
            crop = img[int(y0):int(y1), int(x0):int(x1)]
            if crop.size == 0:
                continue
            up = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2)) \
                if args.mode in ("crop", "tiles") else crop
            r = pose.process(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
            if r.pose_landmarks is None:
                continue
            rw = (x1 - x0) if args.mode in ("crop", "tiles") else W
            rh = (y1 - y0) if args.mode in ("crop", "tiles") else H
            j2, j3 = lm_to_coco17(r.pose_landmarks.landmark, rw, rh, x0, y0)
            if np.abs(j3).sum() == 0:
                continue
            if args.mode == "tiles":
                bb = bbox_of(j2)
                if bb is None or bb[3] < args.feet_y_min:
                    continue
                hbox = bb[3] - bb[1]
                if best is None or hbox > best[0]:
                    best = (hbox, j2, j3)
            elif args.mode == "full":
                bb = bbox_of(j2)
                if bb is None:
                    continue
                if bb[3] <= H + 20 and (bb[3] - bb[1]) >= args.min_h \
                        and bb[0] >= -20 and bb[2] <= W + 20:
                    best = (0, j2, j3)
            else:  # crop
                best = (0, j2, j3)
                break  # 单个裁剪框, 取首个检出即可
        if best is not None:
            j2, j3 = best[1], best[2]
            frames_out.append({"frame_id": fid,
                               "joints2d": np.round(j2, 2).tolist(),
                               "joints3d": np.round(j3, 4).tolist(),
                               "bbox": bbox_of(j2) or [0, 0, 0, 0]})
        fid += 1
        if fid % 1000 == 0:
            print(f"[POSE] {fid} 帧, 检出 {len(frames_out)}", flush=True)
    cap.release()
    pose.close()

    data = {"fps": fps, "width": W, "height": H, "ground_y": 0,
            "source": f"mediapipe_blazepose_static_{args.mode}",
            "person_0": frames_out}
    with open(out_path, "w") as f:
        json.dump(data, f)
    print(f"[OK] {len(frames_out)}/{fid} 帧有检出 -> {out_path}")


if __name__ == "__main__":
    main()
