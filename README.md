# Tennis-3D-Demo

高位网球转播的球场、球轨迹、落地界内界外。测试入口是 `test_court_ball.py`。`tennis_high_angle_01.mp4` 只做测试，不进训练。

测试方法：
python test_court_ball.py --video high_angle_dataset/tennis_high_angle_01.mp4 --out outputs/ft_01 --ball-weights weights/tracknet_ft.pth


权重：

| 文件 | 作用 |
| --- | --- |
| `weights/tracknet_ft.pth` | 微调后的球轨迹，测试时用 `--ball-weights` 指定 |
| `weights/tracknet.pth` | 微调前的 TrackNet，训练起点 |
| `weights/eventnet.pt` | 击球 / 落地 / 飞行 |
| `weights/court_heatmap.pth` | 球场 14 个关键点 |
| `weights/yolo26m.pt` | 球员框 |

## 共用数据结构

球的一帧检测结果是字典。键 `1` 表示球，值是 `[x1, y1, x2, y2, conf]`。没有球就是 `{}`。一条轨迹是按帧排列的这种字典，记为 `filled`。

`CourtDetection`：14 个关键点 `keypoints_xy`（14×2）、参考场地到图像的单应矩阵、`quality_ok`、`method`、`frame_id`。

事件标签 `status`：`0` 飞行，`1` 击球，`2` 落地。球心标签 `vis`：`1` 看得见，`0` 看不见。

球场坐标以球场中心为原点，y 正向是远端。`classify_xy` 返回：

| 字段 | 取值 |
| --- | --- |
| `side` | `far` 远端，`near` 近端 |
| `serve_zone` | `far_left` `far_right` `near_left` `near_right`，不在发球区则为空 |
| `rally_zone` | `far_in` `near_in` `out` |
| `in_singles` | 单打界内，压线算界内 |
| `in_doubles` | 双打边线内 |

## 项目目录下：

### 测试总入口 `test_court_ball.py`

对一段视频出球场、轨迹、落地界内界外和叠加视频。

- `parse_args`：读命令行。`--ball-weights` 指定球权重，`--out` 是输出目录。
- `main`：检测球场和球，落地帧重测球场，调用 `analyze_score`，写视频。

输出：

- `overlay.mp4`：球场线、球轨迹、落点。绿色落点是界内，红色是界外。
- `ball_tracking.json`：每帧 `frame_id`、`center`、`bbox`、`confidence`。
- `court_keypoints.json`：`CourtDetection.to_json()`。
- `score.json`：`bounces` 列表。每项有 `frame_id`、`image_xy`、`xyz`，以及 `classify_xy` 的字段。
- `ball_3d.json`、`overlay_3d.mp4`：球的三维轨迹。

### 训练准备脚本 `prepare_events.py`

从高位视频抽出球轨迹和球员框，写成 EventNet 的训练片段。跳过文件名里带 `high_angle_01` 的视频。

- `_segments`：把球场检测成功的帧切成片段。间隔不超过 12 帧的段会合并，短于 45 帧的段丢掉。
- `_load_marks`：读 `labels/<视频名>.csv` 的 `frame,status`。
- `_write_clip`：写一个片段。
- `_process_video`：逐帧跟踪球场和球。球场失败的帧不保留球点。
- `parse_args` / `main`：`--only` 只处理文件名包含该字符串的视频。

每个 `high_angle_dataset/events/<视频名>/clip_xxxx/`：

- `Label.csv`：`file name, visibility, x-coordinate, y-coordinate, status`
- `players.json`：`frames`，每项是 `frame_id`、`src_frame`（原视频帧号）、`players`（框和 `side`）
- `candidates.txt`：几何法猜的落地帧，只供标注时参考

### 击球/落地事件标注脚本 `label_events.py`

看视频标注击球和落地。

- `_load_marks` / `_marks_from_clips`：读已有 `status`。
- `_load_tracks`：从 `events` 读球点和球员框。
- `_save` / `_sync_clips`：写入 `high_angle_dataset/labels/<视频名>.csv`，并回写已有片段的 `Label.csv`。
- `_draw` / `run`：播放和按键。

按键：空格播放，`a` `d` 一帧，`q` `e` 十帧，`1` 击球，`2` 落地，`0` 清除，`n` `p` 跳到已标事件，`Esc` 保存。两帧之间的接触只标后一帧。

### 球心位置标注脚本 `label_ball.py`

改正球心。只把改过的帧写入 `high_angle_dataset/labels_ball/<视频名>.csv`，列是 `frame,x,y,vis`。没改的帧继续用 `events` 里的自动轨迹。

- `_load_auto`：读自动球心，`{帧号: (x, y)}`。
- `_load_fixes` / `_save`：读写人工修正，`{帧号: (x, y, vis)}`。
- `_at`：这一帧的球心。`vis<=0` 表示没有球。
- `_jumps`：相邻帧位移超过 100 像素的帧号。
- `_draw` / `_click_xy` / `_load_frame` / `run`：显示、点击和缓存附近帧。

按键：左键点球心，`0` 没有球，`u` 撤销，`[` `]` 把一段全部标成没有球，`n` `p` 跳到大跳，`Esc` 保存。

### 训练数据准备 `export_ball.py`

把 `labels_ball` 转成 TrackNet 训练数据。

- `_load_fixes` / `_load_events`：读球心修正和击球落地标签。
- `_smooth`：连续可见的飞行帧做前后各 1 帧中值。击球帧和落地帧不动。
- `_grab`：从视频抽出需要的帧，缩到 640×360。
- `main`：写出 `dataset/tracknet_ball/`。

`labels_train.csv` 和 `labels_val.csv` 的列：`path1,path2,path3,x,y,vis`。`path1` 是当前帧，`path2`、`path3` 是前两帧。`x,y` 在 640×360 上。`--only` 可只导出一个视频。

### 微调TrackNet V1 `train_tracknet.py`

用上面的 CSV 微调 TrackNet V1。

- `TrackNetCsv`：读三帧，拼成 9 通道输入。`vis!=0` 时在 `(x,y)` 放高斯热力图，否则热力图全 0。
- `parse_args` / `main`：从 `--weights` 加载，验证损失下降时写入 `--out`。

### 训练“击球/落地”事件判断模型 `train_eventnet.py`

用 `events` 片段训练落地和击球分类。`status` 里没有击球或落地时直接退出。

- `_find_data`：默认数据目录 `high_angle_dataset/events`。
- `_scores`：击球和落地的精确率、召回率、F1。
- `main`：按比赛名划分验证集，保存 `weights/eventnet.pt`。权重字典含 `state`、`in_dim`、`hidden`、`layers`、`window`。

## `src/ball` 目录下：主要做球检测和轨迹。

`tracknet_models.py`

- `TrackNetV1`：三帧 640×360，输出 256 类热力图。
- `TrackNetV4`、`MotionPrompt`、`FusionTypeA`：V4 结构，当前测试用的是 V1。

`tracknet_tracker.py`

- `_load_state`：读 `.pth`。
- `detect_tracknet_kind`：用权重键名区分 V1 和 V4。
- `decode_v1_from_argmax` / `decode_float_heatmap`：热力图峰值变成原图坐标。
- `TrackNetBallTracker`：维护最近三帧，`detect_frame` 返回 `{1: [x1,y1,x2,y2,conf]}` 或 `{}`。`set_court` 丢掉扩大场地外的点。
- `create_ball_tracker`：按权重文件名选择 TrackNet 或 YOLO。
- `default_ball_weights`：依次查找 `tracknet_v4.pth`、`tracknet.pth`、`yolo5_last.pt`。微调权重不在这个列表里。

`ball_tracker.py`

- `BallTracker`：YOLO 检测、短缺口插值、大跳断开。`interpolate_ball_positions`、`get_ball_shot_frames`、`to_json`、`draw_frame` 也被 TrackNet 跟踪器使用。
- `to_json` 的一帧：`frame_id`、`time_sec`、`bbox`、`center`、`confidence`。

## `src/court` 目录下：球场14点定位

`heatmap_net.py`：`CourtHeatmapNet`，输出 14 个关键点热力图。

`heatmap_court.py`

- `HeatmapCourtDetector.predict`：图像到 14 个点，再配准到标准场地。
- `heatmap_peak`：单个热力图的峰值。
- `snap_with_partial_points`：用找得到的点估计单应矩阵，补全其余点。

`geometric_court.py`

- `detect_geometric`：用场地颜色和四边形找球场，热力图失败时的备用。
- `adaptive_surface_mask`、`quad_from_mask`、`order_quad`。

`court_line_detector.py`

- `CourtLineDetector.predict`：先热力图，再几何，再 ResNet。`fast=True` 时只用热力图。
- `track_frame`：隔若干帧重测，中间用 ECC 跟着镜头。
- `ball_on_singles_line`：球在模型线和真实白线之间，或距白线不超过 16 像素，则视为压线。
- `align_keypoints_to_white_lines`：把关键点平移到白线上，平移不超过 30 像素。
- `read_frame_at`、`choose_sample_ids`：抽帧读视频。

## `src/score` 目录下：判断落点和界内界外。

`geom.py`

- `ball_center`：框的中心。
- `image_to_court`：用单应矩阵把像素变到参考场地。
- `classify_bounce`：参考场地坐标上的界内界外。

`court3d.py`

- `estimate_camera`：由 14 个关键点估计相机，返回 `K`、`R`、`t`。
- `ray_at_z`：像素射线与高度 `z` 的交点。落地用 `z=0`。
- `classify_xy`：三维落点的发球区和单打界内界外。调用处使用 `eps=0.35` 米，压线算界内。

`eventnet.py`

- `clip_arrays`：一个片段变成特征 `(帧数, 12)` 和 `status`。12 维是归一化的球 `x,y,dx,dy,vis`、近端和远端球员脚点、球到两人的距离、是否有球员。
- `windows`：以当前帧为中心的 21 帧窗口。
- `build_split`：按比赛目录划分。目录名含 `tennis_high_angle_01` 的片段不进训练集。
- `EventNet`：双向 GRU，三类。
- `load_eventnet` / `predict_status`：加载权重并逐帧预测。

`bounce.py`

- `_centers`：从 `filled` 取出球心序列。
- `_geom_bounces`：用画面 y 方向折返猜落地帧。
- `detect_bounces`：没有 EventNet 时使用。优先 CatBoost 回归器 `ctb_regr_bounce.cbm`。

`pipeline.py`

- `analyze_score`：有 `weights/eventnet.pt` 时，落地帧只来自 `status==2`。否则走 `detect_bounces`，并去掉贴着球拍、同一侧连续下落的点。
- 每个落地点要能投影到 `z=0`，且落在场地范围附近。
- 返回 `{"fps", "num_bounces", "bounces"}`。

`players.py`

- `PlayerTracker.detect`：YOLO 人体框，分成 `far` 和 `near`。每人是 `bbox`、`side`、`track_id`。
- `lift_players`：脚点经相机射线落到 `z=0`，写入 `xyz`。
- `draw_players`：画框。

`traj3d.py`

- `reconstruct_ball_3d`：用落地帧把球投影到地面，帧间用重力拟合高度。返回每帧 `frame_id`、`uv`、`xyz` 和 `bounce_frames`。

`viz3d.py`

- `write_traj3d_video`：把 `xyz` 画成三维视频。

`cntext.py`

- `zone_cn`：`far_in` 等代号换成中文。
- `put_cn`：用系统中文字体写到画面上。
