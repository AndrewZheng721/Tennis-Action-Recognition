# 网球击球动作识别 (Tennis Stroke Action Recognition)

对网球视频逐帧识别球员的击球动作：**ready（准备）/ forehand（正手）/ backhand（反手）**，
并输出带大字动作标签 + 3D 骨架叠加的标注视频。可选启用球场关键点检测，
在画面右侧扩展俯视热力图与跑动统计（总距离 / 平均 / 最大速度 / 场地覆盖范围）。

本目录是仓库 `Tennis-Action-Recognition` 的动作识别子模块，俯视图部分复用仓库根目录的
`src/court`（球场关键点检测）与 `src/score`（单应几何），请放置在**仓库根目录下**使用。

## 方法

```
输入视频
  │  ① extract_pose.py  (MediaPipe BlazePose, 纯 CPU)
  ▼
逐帧 COCO17 3D 骨架 (json)
  │  ② extract_features.py  (冻结 Qwen2.5-VL-3B Vision Tower, GPU)
  │     30 帧滑窗 → 每窗按 3D 包围盒归一化 → 渲染双视角骨架图
  │     (左半正面 x,-y / 右半侧面 z,-y, 448×448)
  │     → ViT 逐帧编码 mean-pool → 1280 维特征
  ▼
窗口特征 [N,30,1280]
  │  ③ infer.py  (时序 Transformer × 5 折集成, 3 类)
  │     Frame Projection(1280→128) + 时间位置编码
  │     → TransformerEncoder × 2 (4 头, ff 512, pre-norm)
  │     → global mean pooling → MLP 头 → softmax
  │     → 5 折 softmax 取均值 → 滑窗概率按帧聚合 → 帧级标签
  ▼
标注视频 (骨架 + 人物框上方大字动作标签 [+ 右侧俯视热力图/跑动统计])
```

模型权重：`weights/action_transformer_3cls_fold{0..4}.pt`（5 个，共约 11.6MB，
按训练 clip 分组 5 折交叉验证产出，推理时 softmax 取均值集成；每个含标准化统计
mu/sd 与标签表，加载方式见 `model.py: load_ensemble`）。

**训练数据概况**（仅作模型卡说明，训练代码不在本目录）：7 段固定机位近景网球视频，
MediaPipe 3D 骨架、30 帧窗 stride=5 切窗共 1420 窗
（ready 345 / forehand 718 / backhand 357）。

## 运行环境

| 项目 | 要求 |
| --- | --- |
| GPU | 必须（ViT 编码），显存 ≥ 16GB 推荐；`--chunk 48` 时约 14GB |
| 系统 | Linux（Windows 未测试） |
| 第一步环境 | Python 3.9+，`mediapipe==0.10.9`，`opencv-python>=4.8`，`numpy` |
| 第二/三步环境 | Python 3.10+，`torch>=2.1`（+CUDA），`transformers==4.49.0`，`Pillow` |

> `transformers` 版本请保持 **4.49.0**：ViT 前向直接调用 Qwen2.5-VL 内部结构
> （`visual_tokens_1280`），升级版本可能不兼容。

推荐创建两个 conda 环境（第一步纯 CPU 可与第二三步分开）：

```bash
conda create -n mp python=3.9 -y
conda activate mp
pip install mediapipe==0.10.9 opencv-python numpy

conda create -n qwen python=3.10 -y
conda activate qwen
pip install torch --index-url https://download.pytorch.org/whl/cu130   # 按实际 CUDA 版本
pip install transformers==4.49.0 opencv-python numpy Pillow
```

**Qwen2.5-VL-3B-Instruct 权重**：默认从 HuggingFace 自动下载（首次约 6GB），
或下载到本地目录后所有命令加 `--qwen /本地路径/Qwen2.5-VL-3B-Instruct`。

## 快速开始

在**仓库根目录**下运行（`action_recognition/` 与 `src/`、`weights/` 同级）。

### ① 提取 3D 骨架（CPU，约 5–10 分钟 / 20 分钟视频）

```bash
conda activate mp
# 近景（球员占画面较大）
python action_recognition/extract_pose.py --video in.mp4 --out in_pose.json --mode full
# 高视角（球员在画面下半部）：固定裁剪框 + 2x 上采样
python action_recognition/extract_pose.py --video in.mp4 --out in_pose.json --mode crop \
    --crop 400 1500 520 1080
# 高转播视角（左右半场分块检测，自动选近端球员）
python action_recognition/extract_pose.py --video in.mp4 --out in_pose.json --mode tiles \
    --tiles 250 1000 520 1080,950 1750 520 1080 --feet-y-min 580
```

### ② 提取 ViT 特征（GPU，约 15 分钟 / 20 分钟视频）

```bash
conda activate qwen
python action_recognition/extract_features.py --skeleton in_pose.json --out in_feats.npz
```

### ③ 推理并渲染标注视频（GPU，1–2 分钟）

```bash
# 基础版: 骨架 + 动作标签
python action_recognition/infer.py --video in.mp4 --skeleton in_pose.json --out pred.mp4

# 完整版: 动作识别 + 球场关键点连线 + 右侧俯视热力图/跑动统计
python action_recognition/infer.py --video in.mp4 --skeleton in_pose.json --out pred.mp4 \
    --court-weights weights/court_heatmap.pth

# 附带精度评估 (标注 txt 每行: "xxx 起始秒 结束秒 xxx 标签")
python action_recognition/infer.py --video in.mp4 --skeleton in_pose.json --out pred.mp4 \
    --court-weights weights/court_heatmap.pth --gt in.txt
```

第②步产物 `in_feats.npz` 会被第③步自动复用（`--feats-cache` 指定路径，默认
`<out去后缀>_feats.npz`）；重复运行 / 调渲染参数时无需重新编码特征。

## 输出说明

- **标注视频**（`--out` 指定）：
  - 原画面上叠加 COCO17 骨架（漏检帧自动插值，不闪烁）、人物包围框、
    框上方大字动作标签（ready 黄 / forehand 蓝 / backhand 绿）；
  - `--court-weights` 时画面右侧扩展 640px 栏：1.5 倍放大球场俯视图 + JET 热力
    （球员位置累积）+ 90 帧轨迹尾迹 + 底部大字号跑动统计（总距离 / 平均 / 最大速度 /
    场地覆盖 X-Y 范围），原画面上同时绘制球场关键点连线与球网线。
- **控制台**：帧级精度、逐类召回（给 `--gt` 时）、跑动统计汇总。

## 常用参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--stride` | 1 | 滑窗步长（帧）。调大提速、时间分辨率下降 |
| `--chunk` | 48 | ViT 批大小，显存紧张调小 |
| `--device` | cuda:0 | 推理设备；球场检测同样使用该设备 |
| `--mode` (①) | full | full / crop / tiles，按机位选择 |
| `--min-h` (①) | 200 | full 模式过滤小目标的最小框高（像素） |

## 注意事项

- 机位假设为**固定机位**；切镜/剧烈运镜的视频骨架提取不保证稳定。
- **实测表现**：近景片段帧精度约 0.90（ready 召回 0.82 / 正手 1.00 / 反手 0.95）；
  高位转播视角短片段帧精度约 0.81。高视角下人物占比小，识别难度本身更高，
  务必配合 `crop`/`tiles` 提取模式使用。
- 高视角下 BlazePose 对小目标容易漏检：务必用 `crop`/`tiles` 模式把球员放大 2 倍，
  裁剪框按实际机位调整（默认参数对应 1920×1080 下半部机位）。
- 视频帧率建议 30fps（特征与滑窗均按 30fps 设计）；其他帧率可先转码：
  `ffmpeg -i in.mp4 -vf fps=30 in30.mp4`。
- 球场俯视图精度依赖 `src/court` 关键点检测质量， detection 失败会 assert 报错。
- 本模型无需能量门控（旧模型遗留的 ready 召回问题已通过重训练解决）。

## 目录结构

```
action_recognition/
├── README.md
├── requirements.txt
├── weights/
│   └── action_transformer_3cls_fold{0..4}.pt  # 5 折时序 Transformer 权重 (3 类)
├── extract_pose.py        # ① MediaPipe BlazePose 3D 骨架提取 (CPU)
├── extract_features.py    # ② 冻结 Qwen2.5-VL ViT 特征提取 (GPU)
├── model.py               # 时序 Transformer 定义 + 5 折集成加载
└── infer.py               # ③ 推理 + 标注渲染 (+ 可选球场俯视图)
```
