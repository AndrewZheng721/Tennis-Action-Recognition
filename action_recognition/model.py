#!/usr/bin/env python3
"""
时序动作识别模型: TemporalActionTransformer (3 类: ready / forehand / backhand)

  feats [N,30,1280] (冻结 ViT 逐帧特征, 见 extract_features.py)
    -> Frame Projection Linear(1280->d_model)
    -> + learnable time positional encoding
    -> TransformerEncoder x n_layers (pre-norm)
    -> LayerNorm + global mean pooling
    -> MLP head (d_model->128->n_cls)

部署形态为 5 折集成 (按训练 clip 分组交叉验证的 5 个模型, softmax 取均值):
  load_ensemble() 一次加载全部折; predict_ensemble_probs() 做集成推理。
每个 checkpoint 内含: state_dict / 训练折标准化统计 mu / sd / 标签表 labels。
"""
import numpy as np
import torch
import torch.nn as nn

LABELS = ["ready", "forehand", "backhand"]  # 默认 3 类, 以 ckpt 内 labels 为准


class TemporalActionTransformer(nn.Module):
    def __init__(self, dim_in=1280, T=30, d_model=128, n_layers=2, n_heads=4,
                 ff=512, dropout=0.4, n_cls=4):
        super().__init__()
        self.proj = nn.Linear(dim_in, d_model)
        self.pos = nn.Parameter(torch.randn(1, T, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_cls),
        )

    def forward(self, x, return_frames=False):  # x: [B,T,dim_in]
        x = self.proj(x) + self.pos             # [B,T,d_model]
        x = self.encoder(x)
        x = self.norm(x)
        feats = x.mean(dim=1)                   # global mean pool -> [B,d_model]
        main = self.head(feats)                 # [B,n_cls]
        if return_frames:
            return main, self.head(x)           # 逐帧辅助分类 [B,T,n_cls]
        return main


def load_checkpoint(ckpt_path, device):
    """加载单个模型; 返回 (model, mu, sd, labels)

    架构参数优先读 ckpt 内保存的训练 args; 无 args 时从 state dict 推导,
    n_heads 无法从权重形状推导 (QKV 投影与头数无关), 缺省 4 (本项目训练配置)。
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    labels = list(ck.get("labels") or LABELS)
    sd = ck["state"]
    args = ck.get("args") or {}
    d_model = int(args.get("d_model", sd["proj.weight"].shape[0]))
    n_layers = int(args.get("n_layers", len(set(
        int(k.split(".")[2]) for k in sd if k.startswith("encoder.layers.")))))
    n_heads = int(args.get("n_heads", 4))
    ff = int(args.get("ff", sd["encoder.layers.0.linear1.weight"].shape[0]))
    T = sd["pos"].shape[1]
    dim_in = sd["proj.weight"].shape[1]
    model = TemporalActionTransformer(dim_in=dim_in, T=T, d_model=d_model,
                                      n_layers=n_layers, n_heads=n_heads, ff=ff,
                                      dropout=0.0, n_cls=len(labels)).to(device)
    model.load_state_dict(sd)
    model.eval()
    mu = ck["mu"].astype(np.float32)
    sd_stat = ck["sd"].astype(np.float32)
    return model, mu, sd_stat, labels


def load_ensemble(ckpt_paths, device):
    """加载 5 折集成; 返回 (folds, labels), folds=[(model, mu, sd), ...]"""
    folds, labels = [], None
    for p in ckpt_paths:
        model, mu, sd, lab = load_checkpoint(p, device)
        if labels is None:
            labels = lab
        assert lab == labels, f"折间标签不一致: {p} {lab} vs {labels}"
        folds.append((model, mu, sd))
    return folds, labels


@torch.no_grad()
def predict_probs(model, feats, mu, sd, device, batch=128):
    """feats: [N,30,1280] float32 -> softmax 概率 [N,n_cls]"""
    X = (feats.astype(np.float32) - mu) / sd
    out = []
    for c0 in range(0, len(X), batch):
        xb = torch.tensor(X[c0:c0 + batch]).to(device)
        out.append(torch.softmax(model(xb), -1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(labels)))


@torch.no_grad()
def predict_ensemble_probs(folds, feats, device, batch=128):
    """5 折 softmax 取均值; feats: [N,30,1280] -> [N,n_cls]"""
    avg = None
    for model, mu, sd in folds:
        p = predict_probs(model, feats, mu, sd, device, batch=batch)
        avg = p if avg is None else avg + p
    return avg / len(folds)
