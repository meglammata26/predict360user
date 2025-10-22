# predict360user/models/hybrid_model.py
from __future__ import annotations
import math
from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from predict360user.base_model import BaseModel
from predict360user.run_config import RunConfig
from .spherical_cnn import SphericalConv2d, SphericalMaxPool2d


# ---------- SPVP360-style spherical encoder (3 stages + refinement) ----------
class SphericalCNNEncoder(nn.Module):
    def __init__(self, in_ch=3, feat_dim=256):
        super().__init__()
        # Stage 1
        self.s1_conv1 = SphericalConv2d(in_ch, 64, 3, 1, padding="sphere")
        self.s1_conv2 = SphericalConv2d(64, 64, 3, 1, padding="sphere")
        self.s1_pool  = SphericalMaxPool2d(2)
        # Stage 2
        self.s2_conv1 = SphericalConv2d(64, 128, 3, 1, padding="sphere")
        self.s2_conv2 = SphericalConv2d(128, 128, 3, 1, padding="sphere")
        self.s2_pool  = SphericalMaxPool2d(2)
        # Stage 3
        self.s3_conv1 = SphericalConv2d(128, 256, 3, 1, padding="sphere")
        self.s3_conv2 = SphericalConv2d(256, 256, 3, 1, padding="sphere")
        self.s3_pool  = SphericalMaxPool2d(2)
        # Refinement
        self.ref1 = SphericalConv2d(256, 256, 3, 1, padding="sphere")
        self.ref2 = SphericalConv2d(256, 256, 3, 1, padding="sphere")

        self.gap  = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, feat_dim)

        self.bn = nn.ModuleList([nn.BatchNorm2d(c) for c in
                                 [64, 64, 128, 128, 256, 256, 256, 256]])

    def forward(self, x):
        b = self.bn
        x = F.relu(b[0](self.s1_conv1(x)))
        x = F.relu(b[1](self.s1_conv2(x)))
        x = self.s1_pool(x)
        x = F.relu(b[2](self.s2_conv1(x)))
        x = F.relu(b[3](self.s2_conv2(x)))
        x = self.s2_pool(x)
        x = F.relu(b[4](self.s3_conv1(x)))
        x = F.relu(b[5](self.s3_conv2(x)))
        x = self.s3_pool(x)
        x = F.relu(b[6](self.ref1(x)))
        x = F.relu(b[7](self.ref2(x)))
        x = self.gap(x).flatten(1)
        return self.proj(x)


# ---------- Vision Transformer branch ----------
class SaliencyViT(nn.Module):
    def __init__(self, feat_dim=256, vit_width=768, pretrained=False):
        super().__init__()
        try:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            self.vit = vit_b_16(weights=weights)
            self.vit.heads = nn.Identity()
            vit_out = vit_width
        except Exception:
            self.vit = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=4, padding=3),
                nn.ReLU(True),
                nn.AdaptiveAvgPool2d((14, 14)),
                nn.Flatten(),
                nn.Linear(64 * 14 * 14, vit_width),
                nn.ReLU(True),
            )
            vit_out = vit_width
        self.proj = nn.Linear(vit_out, feat_dim)

    def forward(self, x):
        return self.proj(self.vit(x))


def _unit_norm(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


# ---------- Hybrid 360 model ----------
class Hybrid360Torch(BaseModel):
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.geom = SphericalCNNEncoder()
        self.sal  = SaliencyViT()
        combined = 256 + 256
        hidden  = getattr(cfg, "lstm_hidden_dim", 512)
        layers  = getattr(cfg, "lstm_layers", 2)

        self.lstm = nn.LSTM(combined, hidden, layers,
                            batch_first=True, dropout=0.3)
        self.fc   = nn.Linear(hidden, 3)

        self.net   = nn.ModuleList([self.geom, self.sal, self.lstm, self.fc]).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.crit  = nn.MSELoss()
        self._epochs = getattr(cfg, "epochs", 2)
        self._bsz    = getattr(cfg, "batch_size", 8)

    # --- helpers ---
    def _ensure_frames(self, frames_seq, seq_len, H=224, W=224):
        if frames_seq is None:
            return torch.zeros((1, seq_len, 3, H, W), device=self.device)
        t = torch.as_tensor(frames_seq, dtype=torch.float32)
        if t.ndim == 3:
            t = t.unsqueeze(1).repeat(1, 3, 1, 1)
        if t.shape[1] != 3:
            t = t.unsqueeze(1).repeat(1, 3, 1, 1)
        if t.shape[-2:] != (H, W):
            t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        return t.unsqueeze(0).to(self.device)

    def _encode_seq(self, vid):
        B, T, C, H, W = vid.shape
        feats = []
        for t in range(T):
            x = vid[:, t]
            g = self.geom(x)
            s = self.sal(x)
            feats.append(torch.cat([g, s], 1))
        return torch.stack(feats, 1)

    # --- BaseModel API ---
    def fit(self, df):
        self.net.train()
        if len(df) == 0:
            return self
        rows = df[df["partition"].isin(["train", "val"])].to_dict("records") or df.to_dict("records")
        steps = max(1, math.ceil(len(rows)/self._bsz))
        for _ in range(self._epochs):
            idx = 0
            for _ in range(steps):
                batch = rows[idx:idx+self._bsz]
                idx = (idx+self._bsz)%len(rows)
                vids, ys = [], []
                for r in batch:
                    m = np.asarray(r["m_window"])
                    f = r.get("frames_seq")
                    vids.append(self._ensure_frames(f, len(m)))
                    h = np.asarray(r["h_window"])
                    ys.append(torch.from_numpy(h[0:1]))
                vid = torch.cat(vids, 0)
                y = torch.cat(ys, 0).float().to(self.device)
                seq = self._encode_seq(vid)
                out, _ = self.lstm(seq)
                p = _unit_norm(self.fc(out[:, -1]))
                loss = self.crit(p, y.squeeze(1))
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
        return self

    def predict(self, df) -> Sequence:
        self.net.eval()
        preds, H = [], self.cfg.h_window
        with torch.no_grad():
            for _, r in df.iterrows():
                m = np.asarray(r["m_window"])
                f = r.get("frames_seq")
                vid = self._ensure_frames(f, len(m))
                seq = self._encode_seq(vid)
                out, (h, c) = self.lstm(seq)
                last = vid[:, -1:]
                steps = []
                for _ in range(H):
                    enc = self._encode_seq(last)
                    out, (h, c) = self.lstm(enc, (h, c))
                    p = _unit_norm(self.fc(out[:, -1]))
                    steps.append(p.squeeze(0).cpu().numpy())
                preds.append(np.stack(steps))
        return preds
 