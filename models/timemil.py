"""Temporal MIL head for HyperMIL with configurable positional encoding and pooling.

Adapted from TimeMIL (Chen et al., 2024):
  TimeMIL: Advancing multivariate time series classification via a time-aware
  multiple instance learning. arXiv:2405.03140
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nystrom_attention import NystromAttention


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dropout=0.2, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=max(4, dim // 2),
            pinv_iterations=6,
            residual=True,
            dropout=dropout,
        )

    def forward(self, x, mask=None, return_attn=False):
        x_norm = self.norm(x)
        if return_attn:
            x_att, attn = self.attn(x_norm, mask=mask, return_attn=True)
            return x + x_att, attn
        return x + self.attn(x_norm, mask=mask), None


def mexican_hat_wavelet(size, scale, shift, device):
    x = torch.linspace(-(size[1] - 1) // 2, (size[1] - 1) // 2, size[1], device=device)
    x = x.reshape(1, -1).repeat(size[0], 1) - shift
    c = 2 / (3**0.5 * torch.pi**0.25)
    return c * (1 - (x / scale) ** 2) * torch.exp(-(x / scale) ** 2 / 2) / (torch.abs(scale) ** 0.5)


class WaveletEncoding(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, wave1, wave2, wave3):
        cls_token, feat_token = x[:, 0], x[:, 1:]
        x_t = feat_token.transpose(1, 2)
        d = x_t.shape[1]
        device = x.device

        scale1, shift1 = wave1[0, :], wave1[1, :]
        scale2, shift2 = wave2[0, :], wave2[1, :]
        scale3, shift3 = wave3[0, :], wave3[1, :]
        k1 = mexican_hat_wavelet((d, 19), scale1, shift1, device)
        k2 = mexican_hat_wavelet((d, 19), scale2, shift2, device)
        k3 = mexican_hat_wavelet((d, 19), scale3, shift3, device)

        pos = (
            F.conv1d(x_t, k1.unsqueeze(1), groups=d, padding="same")
            + F.conv1d(x_t, k2.unsqueeze(1), groups=d, padding="same")
            + F.conv1d(x_t, k3.unsqueeze(1), groups=d, padding="same")
        )
        out = feat_token + self.proj(pos.transpose(1, 2))
        return torch.cat((cls_token.unsqueeze(1), out), dim=1)


class SinusoidalEncoding(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim

    def _generate_positional_encoding(self, seq_len, device):
        position = torch.arange(0, seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=device) * -(math.log(10000.0) / self.dim))
        pe = torch.zeros(seq_len, self.dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, seq_len):
        pos = self._generate_positional_encoding(seq_len, x.device)
        return torch.cat((x[:, :1, :], x[:, 1:, :] + pos[:, 1:, :]), dim=1)


class PoolingLayer(nn.Module):
    def __init__(self, pooling_type="cls"):
        super().__init__()
        self.pooling_type = pooling_type.lower()

    def forward(self, x, mask, instance_head=None, warmup=False, global_token=None):
        instance_repr = x[:, 1:]
        if mask is not None:
            flat_mask = mask.to(dtype=torch.bool, device=x.device)
            mask_float = flat_mask.float()
        else:
            flat_mask = torch.ones(instance_repr.shape[:2], dtype=torch.bool, device=x.device)
            mask_float = flat_mask.float()

        if self.pooling_type == "cls":
            pooled = x[:, 0]
        elif self.pooling_type == "mean":
            pooled = (instance_repr * mask_float.unsqueeze(-1)).sum(dim=1) / mask_float.sum(dim=1, keepdim=True).clamp_min(1e-6)
        elif self.pooling_type == "max":
            pooled = instance_repr.masked_fill(~flat_mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        elif self.pooling_type == "attention":
            if instance_head is None:
                raise ValueError("instance_head must be provided for attention pooling")
            cls_attn = instance_head(instance_repr).squeeze(-1).masked_fill(flat_mask == 0, float("-inf"))
            pooled = (instance_repr * torch.softmax(cls_attn, dim=-1).unsqueeze(-1)).sum(dim=1)
        elif self.pooling_type == "conjunct":
            mean_pooled = (instance_repr * mask_float.unsqueeze(-1)).sum(dim=1) / (mask_float.sum(dim=1, keepdim=True) + 1e-6)
            max_pooled = instance_repr.masked_fill(~flat_mask.unsqueeze(-1), float("-inf")).max(dim=1).values
            pooled = mean_pooled + max_pooled
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        if warmup and global_token is not None:
            pooled = 0.1 * pooled + 0.99 * global_token
        return pooled


class TimeMIL(nn.Module):
    def __init__(self, in_features, n_classes=2, mDim=64, max_seq_len=400, dropout=0.0, encoding="wavelet", pooling="cls", num_layers=2):
        super().__init__()
        self.encoding = encoding
        self.pooling = pooling
        self.pooling_layer = PoolingLayer(pooling_type=pooling)
        self.num_layers = num_layers

        if encoding == "sinusoidal":
            self.pos_layer = SinusoidalEncoding(dim=mDim)
            self.pos_layer2 = SinusoidalEncoding(dim=mDim)
        elif encoding == "wavelet":
            self.wave1 = nn.Parameter(torch.randn(2, mDim, 1))
            self.wave2 = nn.Parameter(torch.randn(2, mDim, 1))
            self.wave3 = nn.Parameter(torch.randn(2, mDim, 1))
            self.wave1_ = nn.Parameter(torch.randn(2, mDim, 1))
            self.wave2_ = nn.Parameter(torch.randn(2, mDim, 1))
            self.wave3_ = nn.Parameter(torch.randn(2, mDim, 1))
            self.pos_layer = WaveletEncoding(mDim)
            self.pos_layer2 = WaveletEncoding(mDim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        self.layer1 = TransLayer(dim=mDim, dropout=dropout)
        self.layers = nn.ModuleList([TransLayer(dim=mDim, dropout=dropout) for _ in range(self.num_layers)])
        self._fc2 = nn.Sequential(nn.Linear(mDim, mDim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mDim, n_classes))
        self.instance_head = nn.Linear(mDim, 1)
        initialize_weights(self)

    def forward(self, x, mask=None, warmup=False, return_attn=False):
        b, seq_len, _ = x.shape
        if mask is not None:
            if mask.shape[1] < x.shape[1]:
                mask = F.pad(mask, (0, x.shape[1] - mask.shape[1]), value=0)
            mask_ = mask.unsqueeze(-1).float().to(x.device)
            global_token = (x * mask_).sum(dim=1) / (mask_.sum(dim=1) + 1e-6)
            flat_mask = mask.squeeze(-1) if mask.dim() == 3 else mask
            flat_mask = flat_mask.to(dtype=torch.bool, device=x.device)
        else:
            global_token = x.mean(dim=1)
            flat_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)

        cls_mask = torch.ones(b, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, flat_mask], dim=1)
        x = torch.cat((self.cls_token.expand(b, -1, -1), x), dim=1)

        if self.encoding == "sinusoidal":
            x = self.pos_layer(x, seq_len + 1)
            x, _ = self.layer1(x, mask=full_mask)
            x = self.pos_layer2(x, seq_len + 1)
        elif self.encoding == "wavelet":
            x = self.pos_layer(x, self.wave1, self.wave2, self.wave3)
            x, _ = self.layer1(x, mask=full_mask)
            x = self.pos_layer2(x, self.wave1_, self.wave2_, self.wave3_)

        attn_map = None
        for idx, layer in enumerate(self.layers):
            x, attn_map = layer(x, mask=full_mask, return_attn=False if idx < self.num_layers - 1 else return_attn)

        pooled = self.pooling_layer(
            x=x,
            mask=flat_mask,
            instance_head=self.instance_head if self.pooling == "attention" else None,
            warmup=warmup,
            global_token=global_token,
        )
        logits = self._fc2(pooled)
        return (logits, attn_map) if return_attn else logits
