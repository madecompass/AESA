# -*- coding: utf-8 -*-

# src/models/reasoner_heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReasonerHead(nn.Module):
    """
    Reasoner Head
    입력:
      - feat: [B, feat_dim]  (메타 특징: 임베딩 + 서브출력 통계 등)
      - sent_feats: [B, T, Dp] (옵션; 문장 수준 임베딩/투영), sent_mask: [B, T] bool
    출력:
      - unc: [B, 1]              (불확실성 0~1)
      - cause_logits: [B, V]     (원인 어휘 분포 로짓; V=보카브 크기)
      - pivot_logits: [B, T]     (피벗 문장 포인터 로짓)
    """
    def __init__(self, feat_dim: int, cause_vocab: int = 128, pivot_dim: int = 128, use_pivot: bool = True, dropout: float = 0.1):
        super().__init__()
        h = max(128, feat_dim // 4)
        self.unc_mlp = nn.Sequential(
            nn.Linear(feat_dim, h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, 1)
        )
        self.cause_mlp = nn.Sequential(
            nn.Linear(feat_dim, h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, cause_vocab)
        )
        self.use_pivot = use_pivot
        if use_pivot:
            self.pivot_q = nn.Linear(feat_dim, pivot_dim)
            self.pivot_norm = nn.LayerNorm(pivot_dim)

    def forward(self, feat, sent_feats=None, sent_mask=None):
        # feat: [B, F]
        unc = torch.sigmoid(self.unc_mlp(feat))               # [B,1]
        cause_logits = self.cause_mlp(feat)                   # [B,V]

        pivot_logits = None
        if self.use_pivot and sent_feats is not None:
            # sent_feats: [B,T,Dp]; 투영 차원은 pivot_q 출력과 동일해야 함
            q = self.pivot_norm(self.pivot_q(feat))           # [B,Dp]
            # 유사도: q · s_t
            pivot_logits = torch.einsum("bd,btd->bt", q, sent_feats)  # [B,T]
            if sent_mask is not None:
                pivot_logits = pivot_logits.masked_fill(~sent_mask.bool(), -1e9)

        return unc, cause_logits, pivot_logits
