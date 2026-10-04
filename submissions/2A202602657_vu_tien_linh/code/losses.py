"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).
Giao diện giữ nguyên:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted"."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        smoothing = kw.get("smoothing", kw.get("label_smoothing", 0.1))
        return LabelSmoothingCE(smoothing=smoothing)
    elif kind == "focal":
        gamma = kw.get("gamma", kw.get("focal_gamma", 2.0))
        alpha = kw.get("alpha", None)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        weight = kw.get("weight", None)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        return nn.CrossEntropyLoss()


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56)."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = float(smoothing)
        self.ce = nn.CrossEntropyLoss(label_smoothing=self.smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, target)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).
    Khi gamma = 0 và alpha = None, bằng đúng standard CrossEntropyLoss.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (N, C), target: (N,)
        log_p = F.log_softmax(logits, dim=-1)
        target = target.view(-1, 1)
        log_pt = log_p.gather(1, target).squeeze(1)
        pt = log_pt.exp()

        focal_term = (1.0 - pt) ** self.gamma
        loss = -focal_term * log_pt

        if self.alpha is not None:
            at = self.alpha.to(logits.device).gather(0, target.squeeze(1))
            loss = at * loss

        return loss.mean()


def class_weights(counts: Union[list, np.ndarray, dict], beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo "số mẫu hiệu dụng": w_c = (1 - beta) / (1 - beta ** n_c)
      (slide trang 57, Cui et al. arXiv:1901.05555); chuẩn hoá tổng trọng số về số lớp
    """
    if isinstance(counts, dict):
        counts = [counts[k] for k in sorted(counts.keys())]
    counts_arr = np.array(counts, dtype=np.float64)
    num_classes = len(counts_arr)

    if beta <= 0.0:
        weights = 1.0 / np.maximum(counts_arr, 1.0)
        weights = weights / weights.mean()
    else:
        effective_num = 1.0 - np.power(beta, counts_arr)
        effective_num = np.maximum(effective_num, 1e-8)
        weights = (1.0 - beta) / effective_num
        weights = weights * (num_classes / weights.sum())

    return torch.tensor(weights, dtype=torch.float32)


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix") -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]]:
    """Trộn một batch ảnh và nhãn theo Mixup hoặc CutMix."""
    if alpha <= 0:
        return x, (y, y, 1.0)

    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)

    y_a = y
    y_b = y[perm]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[perm]
        return x_mixed, (y_a, y_b, lam)

    elif mode == "cutmix":
        _, _, h, w = x.shape
        # Tính kích thước hộp
        cut_rat = math.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = int(np.random.randint(w))
        cy = int(np.random.randint(h))

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[perm, :, bby1:bby2, bbx1:bbx2]

        # Điều chỉnh lại lam theo diện tích thực tế
        actual_lam = 1.0 - float((bbx2 - bbx1) * (bby2 - bby1)) / float(w * h)
        return x_mixed, (y_a, y_b, actual_lam)

    else:
        return x, (y, y, 1.0)


def mixed_loss(criterion, logits: torch.Tensor, targets: Tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
