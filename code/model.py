"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện giữ nguyên:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import timm

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    is_pretrained = pretrained and (init != "scratch")
    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    # Ghi lại pretrained tag
    tag = "scratch"
    if is_pretrained:
        cfg = getattr(model, "pretrained_cfg", {}) or getattr(model, "default_cfg", {})
        tag = cfg.get("tag", cfg.get("architecture", name)) if isinstance(cfg, dict) else str(cfg)
    model.pretrained_tag = tag
    model.init_mode = init

    if init == "frozen":
        freeze_backbone(model)
        model._frozen_backbone = True
    else:
        model._frozen_backbone = False

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ classifier head.
    Khi backbone đóng băng, BatchNorm cũng cần đặt ở chế độ eval.
    """
    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        head_params = set(classifier.parameters())
    elif isinstance(classifier, nn.Parameter):
        head_params = {classifier}
    else:
        head_params = set()

    for p in model.parameters():
        if p not in head_params:
            p.requires_grad = False
        else:
            p.requires_grad = True

    # Đưa các lớp BatchNorm về chế độ eval
    for m in model.modules():
        if isinstance(m, (nn.modules.batchnorm._BatchNorm,)):
            m.eval()


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay
    """
    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        head_params_set = set(classifier.parameters())
    elif isinstance(classifier, nn.Parameter):
        head_params_set = {classifier}
    else:
        head_params_set = set()

    head_params = []
    backbone_decay = []
    backbone_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param in head_params_set:
            head_params.append(param)
        else:
            if param.ndim <= 1 or name.endswith(".bias"):
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})
    if backbone_decay:
        groups.append({"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if backbone_no_decay:
        groups.append({"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0})

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (Multiply-Accumulate operations tính theo tỷ)."""
    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    model_eval = copy.deepcopy(model).eval()
    try:
        import thop
        macs, _ = thop.profile(model_eval, inputs=(dummy,), verbose=False)
        return float(macs) / 1e9
    except Exception:
        # Fallback đếm thô nếu thop lỗi
        total_flops = sum(p.numel() for p in model.parameters()) * 2
        return float(total_flops) / 1e9
