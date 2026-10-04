"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).
"""
from __future__ import annotations

import copy
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_np(z: np.ndarray) -> np.ndarray:
    z_max = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - z_max)
    return e / np.sum(e, axis=-1, keepdims=True)


def predict_logits(model: nn.Module, loader, device: torch.device,
                   view: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    model.eval()
    filenames = []
    y_true_list = []
    logits_list = []

    with torch.inference_mode():
        for x, y, fns in loader:
            if view is not None:
                x = view(x)
            x = x.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available() and device.type == "cuda"):
                out = model(x)

            filenames.extend(fns)
            y_true_list.append(y.numpy())
            logits_list.append(out.cpu().float().numpy())

    all_y = np.concatenate(y_true_list, axis=0)
    all_logits = np.concatenate(logits_list, axis=0)
    return filenames, all_y, all_logits


def view_identity(x: torch.Tensor) -> torch.Tensor:
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch (N, C, H, W) (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int) -> List[torch.Tensor]:
    """5 crop (4 góc + giữa) kích thước `crop`."""
    _, _, h, w = x.shape
    if h < crop or w < crop:
        raise ValueError(f"Kích thước ảnh ({h}x{w}) nhỏ hơn crop ({crop})")

    tl = x[:, :, 0:crop, 0:crop]
    tr = x[:, :, 0:crop, w - crop:w]
    bl = x[:, :, h - crop:h, 0:crop]
    br = x[:, :, h - crop:h, w - crop:w]

    ch = (h - crop) // 2
    cw = (w - crop) // 2
    cc = x[:, :, ch:ch + crop, cw:cw + crop]

    return [cc, tl, tr, bl, br]


def views_multiscale(x: torch.Tensor, sizes: List[int]) -> List[torch.Tensor]:
    """Resize batch về từng kích thước trong `sizes`."""
    return [F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False) for s in sizes]


def aggregate_views(logits_per_view: List[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62)."""
    if space == "prob":
        probs_list = [softmax_np(lg) for lg in logits_per_view]
        mean_probs = np.mean(probs_list, axis=0)
        return mean_probs / np.sum(mean_probs, axis=-1, keepdims=True)
    elif space == "logit":
        mean_logits = np.mean(logits_per_view, axis=0)
        return softmax_np(mean_logits)
    else:
        raise ValueError(f"Không hỗ trợ space={space}. Chọn 'prob' hoặc 'logit'.")


def ensemble_probs(list_of_probs: List[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình."""
    mean_p = np.mean(list_of_probs, axis=0)
    return mean_p / np.sum(mean_p, axis=-1, keepdims=True)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69)."""
    val_labels = np.asarray(val_labels, dtype=int)
    n = len(val_labels)

    def nll_loss(log_t_arr):
        t = np.exp(log_t_arr[0])
        scaled = val_logits / t
        scaled_max = np.max(scaled, axis=1, keepdims=True)
        exp_scaled = np.exp(scaled - scaled_max)
        log_sum_exp = np.log(np.sum(exp_scaled, axis=1)) + scaled_max.squeeze(1)
        correct_logits = scaled[np.arange(n), val_labels]
        return np.mean(log_sum_exp - correct_logits)

    res = minimize(nll_loss, x0=[0.0], method="Nelder-Mead")
    t_opt = float(np.exp(res.x[0]))
    return float(np.clip(t_opt, 0.05, 10.0))


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Trả về softmax(logits / T)."""
    t_val = max(1e-4, float(T))
    return softmax_np(logits / t_val)


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75)."""
    fused_model = copy.deepcopy(model).eval()

    # Dùng tiện ích chính thức của PyTorch nếu có
    try:
        from torch.nn.utils.fusion import fuse_conv_bn_eval

        # Duyệt qua các submodule tìm cặp (Conv2d, BatchNorm2d)
        for name, module in list(fused_model.named_children()):
            if isinstance(module, nn.Sequential):
                for i in range(len(module) - 1):
                    if isinstance(module[i], nn.Conv2d) and isinstance(module[i + 1], nn.BatchNorm2d):
                        module[i] = fuse_conv_bn_eval(module[i], module[i + 1])
                        module[i + 1] = nn.Identity()
            # Đệ quy với các block con
            fused_model._modules[name] = fuse_conv_bn(module)
    except Exception:
        pass

    return fused_model
