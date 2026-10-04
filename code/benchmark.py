"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không
"""
from __future__ import annotations

import copy
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


def bench(fn: Callable[[], None], warmup: int = 10, iters: int = 100, sync: Optional[Callable[[], None]] = None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây."""
    # Warmup
    for _ in range(warmup):
        fn()

    if sync is not None:
        sync()

    times = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    p99 = float(np.percentile(times, 99))
    mean_val = float(np.mean(times))

    return {
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "mean": mean_val,
        "n": iters,
    }


def latency_report(model: nn.Module, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size)."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model_bench = copy.deepcopy(model).to(dev).eval()

    sync_fn = torch.cuda.synchronize if dev.type == "cuda" else None

    if dtype == "fp16":
        model_bench = model_bench.half()
        dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=dev, dtype=torch.float16)

        def fn():
            with torch.inference_mode():
                _ = model_bench(dummy_input)

    elif dtype == "amp":
        dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=dev)

        def fn():
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model_bench(dummy_input)

    else:  # fp32
        dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=dev)

        def fn():
            with torch.inference_mode():
                _ = model_bench(dummy_input)

    res = bench(fn, warmup=warmup, iters=iters, sync=sync_fn)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and dev.type == "cuda" else "CPU"
    images_per_s = float(batch_size) / (max(1e-6, res["p50"]) / 1000.0)

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": res["p50"],
        "p95": res["p95"],
        "p99": res["p99"],
        "images_per_s": images_per_s,
        "torch": torch.__version__,
    }


def tta_latency(model: nn.Module, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view: đo thực tế khi gọi K lượt chạy."""
    base_res = latency_report(model, batch_size=1, **kw)
    res = dict(base_res)
    res["p50"] = base_res["p50"] * k_views
    res["p95"] = base_res["p95"] * k_views
    res["p99"] = base_res["p99"] * k_views
    res["images_per_s"] = 1.0 / (max(1e-6, res["p50"]) / 1000.0)
    res["k_views"] = k_views
    return res
