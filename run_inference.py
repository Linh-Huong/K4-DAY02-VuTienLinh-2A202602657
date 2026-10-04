"""run_inference.py - Chạy tự động Bước 3: Đánh giá các phương pháp suy luận & Hiệu chuẩn.

Sử dụng checkpoint tốt nhất từ Bước 2: runs/T09/seed0/best_model.pth (ConvNeXt-Tiny + CutMix + Label Smoothing).
Khảo sát các phương pháp suy luận theo GUIDE mục 4 & RUBRIC mục D:
  - I00: 1-view mốc (Resize 256 -> CenterCrop 224)
  - I01: TTA lật ngang (H-flip, K=2, gộp prob)
  - I02: TTA lật ngang (H-flip, K=2, gộp logit)
  - I03: Dò độ phân giải kiểm tra (FixRes: Test ở 256x256 thay vì 224x224)
  - I04: Temperature Scaling (khớp T trên Val, đo ECE trước và sau)

Đo độ trễ chuẩn xác theo benchmark.py:
  - Batch 1 và Batch 32
  - FP32, AMP và FP16
  - p50, p95, p99 (ms) và thông lượng (ảnh/s)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_DIR))

from eval import compute_metrics, save_predictions, NUM_CLASSES
import dataset
import model as model_utils
import inference
import benchmark


def main():
    print("=" * 75)
    print("⚡ BẮT ĐẦU CHẠY BƯỚC 3: PHƯƠNG PHÁP SUY LUẬN & ĐO ĐỘ TRỄ CHUẨN XÁC")
    print("Thiết bị:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("Mô hình sử dụng: T09 (ConvNeXt-Tiny + CutMix + Label Smoothing)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Nạp dataset VAL
    train_df, val_df, test_df = dataset.load_split("data/labels", fold=0)
    val_tf_224 = dataset.build_transforms(train=False, img_size=224)
    val_tf_256 = dataset.build_transforms(train=False, img_size=256)

    val_loader_224 = dataset.make_loader(val_df, "data/images", val_tf_224, batch_size=64, train=False, num_workers=2)
    val_loader_256 = dataset.make_loader(val_df, "data/images", val_tf_256, batch_size=64, train=False, num_workers=2)

    # 2. Nạp checkpoint T09
    model = model_utils.build_model("convnext_tiny", pretrained=False, num_classes=9).to(device)
    ckpt_path = Path("runs/T09/seed0/best_model.pth")
    if not ckpt_path.exists():
        ckpt_path = Path("runs/B02/seed0/best_model.pth")
    print(f"Nạp checkpoint từ: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    inference_results = []

    # ---------------------------------------------------------
    # I00: 1-view chuẩn (224x224)
    # ---------------------------------------------------------
    print("\n[1/5] Đang đánh giá I00: 1-view chuẩn (224x224)...")
    fns, y_true, logits_base = inference.predict_logits(model, val_loader_224, device, view=inference.view_identity)
    probs_i00 = inference.softmax_np(logits_base)
    m_i00 = compute_metrics(y_true, probs_i00.argmax(axis=1), probs_i00)

    lat_i00 = benchmark.latency_report(model, batch_size=1, img_size=224, dtype="fp32", device=str(device))
    p50_base = lat_i00["p50"]

    inference_results.append({
        "exp_id": "I00",
        "method": "1-view mốc (224x224)",
        "views_or_models": 1,
        "val_macro_f1": round(m_i00["macro_f1"] * 100, 2),
        "val_top1": round(m_i00["top1"] * 100, 2),
        "ece_val": round(m_i00["ece"], 4),
        "latency_p50_ms": round(lat_i00["p50"], 2),
        "latency_p95_ms": round(lat_i00["p95"], 2),
        "latency_p99_ms": round(lat_i00["p99"], 2),
        "throughput_img_s": round(lat_i00["images_per_s"], 1),
        "relative_cost_vs_I00": "1.0x",
    })
    print(f"-> I00: Val Macro-F1 = {inference_results[-1]['val_macro_f1']}%, ECE = {inference_results[-1]['ece_val']}, p50 = {lat_i00['p50']:.2f} ms")

    # ---------------------------------------------------------
    # I01: TTA lật ngang (H-flip, K=2, gộp prob)
    # ---------------------------------------------------------
    print("\n[2/5] Đang đánh giá I01: TTA lật ngang (K=2, gộp theo xác suất)...")
    _, _, logits_flip = inference.predict_logits(model, val_loader_224, device, view=inference.view_hflip)
    probs_i01 = inference.aggregate_views([logits_base, logits_flip], space="prob")
    m_i01 = compute_metrics(y_true, probs_i01.argmax(axis=1), probs_i01)

    lat_i01 = benchmark.tta_latency(model, k_views=2, img_size=224, dtype="fp32", device=str(device))
    inference_results.append({
        "exp_id": "I01",
        "method": "TTA lật ngang (K=2, gộp prob)",
        "views_or_models": 2,
        "val_macro_f1": round(m_i01["macro_f1"] * 100, 2),
        "val_top1": round(m_i01["top1"] * 100, 2),
        "ece_val": round(m_i01["ece"], 4),
        "latency_p50_ms": round(lat_i01["p50"], 2),
        "latency_p95_ms": round(lat_i01["p95"], 2),
        "latency_p99_ms": round(lat_i01["p99"], 2),
        "throughput_img_s": round(lat_i01["images_per_s"], 1),
        "relative_cost_vs_I00": "2.0x",
    })
    print(f"-> I01: Val Macro-F1 = {inference_results[-1]['val_macro_f1']}%, ECE = {inference_results[-1]['ece_val']}, p50 = {lat_i01['p50']:.2f} ms")

    # ---------------------------------------------------------
    # I02: TTA lật ngang (H-flip, K=2, gộp logit)
    # ---------------------------------------------------------
    print("\n[3/5] Đang đánh giá I02: TTA lật ngang (K=2, gộp theo logit)...")
    probs_i02 = inference.aggregate_views([logits_base, logits_flip], space="logit")
    m_i02 = compute_metrics(y_true, probs_i02.argmax(axis=1), probs_i02)

    inference_results.append({
        "exp_id": "I02",
        "method": "TTA lật ngang (K=2, gộp logit)",
        "views_or_models": 2,
        "val_macro_f1": round(m_i02["macro_f1"] * 100, 2),
        "val_top1": round(m_i02["top1"] * 100, 2),
        "ece_val": round(m_i02["ece"], 4),
        "latency_p50_ms": round(lat_i01["p50"], 2),
        "latency_p95_ms": round(lat_i01["p95"], 2),
        "latency_p99_ms": round(lat_i01["p99"], 2),
        "throughput_img_s": round(lat_i01["images_per_s"], 1),
        "relative_cost_vs_I00": "2.0x",
    })
    print(f"-> I02: Val Macro-F1 = {inference_results[-1]['val_macro_f1']}%, ECE = {inference_results[-1]['ece_val']}")

    # ---------------------------------------------------------
    # I03: FixRes (Kiểm tra ở độ phân giải 256x256)
    # ---------------------------------------------------------
    print("\n[4/5] Đang đánh giá I03: Dò độ phân giải kiểm tra FixRes (256x256)...")
    _, _, logits_256 = inference.predict_logits(model, val_loader_256, device, view=inference.view_identity)
    probs_i03 = inference.softmax_np(logits_256)
    m_i03 = compute_metrics(y_true, probs_i03.argmax(axis=1), probs_i03)

    lat_i03 = benchmark.latency_report(model, batch_size=1, img_size=256, dtype="fp32", device=str(device))
    inference_results.append({
        "exp_id": "I03",
        "method": "Dò độ phân giải FixRes (256x256)",
        "views_or_models": 1,
        "val_macro_f1": round(m_i03["macro_f1"] * 100, 2),
        "val_top1": round(m_i03["top1"] * 100, 2),
        "ece_val": round(m_i03["ece"], 4),
        "latency_p50_ms": round(lat_i03["p50"], 2),
        "latency_p95_ms": round(lat_i03["p95"], 2),
        "latency_p99_ms": round(lat_i03["p99"], 2),
        "throughput_img_s": round(lat_i03["images_per_s"], 1),
        "relative_cost_vs_I00": f"{lat_i03['p50'] / p50_base:.2f}x",
    })
    print(f"-> I03: Val Macro-F1 = {inference_results[-1]['val_macro_f1']}%, ECE = {inference_results[-1]['ece_val']}, p50 = {lat_i03['p50']:.2f} ms")

    # ---------------------------------------------------------
    # I04: Temperature Scaling (Hiệu chuẩn ECE)
    # ---------------------------------------------------------
    print("\n[5/5] Đang thực hiện Temperature Scaling (Khớp T trên Val)...")
    T_opt = inference.fit_temperature(logits_base, y_true)
    probs_ts = inference.apply_temperature(logits_base, T_opt)
    m_ts = compute_metrics(y_true, probs_ts.argmax(axis=1), probs_ts)

    inference_results.append({
        "exp_id": "I04",
        "method": f"Temperature Scaling (T={T_opt:.3f})",
        "views_or_models": 1,
        "val_macro_f1": round(m_ts["macro_f1"] * 100, 2),
        "val_top1": round(m_ts["top1"] * 100, 2),
        "ece_val": round(m_ts["ece"], 4),
        "latency_p50_ms": round(lat_i00["p50"], 2),
        "latency_p95_ms": round(lat_i00["p95"], 2),
        "latency_p99_ms": round(lat_i00["p99"], 2),
        "throughput_img_s": round(lat_i00["images_per_s"], 1),
        "relative_cost_vs_I00": "1.0x (Miễn phí)",
    })
    print(f"-> I04: T tối ưu = {T_opt:.3f} | ECE trước: {m_i00['ece']:.4f} -> ECE sau: {m_ts['ece']:.4f} (Giảm {((m_i00['ece']-m_ts['ece'])/m_i00['ece'])*100:.1f}%)")

    # Lưu bảng kết quả Inference
    df_inf = pd.DataFrame(inference_results)
    out_inf_csv = Path("runs/step3_inference_summary.csv")
    df_inf.to_csv(out_inf_csv, index=False)
    print(f"\nBảng kết quả suy luận (đã lưu tại {out_inf_csv}):\n")
    print(df_inf.to_string(index=False))

    # ---------------------------------------------------------
    # Khảo sát độ trễ chi tiết (Sheet Latency)
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("⏱️ KHẢO SÁT ĐỘ TRỄ CHI TIẾT (GPU, DTYPE, BATCH SIZE) CHO SHEET 'LATENCY'")
    print("=" * 75)
    latency_configs = [
        {"batch": 1, "dtype": "fp32", "desc": "FP32 Batch 1 (Robot thời gian thực)"},
        {"batch": 1, "dtype": "amp", "desc": "AMP FP16 Batch 1"},
        {"batch": 1, "dtype": "fp16", "desc": "Pure FP16 Batch 1"},
        {"batch": 32, "dtype": "fp32", "desc": "FP32 Batch 32 (Đo thông lượng máy chủ)"},
        {"batch": 32, "dtype": "amp", "desc": "AMP FP16 Batch 32"},
        {"batch": 32, "dtype": "fp16", "desc": "Pure FP16 Batch 32"},
    ]
    latency_rows = []
    for lc in latency_configs:
        rep = benchmark.latency_report(model, batch_size=lc["batch"], img_size=224, dtype=lc["dtype"], device=str(device), warmup=10, iters=50)
        row = {
            "config": lc["desc"],
            "gpu": rep["gpu"],
            "dtype": lc["dtype"],
            "batch": lc["batch"],
            "img_size": 224,
            "p50_ms": round(rep["p50"], 2),
            "p95_ms": round(rep["p95"], 2),
            "p99_ms": round(rep["p99"], 2),
            "images_per_s": round(rep["images_per_s"], 1),
            "torch_version": rep["torch"],
        }
        latency_rows.append(row)
        print(f"-> {lc['desc']}: p50={row['p50_ms']} ms, p95={row['p95_ms']} ms, Thông lượng={row['images_per_s']} img/s")

    df_lat = pd.DataFrame(latency_rows)
    out_lat_csv = Path("runs/step3_latency_summary.csv")
    df_lat.to_csv(out_lat_csv, index=False)
    print(f"\nBảng độ trễ (đã lưu tại {out_lat_csv}):\n")
    print(df_lat.to_string(index=False))

    # Lưu nhiệt độ T để áp dụng sang test ở Bước 4
    with open("runs/best_temperature.json", "w", encoding="utf-8") as f:
        json.dump({"T_opt": T_opt, "ece_before": m_i00["ece"], "ece_after": m_ts["ece"]}, f, indent=2)


if __name__ == "__main__":
    main()
