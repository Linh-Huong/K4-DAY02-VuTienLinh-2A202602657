"""run_backbones.py - Chạy tự động Bước 1: So sánh 5 Backbone.

Mục tiêu:
  1. B01: resnet50 (Họ ResNet - Mốc chuẩn)
  2. B02: convnext_tiny (Họ ConvNeXt hiện đại)
  3. B03: deit_small_patch16_224 (Họ Transformer)
  4. B04: mobilenetv3_large_100 (Họ Mạng nhẹ)
  5. B05: efficientnet_b0 (Họ Mạng nhẹ thứ hai)

Tính năng thông minh:
  - Tự động nhận diện nếu mô hình đã hoàn thành (như B01), không cần chạy lại,
    giúp tiết kiệm thời gian tối đa!
  - Tự động đo độ trễ suy luận batch 1 và xuất bảng tổng hợp cho sheet 'Backbones'.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_DIR))

import train
import model as model_utils
import benchmark

BACKBONES = [
    {"exp_id": "B01", "name": "resnet50", "desc": "ResNet-50 (Mốc chuẩn)"},
    {"exp_id": "B02", "name": "convnext_tiny", "desc": "ConvNeXt-Tiny (CNN hiện đại)"},
    {"exp_id": "B03", "name": "deit_small_patch16_224", "desc": "DeiT-Small (Vision Transformer)"},
    {"exp_id": "B04", "name": "mobilenetv3_large_100", "desc": "MobileNetV3-Large (Mạng nhẹ)"},
    {"exp_id": "B05", "name": "efficientnet_b0", "desc": "EfficientNet-B0 (Mạng nhẹ)"},
]


def main():
    print("=" * 70)
    print("🚀 BẮT ĐẦU CHẠY BƯỚC 1: SO SÁNH 5 BACKBONE TRÊN DEEPWEEDS")
    print("Thiết bị:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("Số epoch: 12 | Batch size: 64 | Optimizer: AdamW | Seed: 0")
    print("=" * 70)

    results = []
    total_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for idx, item in enumerate(BACKBONES, 1):
        exp_id = item["exp_id"]
        b_name = item["name"]
        desc = item["desc"]

        summary_file = Path("runs") / exp_id / "seed0" / "summary.json"
        best_ckpt = Path("runs") / exp_id / "seed0" / "best_model.pth"

        # Nếu mô hình đã huấn luyện xong trọn vẹn trước đó
        if summary_file.exists() and best_ckpt.exists():
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    res = json.load(f)
                # Kiểm tra nếu đủ dữ liệu
                if res.get("best_epoch") is not None and res.get("best_val_macro_f1") is not None:
                    print(f"\n[{idx}/5] ⏩ {exp_id} ({desc}) đã hoàn thành từ trước, tự động tải kết quả...")
                else:
                    res = None
            except Exception:
                res = None
        else:
            res = None

        if res is None:
            print(f"\n[{idx}/5] 🔄 Đang huấn luyện {exp_id} ({desc})...")
            cfg = train.Config(
                exp_id=exp_id,
                backbone=b_name,
                epochs=12,
                batch_size=64,
                seed=0,
                amp=True,
                num_workers=2,
                save_test_predictions=False
            )
            res = train.run(cfg)

        # Đo độ trễ suy luận batch 1 (Warmup 10, synchronize GPU, 50 lần đo)
        print(f"[{exp_id}] Đang đo độ trễ suy luận chuẩn xác (batch 1, GPU)...")
        net = model_utils.build_model(b_name, pretrained=False, num_classes=9).to(device)
        if best_ckpt.exists():
            net.load_state_dict(torch.load(best_ckpt, map_location=device))

        lat = benchmark.latency_report(net, batch_size=1, img_size=224, dtype="fp32", device=str(device), warmup=10, iters=50)

        row = {
            "exp_id": exp_id,
            "backbone": b_name,
            "desc": desc,
            "params_M": round(res["params_m"], 2),
            "gmacs": round(res["gmacs"], 2),
            "best_epoch": res["best_epoch"],
            "val_macro_f1": round(res["best_val_macro_f1"] * 100, 2),
            "val_top1": round(res["val_top1"] * 100, 2),
            "train_sec_epoch": round(res["avg_epoch_sec"], 1),
            "latency_p50_ms": round(lat["p50"], 2),
            "latency_p95_ms": round(lat["p95"], 2),
            "throughput_img_s": round(lat["images_per_s"], 1),
        }
        results.append(row)
        print(f"-> Hoàn thành {exp_id}: Val Macro-F1 = {row['val_macro_f1']}%, Latency p50 = {row['latency_p50_ms']} ms")

    total_time = (time.time() - total_start) / 60.0
    print("\n" + "=" * 70)
    print(f"🎉 HOÀN THÀNH TẤT CẢ 5 BACKBONE TRONG {total_time:.1f} PHÚT!")
    print("=" * 70)

    # Hiển thị bảng kết quả
    df = pd.DataFrame(results)
    out_csv = Path("runs") / "step1_backbones_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nBảng kết quả tổng hợp (đã lưu tại {out_csv}):\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
