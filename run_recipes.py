"""run_recipes.py - Chạy tự động Bước 2: Thử nghiệm công thức huấn luyện (Training Recipes).

Sử dụng backbone tốt nhất từ Bước 1: convnext_tiny.
Thực hiện ablation có kiểm soát trên 3 trục chính theo GUIDE mục 3:
  - Trục A (Khởi tạo):
      + T00: Mốc nền (Finetune toàn bộ - tái sử dụng kết quả B02)
      + T01: Scratch (Huấn luyện từ đầu không có pretrained)
      + T02: Frozen (Đóng băng backbone, chỉ huấn luyện classifier head)
  - Trục B (Augmentation):
      + T03: Mixup (mix_alpha=0.8)
      + T04: CutMix (mix_alpha=1.0)
      + T05: RandAugment (aug=randaug)
  - Trục C (Loss function):
      + T06: Label Smoothing (epsilon=0.1)
      + T07: Focal Loss (gamma=2.0)
      + T08: Weighted Cross-Entropy (cân bằng lớp)
  - Kết hợp tối ưu:
      + T09: Best Combination (CutMix + Label Smoothing)

Tính năng thông minh:
  - Tự động nhận diện và bỏ qua các thí nghiệm đã hoàn thành trước đó.
  - Lưu bảng kết quả tổng hợp vào runs/step2_training_recipes_summary.csv
    để điền trực tiếp vào sheet 'Training' của results.xlsx.
"""
from __future__ import annotations

import json
import shutil
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

RECIPES = [
    # Trục A: Khởi tạo
    {
        "exp_id": "T00",
        "axis": "A. Khởi tạo (Mốc)",
        "diff": "Finetune toàn bộ (Công thức nền)",
        "cfg": {"backbone": "convnext_tiny", "init": "finetune", "epochs": 12, "batch_size": 64},
        "reuse_from": "B02",  # T00 dùng chung cấu hình với B02
    },
    {
        "exp_id": "T01",
        "axis": "A. Khởi tạo",
        "diff": "Huấn luyện từ đầu (Scratch, không pretrain)",
        "cfg": {"backbone": "convnext_tiny", "init": "scratch", "epochs": 12, "batch_size": 64},
    },
    {
        "exp_id": "T02",
        "axis": "A. Khởi tạo",
        "diff": "Đóng băng backbone (Frozen, chỉ train head)",
        "cfg": {"backbone": "convnext_tiny", "init": "frozen", "epochs": 12, "batch_size": 64},
    },
    # Trục B: Augmentation
    {
        "exp_id": "T03",
        "axis": "B. Augmentation",
        "diff": "Mixup (alpha=0.8)",
        "cfg": {"backbone": "convnext_tiny", "init": "finetune", "mix": "mixup", "mix_alpha": 0.8, "epochs": 12, "batch_size": 64},
    },
    {
        "exp_id": "T04",
        "axis": "B. Augmentation",
        "diff": "CutMix (alpha=1.0)",
        "cfg": {"backbone": "convnext_tiny", "init": "finetune", "mix": "cutmix", "mix_alpha": 1.0, "epochs": 12, "batch_size": 64},
    },
    {
        "exp_id": "T05",
        "axis": "B. Augmentation",
        "diff": "RandAugment (2 ops, mag 9)",
        "cfg": {"backbone": "convnext_tiny", "init": "finetune", "aug": "randaug", "epochs": 12, "batch_size": 64},
    },
    # Trục C: Loss function
    {
        "exp_id": "T06",
        "axis": "C. Hàm Loss",
        "diff": "Label Smoothing CE (eps=0.1)",
        "cfg": {"backbone": "convnext_tiny", "init": "finetune", "loss": "ls", "label_smoothing": 0.1, "epochs": 12, "batch_size": 64},
    },
    {
        "exp_id": "T07",
        "axis": "C. Hàm Loss",
        "diff": "Focal Loss (gamma=2.0)",
        "cfg": {"backbone": "convnext_tiny", "init": "finetune", "loss": "focal", "focal_gamma": 2.0, "epochs": 12, "batch_size": 64},
    },
    {
        "exp_id": "T08",
        "axis": "C. Hàm Loss",
        "diff": "Weighted Cross-Entropy (nghịch đảo số mẫu)",
        "cfg": {"backbone": "convnext_tiny", "init": "finetune", "loss": "ce_weighted", "class_weight_beta": 0.0, "epochs": 12, "batch_size": 64},
    },
    # Kết hợp tốt nhất
    {
        "exp_id": "T09",
        "axis": "Kết hợp tối ưu",
        "diff": "CutMix + Label Smoothing",
        "cfg": {"backbone": "convnext_tiny", "init": "finetune", "mix": "cutmix", "mix_alpha": 1.0, "loss": "ls", "label_smoothing": 0.1, "epochs": 12, "batch_size": 64},
    },
]


def main():
    print("=" * 75)
    print("🔬 BẮT ĐẦU CHẠY BƯỚC 2: KHẢO SÁT CÔNG THỨC HUẤN LUYỆN (TRAINING RECIPES)")
    print("Backbone cố định: convnext_tiny | Thiết bị:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("Khảo sát trên 3 trục: Khởi tạo, Augmentation, Hàm Loss và Kết hợp tối ưu")
    print("=" * 75)

    results = []
    total_start = time.time()
    t00_f1 = None

    for idx, item in enumerate(RECIPES, 1):
        exp_id = item["exp_id"]
        axis = item["axis"]
        diff = item["diff"]
        cfg_dict = item["cfg"]

        summary_file = Path("runs") / exp_id / "seed0" / "summary.json"
        best_ckpt = Path("runs") / exp_id / "seed0" / "best_model.pth"

        # Nếu là T00: tái sử dụng kết quả B02 nếu chưa có T00
        if exp_id == "T00" and not summary_file.exists():
            b02_dir = Path("runs") / "B02" / "seed0"
            if (b02_dir / "summary.json").exists():
                print("\n[1/10] ⏩ T00 (Mốc nền): Tự động kế thừa kết quả hoàn chỉnh từ B02...")
                Path("runs/T00/seed0").mkdir(parents=True, exist_ok=True)
                shutil.copy(b02_dir / "summary.json", summary_file)
                shutil.copy(b02_dir / "best_model.pth", best_ckpt)
                shutil.copy(b02_dir / "history.csv", Path("runs/T00/seed0/history.csv"))
                shutil.copy(b02_dir / "val_logits.npy", Path("runs/T00/seed0/val_logits.npy"))
                if Path("curves/B02_convnext_tiny.png").exists():
                    shutil.copy("curves/B02_convnext_tiny.png", "curves/T00_convnext_tiny.png")
                if Path("predictions/B02_seed0_val.csv").exists():
                    shutil.copy("predictions/B02_seed0_val.csv", "predictions/T00_seed0_val.csv")

        # Kiểm tra đã xong trước đó chưa
        res = None
        if summary_file.exists() and best_ckpt.exists():
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    res = json.load(f)
                if res.get("best_val_macro_f1") is not None:
                    print(f"\n[{idx}/10] ⏩ {exp_id} ({diff}) đã có kết quả, bỏ qua không chạy lại...")
                else:
                    res = None
            except Exception:
                res = None

        if res is None:
            print(f"\n[{idx}/10] 🔄 Đang huấn luyện {exp_id} [{axis}] - {diff}...")
            cfg = train.Config(
                exp_id=exp_id,
                seed=0,
                amp=True,
                num_workers=2,
                save_test_predictions=False,
                **cfg_dict
            )
            res = train.run(cfg)

        f1_val = round(res["best_val_macro_f1"] * 100, 2)
        top1_val = round(res["val_top1"] * 100, 2)

        if exp_id == "T00":
            t00_f1 = f1_val
            delta_str = "0.00% (Mốc)"
        else:
            delta = f1_val - (t00_f1 if t00_f1 is not None else 96.73)
            delta_str = f"{delta:+.2f}%"

        row = {
            "exp_id": exp_id,
            "axis": axis,
            "diff_from_T00": diff,
            "best_epoch": res["best_epoch"],
            "val_macro_f1": f1_val,
            "val_top1": top1_val,
            "delta_f1_vs_T00": delta_str,
            "train_sec_epoch": round(res["avg_epoch_sec"], 1),
        }
        results.append(row)
        print(f"-> Hoàn thành {exp_id}: Val Macro-F1 = {f1_val}% (Δ={delta_str}), Val Top-1 = {top1_val}%")

    total_time = (time.time() - total_start) / 60.0
    print("\n" + "=" * 75)
    print(f"🎉 HOÀN THÀNH TẤT CẢ THÍ NGHIỆM CÔNG THỨC HUẤN LUYỆN TRONG {total_time:.1f} PHÚT!")
    print("=" * 75)

    df = pd.DataFrame(results)
    out_csv = Path("runs") / "step2_training_recipes_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nBảng kết quả tổng hợp (đã lưu tại {out_csv}):\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
