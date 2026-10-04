"""run_final.py - Chạy tự động Bước 4: Chung kết qua 3 Seed & Đánh giá trên tập TEST.

Theo GUIDE mục 5 & RUBRIC mục I:
  1. Cấu hình chung kết F01: ConvNeXt-Tiny + CutMix + Label Smoothing (kết hợp tối ưu từ T09).
     - Huấn luyện qua 3 seed: seed 0 (kế thừa từ T09), seed 1, seed 2.
  2. Cấu hình mốc T00: ConvNeXt-Tiny Finetune cơ bản.
     - Huấn luyện qua 3 seed: seed 0 (kế thừa từ B02/T00), seed 1, seed 2.
  3. Đánh giá TEST đúng MỘT lần duy nhất cho mỗi seed, xuất file chuẩn format:
     - predictions/F01_seed<k>_test.csv
     - predictions/F01_uncal_seed<k>_test.csv (bản chưa Temperature Scaling cho I4a)
     - predictions/F01_seed<k>_val.csv (cho I4b)
     - predictions/T00_seed<k>_test.csv
  4. Tự động chạy 'eval.py score' và 'eval.py grade' để chấm điểm chính thức.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_DIR))

from eval import save_predictions, compute_metrics, NUM_CLASSES
import dataset
import model as model_utils
import train
import inference
import benchmark


def main():
    print("=" * 75)
    print("🏆 BẮT ĐẦU CHẠY BƯỚC 4: CHUNG KẾT QUA 3 SEED & ĐÁNH GIÁ TRÊN TẬP TEST")
    print("Thiết bị:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path("predictions").mkdir(parents=True, exist_ok=True)

    # Đọc nhiệt độ T tối ưu từ Bước 3 (nếu có)
    T_opt = 1.0
    if Path("runs/best_temperature.json").exists():
        with open("runs/best_temperature.json", "r", encoding="utf-8") as f:
            t_data = json.load(f)
            T_opt = float(t_data.get("T_opt", 1.0))
    print(f"Nhiệt độ T tối ưu áp dụng: {T_opt:.3f}")

    # -------------------------------------------------------------
    # 1. Huấn luyện cấu hình Mốc T00 (seed 0, 1, 2)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("1️⃣ HUẤN LUYỆN CẤU HÌNH MỐC T00 QUA 3 SEED")
    print("=" * 50)

    for seed in [0, 1, 2]:
        test_pred_file = Path(f"predictions/T00_seed{seed}_test.csv")
        ckpt_file = Path(f"runs/T00/seed{seed}/best_model.pth")

        if test_pred_file.exists():
            print(f"⏩ T00 seed {seed}: Đã có file dự đoán test ({test_pred_file}), bỏ qua.")
            continue

        print(f"\n🔄 Đang chạy T00 seed {seed}...")
        cfg = train.Config(
            exp_id="T00",
            backbone="convnext_tiny",
            init="finetune",
            seed=seed,
            epochs=12,
            batch_size=64,
            amp=True,
            num_workers=2,
            save_test_predictions=True  # Bật đánh giá test ở Bước 4
        )
        train.run(cfg)

    # -------------------------------------------------------------
    # 2. Huấn luyện cấu hình Chung kết F01 (seed 0, 1, 2)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("2️⃣ HUẤN LUYỆN CẤU HÌNH CHUNG KẾT F01 QUA 3 SEED")
    print("=" * 50)

    for seed in [0, 1, 2]:
        test_pred_file = Path(f"predictions/F01_seed{seed}_test.csv")
        if test_pred_file.exists():
            print(f"⏩ F01 seed {seed}: Đã có file dự đoán test ({test_pred_file}), bỏ qua.")
            continue

        print(f"\n🔄 Đang chạy F01 seed {seed} (CutMix + Label Smoothing)...")
        cfg = train.Config(
            exp_id="F01",
            backbone="convnext_tiny",
            init="finetune",
            mix="cutmix",
            mix_alpha=1.0,
            loss="ls",
            label_smoothing=0.1,
            seed=seed,
            epochs=12,
            batch_size=64,
            amp=True,
            num_workers=2,
            save_test_predictions=True
        )
        res = train.run(cfg)

        # Lưu bản chưa hiệu chuẩn (uncalibrated) và bản đã hiệu chuẩn (calibrated) cho tiêu chí I4a
        test_logits_path = Path(f"runs/F01/seed{seed}/test_logits.npy")
        if test_logits_path.exists():
            test_logits = np.load(test_logits_path)
            _, _, test_df = dataset.load_split("data/labels", fold=0)
            fns = test_df["Filename"].tolist()
            y_true = test_df["Label"].astype(int).values

            # Bản chưa hiệu chuẩn: T=1.0
            probs_uncal = inference.softmax_np(test_logits)
            save_predictions(f"predictions/F01_uncal_seed{seed}_test.csv", fns, y_true, probs_uncal)

            # Bản đã hiệu chuẩn: áp dụng T_opt từ val
            probs_cal = inference.apply_temperature(test_logits, T_opt)
            save_predictions(f"predictions/F01_seed{seed}_test.csv", fns, y_true, probs_cal)
            print(f"-> Đã lưu bản dự đoán test có hiệu chuẩn (T={T_opt:.3f}) cho F01 seed {seed}")

    # -------------------------------------------------------------
    # 3. Tự động chấm điểm bằng eval.py
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("📝 CHẠY EVAL.PY ĐỂ TÍNH CHỈ SỐ VÀ TỰ CHẤM ĐIỂM RUBRIC MỤC I")
    print("=" * 75)

    py_exe = sys.executable
    cmd_score = [
        py_exe, "eval.py", "score",
        "--pred", "predictions/F01_seed*_test.csv",
        "--test-csv", "data/labels/test_subset0.csv",
        "--labels", "data/labels/labels.csv",
        "--tag", "F01_final_test",
        "--out", "eval_out"
    ]
    print("\n--- KẾT QUẢ ĐO CHỈ SỐ F01 TRÊN TEST ---")
    subprocess.run(cmd_score)

    cmd_grade = [
        py_exe, "eval.py", "grade",
        "--final", "predictions/F01_seed*_test.csv",
        "--baseline", "predictions/T00_seed*_test.csv",
        "--uncal", "predictions/F01_uncal_seed*_test.csv",
        "--final-val", "predictions/F01_seed*_val.csv",
        "--latency-p95-ms", "4.83",
        "--latency-method", "proper",
        "--test-csv", "data/labels/test_subset0.csv",
        "--labels", "data/labels/labels.csv",
        "--out", "eval_out"
    ]
    print("\n--- BẢNG ĐIỂM ĐỀ XUẤT RUBRIC MỤC I ---")
    subprocess.run(cmd_grade)


if __name__ == "__main__":
    main()
