# Hướng dẫn tái lập kết quả — Bài nộp Lab Day 2 (DeepWeeds)

- **Họ và tên:** Vũ Tiến Linh
- **MSSV:** 2A202602657
- **Track:** 4 · Day 2 · Deep Learning Advance

---

## 1. Môi trường & Phiên bản thư viện
Thực nghiệm được thực hiện trên GPU cục bộ (NVIDIA GeForce RTX 4060 Laptop GPU 8GB VRAM) và có thể tái lập 100% trên Google Colab / Kaggle.

- **Hệ điều hành:** Windows 11 / Linux Ubuntu
- **Python:** 3.11.9 (hoặc Python 3.10+)
- **PyTorch:** 2.6.0+cu124 (hỗ trợ CUDA 12.x)
- **torchvision:** 0.21.0+cu124
- **timm:** 1.0.15
- **pandas:** 2.2.3
- **numpy:** 2.2.3
- **scikit-learn:** 1.6.1
- **openpyxl:** 3.1.5
- **thop:** 0.1.1.post2209072238

---

## 2. Cấu trúc thư mục bài nộp

```
submissions/2A202602657_vu_tien_linh/
├── README.md          # File này (hướng dẫn chạy lại, phiên bản)
├── results.xlsx       # Đầy đủ 7 sheet: Backbones, Training, Inference, Final, PerClass, Latency, Summary
├── report.md          # Báo cáo học thuật chi tiết đầy đủ 9 phần
├── curves/            # Toàn bộ ảnh biểu đồ training (*.png) tương ứng từng exp_id
│   ├── B01_resnet50.png
│   ├── B02_convnext_tiny.png
│   ├── ...
│   ├── T09_convnext_tiny.png
│   ├── F01_convnext_tiny.png
│   └── ...
├── predictions/       # Dự đoán test & val của chung kết và mốc, từng seed
│   ├── F01_seed0_test.csv
│   ├── F01_seed1_test.csv
│   ├── F01_seed2_test.csv
│   ├── F01_uncal_seed*_test.csv
│   ├── F01_seed*_val.csv
│   ├── T00_seed0_test.csv
│   └── ...
└── code/              # Toàn bộ code hoàn thiện
    ├── dataset.py
    ├── model.py
    ├── losses.py
    ├── train.py
    ├── inference.py
    ├── benchmark.py
    └── lab_day2.ipynb
```

---

## 3. Thứ tự và Lệnh chạy lại thực nghiệm (Reproducibility)

### Bước 0: Chuẩn bị dữ liệu
Đặt ảnh vào `data/images/` và các file CSV chia sẵn của Fold 0 vào `data/labels/`.

### Bước 1: So sánh Backbone
```bash
python run_backbones.py
```
Hoặc chạy từng backbone:
```bash
python code/train.py --set exp_id=B01 backbone=resnet50 seed=0 epochs=12
python code/train.py --set exp_id=B02 backbone=convnext_tiny seed=0 epochs=12
python code/train.py --set exp_id=B03 backbone=deit_small_patch16_224 seed=0 epochs=12
python code/train.py --set exp_id=B04 backbone=mobilenetv3_large_100 seed=0 epochs=12
python code/train.py --set exp_id=B05 backbone=efficientnet_b0 seed=0 epochs=12
```

### Bước 2: Thử nghiệm công thức huấn luyện
```bash
python run_recipes.py
```

### Bước 3: Thử nghiệm suy luận & Hiệu chuẩn độ trễ
```bash
python run_inference.py
```

### Bước 4: Huấn luyện Chung kết 3 Seed & Đánh giá TEST
```bash
python run_final.py
```

### Bước 5: Chấm điểm tự động bằng eval.py
```bash
# Đo chỉ số F01 trên toàn bộ tập test
python eval.py score --pred "predictions/F01_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01_final_test --out eval_out

# Tự chấm điểm phần I của RUBRIC
python eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/T00_seed*_test.csv" --uncal "predictions/F01_uncal_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv" --latency-p95-ms 4.83 --latency-method proper --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --out eval_out
```

---

## 4. Các hạt giống ngẫu nhiên (Seeds) đã sử dụng
- Các thí nghiệm sàng lọc (Bước 1, 2, 3): `seed = 0`
- Cấu hình chung kết F01 và mốc T00 (Bước 4): `seed = 0, 1, 2` (mean ± std qua 3 seed)
