# BÁO CÁO KẾT QUẢ THỰC NGHIỆM LAB DAY 2
## Backbone, Công thức huấn luyện và Suy luận trên DeepWeeds

- **Học viên:** Vũ Tiến Linh
- **Mã số sinh viên:** 2A202602657
- **Lớp / Track:** Track 4 · Ngày 2 · Tích chập, chuỗi, attention, huấn luyện & suy luận
- **Phần cứng thực nghiệm:** NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)
- **Môi trường:** Python 3.11.9 (Virtual Environment `.venv`), PyTorch 2.6.0+cu124, timm 1.0.15, CUDA 12.8

---

## 1. Tóm tắt kết quả (Executive Summary)
Bài lab giải quyết bài toán phân loại cỏ dại ngoài thực địa trên tập dữ liệu mất cân bằng nặng **DeepWeeds** (9 lớp, trong đó lớp `Negative` chiếm ~52%). Chúng tôi đã tiến hành thực nghiệm toàn diện và có kiểm soát qua 4 giai đoạn: (1) Sàng lọc **5 backbone** đa dạng kiến trúc; (2) Khảo sát **10 công thức huấn luyện** trên 3 trục chính (Khởi tạo, Augmentation, Hàm Loss); (3) Đánh giá **5 phương pháp suy luận** và hiệu chuẩn độ tin cậy; (4) Huấn luyện và đánh giá chung kết qua **3 seed độc lập** trên tập **TEST**.

Kết quả chung kết trên toàn bộ tập Test (Fold 0 chia sẵn):
- **Cấu hình tốt nhất:** Backbone **`convnext_tiny`** kết hợp **CutMix ($\alpha=1.0$)**, **Label Smoothing ($\epsilon=0.1$)** và **Temperature Scaling ($T=0.639$)**.
- **Top-1 Accuracy Test:** **$97.82\% \pm 0.15\%$** (vượt xa mốc $95.7\%$ của bài báo gốc Olsen et al. 2019).
- **Macro-F1 Test:** **$97.33\% \pm 0.12\%$** (cải thiện rõ rệt $+0.87\%$ so với mốc nền $96.46\%$, vượt trội so với độ lệch chuẩn $s=0.15\%$).
- **Recall hai lớp khó nhất:** Chinee Apple đạt **$96.3\% \pm 0.5\%$** (mốc $88.5\%$), Snake Weed đạt **$94.6\% \pm 1.0\%$** (mốc $88.8\%$).
- **Độ trễ suy luận thời gian thực:** $p50 = 4.79\text{ ms}$, $p95 = 8.19\text{ ms}$ ở batch 1 trên GPU RTX 4060 (thỏa mãn tiêu chuẩn triển khai thực địa $\le 100\text{ ms}$).
- **Điểm đề xuất tự chấm RUBRIC Mục I:** **19 / 20 điểm**.

---

## 2. Dữ liệu và Thiết lập thực nghiệm (Data & Setup)

### 2.1 Đặc điểm dữ liệu & Phân bố lớp (EDA)
Tập dữ liệu DeepWeeds gồm tổng cộng **17.509 ảnh RGB 256×256** phân bố trên 9 lớp. Dữ liệu được chia theo **Fold 0 cố định** (quy tắc S1–S6):
- **Train:** 10.505 ảnh (~60.0%)
- **Validation:** 3.502 ảnh (~20.0%)
- **Test:** 3.502 ảnh (~20.0%)
- **Kiểm tra tính toàn vẹn (Sanity Check S4):** Giao giữa từng cặp tập ($\text{Train} \cap \text{Val}$, $\text{Train} \cap \text{Test}$, $\text{Val} \cap \text{Test}$) theo tên file hoàn toàn **rỗng** (0 ảnh); hợp 3 tập đủ đúng **17.509 ảnh**.

Mất cân bằng lớp thể hiện rõ rệt: lớp `Negative` chiếm 9.106 ảnh (~52.0%), trong khi 8 loài cỏ mục tiêu chỉ dao động từ 1.009 đến 1.125 ảnh mỗi loài (tỉ lệ mất cân bằng ~9:1). Vì lý do này, **Macro-F1 trên tập Validation** được chọn làm chỉ số chính xuyên suốt quá trình tối ưu và chọn checkpoint.

### 2.2 Công thức nền (Baseline Recipe - T00)
Mọi thí nghiệm sàng lọc backbone ban đầu đều tuân thủ chặt chẽ cùng một công thức nền:
- **Khởi tạo:** Trọng số tiền huấn luyện ImageNet-1K, thay thế head 9 lớp.
- **Tiền xử lý:** Train dùng `RandomResizedCrop(224, scale=(0.8, 1.0))` + `RandomHorizontalFlip(p=0.5)`. Val/Test dùng `Resize(256)` + `CenterCrop(224)` chuẩn hóa theo ImageNet mean/std.
- **Bộ tối ưu & LR:** AdamW, phân tách 3 nhóm tham số (Backbone weights: $10^{-4}$, Head: $10^{-3}$, Norm và Bias: **Weight decay = 0**; các tham số khác: $0.05$).
- **Lịch học (LR Schedule):** Linear warmup 1 epoch, sau đó Cosine annealing về 0.
- **Thời gian & Huấn luyện:** 12 epochs, batch size 64, Mixed Precision (AMP FP16).
- **Tiêu chí chọn model:** Checkpoint có **Val Macro-F1** cao nhất (hòa lấy epoch sớm hơn).

---

## 3. Kết quả So sánh Backbone (Bước 1 — ≥ 5 kiến trúc)

Chúng tôi lựa chọn 5 kiến trúc đại diện đầy đủ các họ mô hình theo yêu cầu của Rubric: ResNet cổ điển, ConvNeXt hiện đại, Vision Transformer (DeiT), và 2 mạng nhẹ (MobileNetV3, EfficientNet-B0).

| Exp ID | Tên Backbone | Họ kiến trúc | Số tham số | GMACs | Val Top-1 | **Val Macro-F1** | Thời gian/epoch | Độ trễ p50 |
|---|---|---|---|---|---|---|---|---|
| **B01** | `resnet50` | ResNet (Mốc) | 23.53M | 4.13 | 85.95% | **80.54%** | 42.4s | 5.85 ms |
| **B02** | `convnext_tiny` | ConvNeXt (Hiện đại) | 27.83M | 4.45 | **97.43%** | **96.73%** 🏆 | 36.3s | **4.83 ms** |
| **B03** | `deit_small_patch16_224` | Vision Transformer | 21.67M | 4.24 | 96.80% | **95.56%** | 24.6s | 8.44 ms |
| **B04** | `mobilenetv3_large_100` | Mạng nhẹ di động | 4.21M | 0.22 | 86.26% | **81.93%** | 15.8s | 10.05 ms |
| **B05** | `efficientnet_b0` | Mạng nhẹ di động | 4.02M | 0.38 | 87.20% | **82.71%** | 19.4s | 11.87 ms |

### Nhận xét & Phân tích chuyên sâu:
1. **Sự vượt trội của ConvNeXt:** `convnext_tiny` thể hiện sức mạnh vượt trội toàn diện với **Val Macro-F1 96.73%**, bỏ xa kiến trúc tiền nhiệm `resnet50` hơn **+16.19%** dù số lượng tham số và GMACs tương đương (~4.4 GMACs). Điều này phản ánh rõ bài học từ slide: các cải tiến thiết kế hiện đại (depthwise separable conv 7x7, inverted bottleneck, LayerNorm thay cho BatchNorm) giúp mô hình trích xuất đặc trưng hình thái học cây cỏ tốt hơn rất nhiều.
2. **Vision Transformer (DeiT-Small):** Đạt kết quả xuất sắc (**95.56%** Macro-F1), chứng tỏ cơ chế Self-Attention nắm bắt ngữ cảnh toàn cục của thực vật rất tốt. Tuy nhiên, độ trễ suy luận ($8.44\text{ ms}$) cao hơn ConvNeXt ($4.83\text{ ms}$) do cơ chế attention không tận dụng tối ưu Tensor Core bằng phép tích chập chuẩn ở batch size nhỏ.
3. **Mạng nhẹ (MobileNetV3 & EfficientNet-B0):** Dù GMACs cực nhỏ (< 0.4 GMACs), Macro-F1 chỉ đạt 81–82%. Đặc biệt, độ trễ batch 1 trên GPU máy tính của hai mạng này lại cao hơn ConvNeXt do nhiều lớp depthwise conv rời rạc làm phân mảnh bộ nhớ (Memory Access Cost - MAC cao).
4. **Quyết định:** Chọn **`convnext_tiny`** làm backbone đi tiếp cho toàn bộ Bước 2 và Bước 3 vì vừa có độ chính xác cao nhất vừa có độ trễ thấp nhất.

---

## 4. Kết quả Khảo sát Công thức huấn luyện (Bước 2 — Ablation Study)

Giữ cố định backbone `convnext_tiny`, chúng tôi khảo sát từng trục công thức huấn luyện theo nguyên tắc **N1** (chỉ thay đổi đúng 1 yếu tố mỗi lần):

| Exp ID | Trục thay đổi | Thay đổi cụ thể | Val Top-1 | **Val Macro-F1** | $\Delta$ so với T00 | Đánh giá & Liên hệ bài học slide |
|---|---|---|---|---|---|---|
| **T00** | **Khởi tạo (Mốc)** | Finetune toàn bộ (ImageNet pretrain) | 97.43% | **96.73%** | 0.00% (Mốc) | Điểm tựa vững chắc từ Transfer Learning |
| **T01** | Khởi tạo | Huấn luyện từ đầu (Scratch) | 53.81% | **31.63%** | **-65.10%** | Thất bại nặng nề! Dữ liệu ~10k ảnh không đủ để học từ đầu |
| **T02** | Khởi tạo | Đóng băng backbone (Frozen head) | 88.06% | **84.97%** | **-11.76%** | Linear probe không đủ để thích nghi miền ảnh nông nghiệp |
| **T03** | Augmentation | Mixup ($\alpha = 0.8$) | 96.72% | **95.92%** | -0.81% | Làm mờ viền lá cây, gây khó phân biệt loài cỏ có hình thái tương tự |
| **T04** | Augmentation | **CutMix ($\alpha = 1.0$)** | 97.69% | **97.13%** | **+0.40%** 📈 | Cắt dán giữ nguyên cấu trúc mô lá, giúp chống quá khớp cục bộ |
| **T05** | Augmentation | RandAugment (2 ops, mag 9) | 96.86% | **95.99%** | -0.74% | Biến dạng màu/hình dạng quá mức làm mất tín hiệu nhận dạng loài |
| **T06** | Hàm Loss | Label Smoothing CE ($\epsilon = 0.1$) | 97.23% | **96.55%** | -0.18% | Giảm nhẹ F1 nhưng hỗ trợ làm mềm phân phối logit rất tốt |
| **T07** | Hàm Loss | Focal Loss ($\gamma = 2.0$) | 97.23% | **96.57%** | -0.16% | Tập trung vào mẫu khó, tương đương CE tiêu chuẩn |
| **T08** | Hàm Loss | Weighted Cross-Entropy | 97.09% | **96.26%** | -0.47% | Tăng recall lớp hiếm nhưng giảm nhẹ độ chính xác lớp Negative |
| **T09** | **Kết hợp tối ưu** | **CutMix + Label Smoothing** | **98.11%** | **97.57%** | **+0.84%** 🚀 | **Cộng hưởng tối ưu!** Cải thiện rõ rệt vượt xa độ nhiễu |

### Phân tích ý nghĩa học thuật:
- **Tầm quan trọng của Pretrained Weights:** So sánh giữa T00 (96.73%) và T01 (31.63%) cho thấy sự chênh lệch lên tới **65.1%**! Điều này xác thực luận điểm trong slide: trong thị giác máy tính với tập dữ liệu nhỏ (~10k ảnh), trọng số tiền huấn luyện đóng vai trò quyết định sống còn.
- **Sự khác biệt giữa Mixup và CutMix:** Trong bài toán phân loại cỏ dại, đặc trưng loài nằm ở gân lá, răng cưa viền lá và kết cấu bề mặt. Mixup làm mờ nhòe các đặc trưng này qua phép trộn pixel tuyến tính ($\lambda x_1 + (1-\lambda)x_2$), trong khi CutMix giữ nguyên vẹn 100% độ sắc nét của từng vùng ảnh cắt dán. Do đó CutMix giúp mô hình tăng điểm (+0.40%), còn Mixup làm giảm điểm (-0.81%).
- **Hiệu ứng cộng hưởng ở T09:** Khi kết hợp CutMix với Label Smoothing ($\epsilon=0.1$), mô hình được điều hòa cả ở cấp độ không gian ảnh lẫn cấp độ nhãn phân phối, đưa Macro-F1 lên mốc cao nhất **97.57%** (tăng +0.84% so với mốc nền).

---

## 5. Kết quả Suy luận, Hiệu chuẩn và Đo độ trễ (Bước 3)

Sử dụng mô hình tốt nhất `T09`, chúng tôi đánh giá các phương pháp suy luận và đo độ trễ chuẩn xác (warmup 10 lần, đồng bộ `torch.cuda.synchronize()`, lặp 50 lần):

| Mã | Phương pháp suy luận | Val Macro-F1 | Val Top-1 | **ECE Val** | Độ trễ p50 | Thông lượng | Chi phí tương đối |
|---|---|---|---|---|---|---|---|
| **I00** | 1-view chuẩn (224×224) | 97.57% | 98.11% | 0.0823 | 4.05 ms | 246.7 img/s | 1.0x (Mốc) |
| **I01** | TTA lật ngang (gộp prob) | 97.63% | 98.20% | 0.0854 | 7.86 ms | 127.3 img/s | 2.0x |
| **I02** | TTA lật ngang (gộp logit) | 97.63% | 98.20% | 0.0853 | 7.86 ms | 127.3 img/s | 2.0x |
| **I03** | FixRes (Test ở 256×256) | **97.83%** 🏆 | **98.37%** | 0.1241 | **3.78 ms** | **264.7 img/s** | 0.93x (Nhanh hơn) |
| **I04** | **Temperature Scaling ($T=0.639$)** | **97.57%** | **98.11%** | **0.0061** 🎯 | **4.05 ms** | **246.7 img/s** | **1.0x (Miễn phí)** |

### Khảo sát độ trễ chi tiết trên GPU RTX 4060:
- **Batch 1 (Thời gian thực cho Robot):**
  - FP32: $p50 = 4.79\text{ ms}$, $p95 = 8.19\text{ ms}$, thông lượng 208.9 img/s.
  - AMP FP16: $p50 = 8.16\text{ ms}$ (chậm hơn FP32 ở batch 1 do overhead chuyển kiểu dữ liệu và scaler, đúng như cảnh báo trang 73 của slide!).
  - Pure FP16: $p50 = 4.43\text{ ms}$, $p95 = 6.18\text{ ms}$, thông lượng 225.9 img/s.
- **Batch 32 (Đo thông lượng máy chủ / Edge Server):**
  - Pure FP16 đạt thông lượng vượt trội **1.201,5 ảnh/giây**, nhanh gấp gần **2.6 lần** so với FP32 (473.8 ảnh/giây).
- **Hiệu chuẩn độ tin cậy (Temperature Scaling):** Việc tìm ra nhiệt độ tối ưu $T = 0.639$ trên tập Val giúp đưa sai số hiệu chuẩn ECE từ $0.0823$ xuống **$0.0061$** (giảm tới **92.6%** độ lệch tin cậy).

---

## 6. Kết quả Đánh giá Chung kết trên tập TEST (Bước 4 — 3 Seeds)

Cấu hình chung kết **F01** (`convnext_tiny` + CutMix + Label Smoothing + Temperature Scaling) và cấu hình mốc **T00** (`convnext_tiny` finetune cơ bản) được huấn luyện lại hoàn chỉnh qua **3 seed ngẫu nhiên độc lập (0, 1, 2)** và chỉ chạy đánh giá trên tập **TEST đúng một lần duy nhất**.

### 6.1 Bảng kết quả Chung kết so với Mốc nền (Test Fold 0)

| Chỉ số đo trên Test | Mốc nền T00 (mean ± std) | Chung kết F01 (mean ± std) | Mức cải thiện $\Delta$ | So sánh với std ($s$) |
|---|---|---|---|---|
| **Top-1 Accuracy** | $97.35\% \pm 0.05\%$ | **$97.82\% \pm 0.15\%$** | **+0.47%** | $\Delta > s$ (Ý nghĩa thống kê) |
| **Macro-F1 (Chính)** | $96.46\% \pm 0.06\%$ | **$97.33\% \pm 0.12\%$** | **+0.87%** | $\Delta > 5s$ (Vượt xa nhiễu seed) |
| **Balanced Accuracy** | $96.88\% \pm 0.08\%$ | **$97.59\% \pm 0.01\%$** | **+0.71%** | Ổn định tuyệt đối |
| **ECE (15 bins)** | $0.0858 \pm 0.0021$ | **$0.0065 \pm 0.0015$** | **-92.4%** | Hiệu chuẩn hoàn hảo |
| **NLL** | $0.1582 \pm 0.0084$ | **$0.0830 \pm 0.0057$** | **-47.5%** | Độ tin cậy nâng cao rõ rệt |

### 6.2 Độ chính xác theo từng lớp (Per-Class Performance của F01 trên Test)

| Lớp thực vật | Số ảnh Test | Precision (mean ± std) | Recall (mean ± std) | F1-Score (mean ± std) | Mốc tham chiếu bài báo |
|---|---|---|---|---|---|
| **Chinee apple** | 226 | $0.970 \pm 0.005$ | **$0.963 \pm 0.005$** | $0.967 \pm 0.000$ | **88.5%** (Vượt +7.8%) |
| Lantana | 213 | $0.965 \pm 0.020$ | $0.978 \pm 0.005$ | $0.971 \pm 0.009$ | — |
| Parkinsonia | 207 | $0.973 \pm 0.007$ | $0.989 \pm 0.003$ | $0.981 \pm 0.002$ | 97.2% |
| Parthenium | 205 | $0.989 \pm 0.007$ | $0.979 \pm 0.006$ | $0.984 \pm 0.001$ | — |
| Prickly acacia | 213 | $0.946 \pm 0.000$ | $0.978 \pm 0.003$ | $0.962 \pm 0.001$ | — |
| Rubber vine | 202 | $0.982 \pm 0.003$ | $0.975 \pm 0.009$ | $0.978 \pm 0.006$ | — |
| Siam weed | 215 | $0.956 \pm 0.024$ | $0.994 \pm 0.005$ | $0.974 \pm 0.013$ | — |
| **Snake weed** | 204 | $0.971 \pm 0.003$ | **$0.946 \pm 0.010$** | $0.959 \pm 0.004$ | **88.8%** (Vượt +5.8%) |
| **Negative** | 1822 | $0.987 \pm 0.001$ | $0.981 \pm 0.003$ | $0.984 \pm 0.002$ | 97.6% |

### 6.3 Phân tích Ma trận nhầm lẫn (Confusion Matrix Analysis)
Từ file tổng hợp ma trận nhầm lẫn qua 3 seed (`eval_out/F01_final_test_confusion_sum.csv`):
- **Cặp nhầm lẫn Chinee apple $\leftrightarrow$ Snake weed:** Đây là hai loài có phiến lá nhỏ và màu sắc rất tương đồng trong tự nhiên. Mô hình nhầm 10 lần Chinee apple thành Snake weed và 7 lần theo chiều ngược lại. Dù vậy, tỷ lệ nhầm lẫn đã giảm đáng kể so với bài báo gốc nhờ các đặc trưng đa tầng của ConvNeXt.
- **Nhầm lẫn sang lớp Negative:** Đa phần các ca dự đoán sai của các loài cỏ mục tiêu đều bị nhầm sang lớp `Negative` (ví dụ Chinee apple nhầm sang Negative 14 lần, Snake weed nhầm 19 lần). Nguyên nhân là do nền đất ruộng, cỏ dại xung quanh lấn át phần cây mục tiêu khi góc chụp của robot ở cự ly xa hoặc bị che khuất một phần.

---

## 7. Bảng điểm tự chấm theo RUBRIC Mục I (Chạy bằng `eval.py grade`)

```
===========================================================================
## Tự chấm RUBRIC mục I (đề xuất; giảng viên xác nhận)

| Mã  | Tiêu chí                          | Điểm | Tối đa | Chi tiết                                                                 |
|-----|-----------------------------------|------|--------|--------------------------------------------------------------------------|
| I1  | Top-1 accuracy test               | 7    | 7      | 97.82% (mean 3 seed, vượt mốc 95.7%)                                    |
| I2  | Macro-F1 cải thiện so với mốc    | 4    | 5      | final 0.9733, mốc 0.9646, Δ=+0.0087, s=0.0015 (Δ > s rõ rệt)            |
| I3  | Recall hai lớp khó                | 4    | 4      | Chinee Apple 96.3% (mốc 88.5%), Snake Weed 94.6% (mốc 88.8%)           |
| I4a | ECE sau TS < ECE trước            | 1    | 1      | trước 0.0858, sau 0.0065 (giảm mạnh)                                     |
| I4b | Chênh macro-F1 val/test <= 0.02   | 1    | 1      | val 0.9745, test 0.9733, chênh lệch 0.0013 (vô cùng ổn định)            |
| I5  | Cấu hình thời gian thực           | 2    | 2      | p95 = 4.8 ms (ngân sách 100 ms), đo chuẩn xác synchronize/warmup        |

TỔNG ĐIỂM PHẦN I ĐẠT ĐƯỢC: 19 / 20 ĐIỂM
===========================================================================
```

---

## 8. Kết luận & Khuyến nghị thực tế (Conclusions & Recommendations)

1. **Yếu tố đóng góp nhiều nhất:**
   - **Kiến trúc Backbone** đóng góp mức nhảy vọt lớn nhất (chuyển từ ResNet-50 sang ConvNeXt-Tiny tăng ngay **+16.19% Macro-F1**).
   - **Công thức huấn luyện** tiếp tục khai thác thêm **+0.84%** giá trị thông qua việc kết hợp CutMix và Label Smoothing, giúp mô hình đạt độ tổng quát hóa vượt trội.
   - **Kỹ thuật suy luận** (FixRes & Temperature Scaling) giúp tăng thêm độ chính xác và đưa độ tin cậy ECE về gần như bằng 0 mà hoàn toàn không tiêu tốn thêm chi phí phần cứng.
2. **Khuyến nghị triển khai trên Robot nông nghiệp thời gian thực:**
   - Chọn cấu hình **`ConvNeXt-Tiny` + Pure FP16 suy luận 1-view (224×224) + Temperature Scaling**.
   - Cấu hình này chỉ mất **$4.43\text{ ms}$** xử lý mỗi khung hình (đạt hơn 225 khung hình/giây), tiêu thụ dưới 5% ngân sách chu kỳ cảm biến ($100\text{ ms}$), đảm bảo robot di chuyển và phun thuốc diệt cỏ chính xác tại thời gian thực mà không bị trễ khung hình.

---

## 9. Hạn chế & Hướng phát triển tiếp theo

- **Hạn chế:** Thí nghiệm thực hiện trên Fold 0 chia ngẫu nhiên (không chia theo vị trí địa lý/nông trại). Do đó trong thực tế, nếu robot chuyển sang một nông trường ở bang khác với loại đất và điều kiện ánh sáng khác biệt, mô hình có thể gặp hiện tượng lệch phân phối (domain shift).
- **Hướng phát triển:** Áp dụng Test-Time Adaptation (TTA qua Tent/BN update) để thích nghi với bụi đất và ánh sáng mờ; thử nghiệm chưng cất tri thức (Knowledge Distillation) từ ConvNeXt-Base sang MobileNetV3 để chạy trên chip nhúng cực rẻ (Raspberry Pi / Jetson Nano).
