# Kết quả đánh giá

Tài liệu này ghi kết quả từ lần chạy Colab trên 1.006 ảnh train và 336 ảnh test. Các bảng phân biệt điểm trung bình qua 5 fold với điểm tính gộp từ các dự đoán ngoài fold.

## Huấn luyện và cross-validation

| Mô hình | `class_weight` | Tham số được chọn | Accuracy train | Accuracy CV | Macro-F1 train | Macro-F1 CV | Độ lệch chuẩn F1 CV |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| SVM | `none` | `C` = 100, `gamma` = 0,0003 | 0,997 | 0,860 | 0,996 | 0,820 | 0,0205 |
| **SVM** | **`balanced`** | `C` = 10, `gamma` = 0,001 | 0,987 | 0,859 | 0,984 | 0,821 | 0,0119 |
| Decision Tree | `none` | `max_depth` = 10 | 0,913 | 0,754 | 0,894 | 0,713 | 0,0264 |
| Decision Tree | `balanced` | `max_depth` = `null` | 1,000 | 0,754 | 1,000 | 0,709 | 0,0306 |

`min_samples_leaf` của từng cây nằm trong cột `best_params` của `cv_results.csv`. Bộ số liệu Colab dùng cho bảng này chỉ ghi `max_depth`.

SVM `balanced` được chốt theo macro-F1 CV trung bình. Dự đoán ngoài fold gộp của ứng viên này đạt accuracy **0,8588** và macro-F1 **0,8214**.

## So sánh train, validation và test

Validation là giá trị trung bình qua các fold trong tập train. SVM `balanced` là ứng viên chốt bằng CV. Tập test chỉ dùng để báo cáo sau khi chọn mô hình.

| Mô hình | `class_weight` | Accuracy train | Accuracy val | Accuracy test | Macro-F1 train | Macro-F1 val | Macro-F1 test |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SVM | `none` | 0,997 | 0,860 | 0,836 | 0,996 | 0,820 | 0,795 |
| **SVM** | **`balanced`** | 0,987 | 0,859 | **0,860** | 0,984 | 0,821 | **0,828** |
| Decision Tree | `none` | 0,913 | 0,754 | 0,783 | 0,894 | 0,713 | 0,744 |
| Decision Tree | `balanced` | 1,000 | 0,754 | 0,738 | 1,000 | 0,709 | 0,698 |

## SVM `balanced` theo từng lớp

| Lớp | Precision CV | Recall CV | F1 CV | Số mẫu train | Precision test | Recall test | F1 test | Số mẫu test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `healthy` | 0,8571 | 0,9057 | 0,8807 | 106 | 0,8611 | 0,8611 | 0,8611 | 36 |
| `leaf_miner` | 0,8085 | 0,8000 | 0,8042 | 190 | 0,8361 | 0,8095 | 0,8226 | 63 |
| `rust` | 0,9075 | 0,8997 | 0,9036 | 349 | 0,9123 | 0,8966 | 0,9043 | 116 |
| `brown_spot` | 0,9419 | 0,9382 | 0,9400 | 259 | 0,9875 | 0,9080 | 0,9461 | 87 |
| `cercospora` | 0,5784 | 0,5784 | 0,5784 | 102 | 0,5333 | 0,7059 | 0,6076 | 34 |

Điểm test theo lớp được tính từ ma trận test bên dưới. Precision bằng số dự đoán đúng lớp chia cho tổng cột. Recall bằng số dự đoán đúng lớp chia cho tổng hàng. F1 bằng hai lần tích precision và recall chia cho tổng của chúng, tính trước khi làm tròn.

### Ma trận nhầm lẫn CV

Hàng là lớp thật, cột là lớp dự đoán. Ma trận gộp dự đoán ngoài fold của 1.006 ảnh train.

| Lớp thật | `healthy` | `leaf_miner` | `rust` | `brown_spot` | `cercospora` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `healthy` | 96 | 4 | 4 | 0 | 2 |
| `leaf_miner` | 6 | 152 | 9 | 8 | 15 |
| `rust` | 7 | 6 | 314 | 2 | 20 |
| `brown_spot` | 1 | 9 | 0 | 243 | 6 |
| `cercospora` | 2 | 17 | 19 | 5 | 59 |

### Ma trận nhầm lẫn test

Hàng là lớp thật, cột là lớp dự đoán trên 336 ảnh test.

| Lớp thật | `healthy` | `leaf_miner` | `rust` | `brown_spot` | `cercospora` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `healthy` | 31 | 2 | 1 | 0 | 2 |
| `leaf_miner` | 1 | 51 | 5 | 0 | 6 |
| `rust` | 3 | 0 | 104 | 0 | 9 |
| `brown_spot` | 0 | 4 | 0 | 79 | 4 |
| `cercospora` | 1 | 4 | 4 | 1 | 24 |

## Độ quan trọng đặc trưng của cây

`decision_tree_none` được chọn bằng CV trong số hai cây. Bảng đọc `feature_importances_` từ cây đã fit trên train. Độ quan trọng đo mức giảm độ hỗn tạp tại các nút chia, chuẩn hóa về tổng 1.

| Nhóm | Độ quan trọng | Vùng | Độ quan trọng |
| --- | ---: | --- | ---: |
| Màu | 0,717 | Toàn lá | 0,742 |
| Kết cấu | 0,229 | Vùng nghi tổn thương | 0,258 |
| Gradient | 0,054 | | |
| Hình dạng | 0,000 | | |

| Đặc trưng | Độ quan trọng |
| --- | ---: |
| `lesion_lab_std_l` | 0,192 |
| `leaf_ab_hist_a05_b07` | 0,168 |
| `leaf_ab_hist_a05_b04` | 0,116 |
| `leaf_ab_hist_a06_b06` | 0,055 |
| `leaf_lbp_a_r2_04` | 0,053 |
| `leaf_lbp_a_r1_04` | 0,048 |
| `leaf_ab_hist_a05_b06` | 0,043 |
| `leaf_ab_hist_a06_b05` | 0,037 |
| `leaf_ab_hist_a04_b08` | 0,026 |
| `leaf_ab_hist_a02_b07` | 0,022 |

Đây là độ quan trọng của một cây đơn lẻ. Các đặc trưng tương quan có thể chia sẻ độ quan trọng. Hình dạng có giá trị 0,000 vì cây này không dùng nhóm đó khi chia nhánh. Giá trị với SVM cần được đo riêng.

## Phân tích

- **So sánh ứng viên.** SVM hơn Decision Tree khoảng 0,11 macro-F1 trên CV ở lần chạy này. Hai bản trọng số của SVM chỉ chênh 0,001, nhỏ hơn độ dao động giữa các fold. Bản `balanced` có macro-F1 CV trung bình cao hơn theo quy tắc chọn và độ lệch chuẩn thấp hơn. Ghi nhận Colab cho biết mở lưới `gamma` xuống 0,0001 gần như không đổi điểm.
- **Quá khớp.** SVM `balanced` có khoảng cách train–validation là 0,163 macro-F1, cho thấy quá khớp ở mức vừa phải. Cây `balanced` đạt 1,000 trên train và 0,698 trên test, phù hợp với việc học quá sát tập train. Cây này không được chọn.
- **Độ tin cậy.** Macro-F1 test 0,828 gần với CV 0,821. Mức chênh này không gợi dấu hiệu rò rỉ qua điểm số, nhưng sự gần nhau không đủ để loại trừ rủi ro. Pipeline chỉ dùng train để fit và chọn ứng viên. Việc thiếu ID cây và phiên chụp giới hạn ý nghĩa của cả hai kết quả.
- **Lớp khó.** `cercospora` có F1 thấp nhất trên CV (0,5784) và test (0,6076). Trong CV, 19 ảnh của lớp này bị dự đoán thành `rust` và 17 thành `leaf_miner`. Trên test có 4 ảnh nhầm sang mỗi lớp đó. Trong CSV gốc, 52 trong 147 dòng `cercospora` còn bật cờ `rust`, cho thấy nhãn chính có thể che khuất dấu hiệu đồng thời. Tập test có 34 ảnh `cercospora`, nên một ảnh đổi kết quả làm recall thay đổi gần 3 điểm phần trăm.

## Hạn chế

- Dự án giữ một nhãn chính trong khi 315 trong 1.685 dòng có mã nhãn hợp lệ bật từ hai cờ bệnh trở lên.
- Nguồn thiếu ID cây và phiên chụp. Mỗi ảnh là một nhóm trong CV. CV và test có thể hơi lạc quan nếu ảnh liên quan xuất hiện ở các tập khác nhau.
- `severity` được suy ra bằng phân vùng tự động, nên chỉ là tham chiếu yếu. Vùng nghi tổn thương nhạy với cách gom cụm và không có mặt nạ gán tay để kiểm chứng.
- Có 44 ảnh với khối đặc trưng vùng nghi tổn thương bằng 0 do vùng này dưới 1% diện tích lá. Theo lớp, số ảnh là `brown_spot` 1, `cercospora` 2, `healthy` 12, `leaf_miner` 8, `rust` 21.
