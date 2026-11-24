# Pipeline và đặc trưng

Tài liệu này mô tả cách dự án biến một ảnh toàn lá thành vector 403 đặc trưng. Vector kết hợp tín hiệu của toàn lá với tín hiệu tập trung ở vùng nghi tổn thương.

## Từ CSV đến hai tập đặc trưng

1. `load_config` đọc [cấu hình YAML](../configs/default.yaml). `build_image_table` giữ ảnh có mã `predominant_stress` từ `0` đến `4` và có file tương ứng với `id`. [Quy tắc nhãn](BRACOL-dataset.md)
2. `split_train_test` chia phân tầng theo lớp với tỷ lệ 75/25 và seed `42`. Có 1.006 ảnh train và 336 ảnh test.
3. `extract_features_from_path` xử lý từng ảnh. [Script trích đặc trưng](../scripts/extract_features.py) ghi `data/processed/features_train.npz` và `data/processed/features_test.npz`. Mỗi tệp có ma trận `features` kiểu `float64`, `label_ids`, `label_names`, `image_ids`, `feature_names`, `feature_groups` và `feature_regions`. Thứ tự tên, nhóm và vùng khớp thứ tự cột.

## Tiền xử lý và tách vùng

| Bước | Kỹ thuật và mục đích |
| --- | --- |
| Đọc và thu nhỏ | `load_resized_rgb` áp dụng hướng xoay từ EXIF, tức thông tin hướng ảnh lưu trong tệp, rồi chuyển sang RGB. Lanczos nội suy khi thu nhỏ và giữ tỷ lệ ảnh để giảm khối lượng tính toán. Ảnh 2.048 × 1.024 px thành 256 × 128 px khi giữ hướng ngang. |
| Đánh dấu vùng lóa | `find_glare_mask` dùng ảnh RGB đã thu nhỏ, trước khi cân bằng trắng. Trong HSV, pixel có độ sáng `V` ≥ 0,98 và độ bão hòa `S` ≤ 0,10 được đánh dấu lóa để loại khỏi vùng xét tổn thương. |
| Cân bằng trắng | `balance_white` dùng gray-world, đưa trung bình ba kênh RGB về gần một mức chung để giảm ám màu. Hệ số từng kênh được giới hạn trong khoảng 0,5 đến 2,0. |
| Tăng tương phản | `equalize_lightness` chạy CLAHE trên kênh sáng `L` của Lab, lưới 8 × 8 và giới hạn 2,0. CLAHE cân bằng histogram theo từng ô nhỏ, có giới hạn để tránh tăng tương phản quá mức. |
| Tách lá | `find_leaf_mask` dùng ảnh RGB đã thu nhỏ trước cân bằng trắng. Mặt nạ ứng viên dùng sắc độ HSV trong 8–170°, `S` > 0,15 và `V` > 0,06. Phép mở rồi đóng với kernel 3 × 3 dọn các vùng nhỏ và khe hở. Mã chọn vùng liên thông 4 hướng chứa nhiều pixel xanh theo chỉ số Excess Green nhất, hoặc vùng lớn nhất nếu không có pixel mốc. Sau khi lấp lỗ, mặt nạ dưới 40 pixel bị báo lỗi. |
| Khoanh vùng nghi tổn thương | `find_lesion_mask` xét pixel trong lá và ngoài vùng lóa trên ảnh đã chỉnh màu. K-means chia hai kênh màu `a`, `b` của Lab thành 3 cụm. Cụm xa nhất so với cụm đông nhất được giữ nếu khoảng cách tâm cụm từ 8 trở lên. Các vùng liên thông dưới 3 pixel bị bỏ. |

Mặt nạ lá giới hạn đặc trưng vào lá thay vì nền. Vùng nghi tổn thương giúp tập trung vào màu hoặc kết cấu khác phần lá chủ đạo. Nếu vùng này nhỏ hơn 1% diện tích lá, pipeline đặt 135 đặc trưng tương ứng bằng 0 và vẫn tính 268 đặc trưng toàn lá. [Xử lý ảnh](../src/coffee_leaf/preprocessing/image.py) · [Tách vùng](../src/coffee_leaf/preprocessing/segmentation.py) · [Ghép vector](../src/coffee_leaf/features/extractor.py)

## Vector 403 đặc trưng

| Nhóm | Vùng nghi tổn thương (`lesion`) | Toàn lá (`leaf`) | Tổng |
| --- | ---: | ---: | ---: |
| Màu (`color`) | 102 | 100 | 202 |
| Kết cấu (`texture`) | 26 | 144 | 170 |
| Hình dạng (`shape`) | 7 | 0 | 7 |
| Gradient (`gradient`) | 0 | 24 | 24 |
| **Tổng** | **135** | **268** | **403** |

| Khối theo thứ tự vector | Cách tính |
| --- | --- |
| Màu vùng nghi tổn thương (102) | 96 ô histogram HSV và Lab, 16 ô mỗi kênh trên ảnh đã chỉnh màu. Thêm trung bình và độ lệch chuẩn của 3 kênh Lab. |
| Kết cấu vùng nghi tổn thương (26) | 6 chỉ số GLCM ở khoảng cách 1 px và 20 ô histogram LBP ở bán kính 1 và 2 px trên ảnh xám đã chỉnh màu. |
| Hình dạng vùng nghi tổn thương (7) | Số vùng, diện tích trung bình và lớn nhất so với diện tích lá, độ lệch tâm và độ đặc trung bình, khoảng cách trung bình tới mép lá đã chuẩn hóa và tỷ lệ diện tích vùng nghi tổn thương. |
| Màu toàn lá (100) | Histogram hai chiều 10 × 10 của `a`, `b` trong Lab từ ảnh đã chỉnh màu, chuẩn hóa rồi lấy căn bậc hai. |
| LBP toàn lá (90) | 3 kênh Lab × 3 bán kính (1, 2, 3 px) × 10 ô histogram trên ảnh trước chỉnh màu. Histogram được chuẩn hóa rồi lấy căn bậc hai. |
| Gradient toàn lá (24) | 8 ô histogram độ lớn gradient cho mỗi kênh Lab trên ảnh trước chỉnh màu. Mặt nạ được co bằng cửa sổ 3 × 3 để giảm ảnh hưởng biên. Histogram được chuẩn hóa rồi lấy căn bậc hai. |
| GLCM toàn lá (54) | 3 kênh Lab × 3 khoảng cách (1, 2, 4 px) × 6 chỉ số trên ảnh trước chỉnh màu. |

- **LBP** so sánh pixel trung tâm với 8 điểm lân cận để mã hóa kiểu sáng tối cục bộ. Tâm lấy mẫu nằm trong mặt nạ đã co theo bán kính.
- **GLCM** đếm các cặp mức xám xuất hiện ở một hướng và khoảng cách nhất định. Code lượng tử hóa thành 16 mức, chỉ lấy cặp nằm trong cùng mặt nạ và lấy trung bình 6 chỉ số qua 4 hướng.
- **Gradient** đo mức thay đổi của mỗi kênh giữa các pixel lân cận theo hai chiều. Histogram độ lớn mô tả mức biến đổi bên trong lá.

Tên đặc trưng theo dạng `vùng_nhóm_chi_tiết`. Thành phần nhóm trong tên là họ kỹ thuật, ví dụ `lab`, `ab`, `lbp`, `glcm` hoặc `grad`. `describe_feature` ánh xạ các họ này sang 4 nhóm báo cáo. Ví dụ `lesion_lab_std_l` là độ lệch chuẩn kênh sáng Lab của vùng nghi tổn thương, còn `leaf_ab_hist_a05_b07` là một ô histogram màu toàn lá. [Quy tắc tên](../src/coffee_leaf/features/feature_names.py)

Trong lần chạy được ghi nhận, 62 cột histogram luôn bằng 0 trên tập train vì không có pixel rơi vào các ô đó. Pipeline giữ đủ 403 cột để duy trì cùng cấu trúc đặc trưng giữa train, test và mô hình.
