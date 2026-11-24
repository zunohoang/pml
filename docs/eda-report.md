# Khảo sát dữ liệu và đặc trưng

Tài liệu này ghi kết quả EDA để đọc dữ liệu trước khi huấn luyện và trả lời câu hỏi khi bảo vệ. Thống kê từ ảnh và đặc trưng chỉ dùng 1.006 ảnh train. Phân bố nhãn, cờ bệnh và severity lấy từ CSV, còn số test được suy ra bằng phép trừ.

Bảng nguồn nằm trong `eda/tables/`, gồm [kiểm kê dữ liệu](eda/tables/dataset_overview.csv). Các số chính có trong [eda_summary.json](eda/eda_summary.json). Các mốc đối chiếu đều khớp, riêng số ảnh có khối vùng nghi tổn thương bằng 0 cần phân định phạm vi với số cũ.

## Dữ liệu và phân bố lớp

CSV có 1.747 dòng, 1.401 dòng có ảnh khớp, 1.685 dòng mang mã 0–4 và 1.342 ảnh dùng được. Cả 1.006 ảnh train có kích thước gốc 2.048 × 1.024 px, đo từ thông tin kích thước của tệp ảnh.

| Lớp | Tổng | Train | Test suy ra |
| --- | ---: | ---: | ---: |
| `healthy` | 142 | 106 | 36 |
| `leaf_miner` | 253 | 190 | 63 |
| `rust` | 465 | 349 | 116 |
| `brown_spot` | 346 | 259 | 87 |
| `cercospora` | 136 | 102 | 34 |
| **Tổng** | **1.342** | **1.006** | **336** |

Lớp lớn nhất `rust` gấp 3,42 lần lớp nhỏ nhất `cercospora`.

![Phân bố lớp theo tập](eda/figures/01_class_distribution.png)

## Cờ bệnh và severity

Có 372 dòng bật ít nhất hai cờ bệnh, trong đó 315 dòng mang mã chính 0–4. Riêng 52 trong 147 dòng mã `cercospora` còn bật cờ `rust`, nên một nhãn chính có thể tóm lược nhiều dấu hiệu đồng thời.

![Bảng chéo của mã chính và các cờ bệnh](eda/figures/02_multilabel.png)

| Lớp | Số dòng CSV | Thấp nhất | Trung vị | Cao nhất |
| --- | ---: | ---: | ---: | ---: |
| `healthy` | 272 | 0 | 0 | 0 |
| `leaf_miner` | 387 | 1 | 1 | 3 |
| `rust` | 531 | 1 | 1 | 4 |
| `brown_spot` | 348 | 1 | 1 | 4 |
| `cercospora` | 147 | 1 | 1 | 4 |

Đây là mức ghi trong CSV. Trung vị của cả bốn lớp bệnh là 1.

![Phân bố severity theo lớp](eda/figures/03_severity_by_class.png)

## Ảnh mẫu và tách vùng

Chọn 3 ảnh train mỗi lớp bằng seed 42. Hai ảnh đầu mỗi lớp được vẽ thêm mặt nạ lá màu xanh và vùng nghi tổn thương màu cam. Các hình giúp kiểm tra trực quan xem vùng được chọn có nằm trong lá hay bám theo mép, gân lá và các mảng đổi màu.

![Mười lăm ảnh mẫu từ tập train](eda/figures/04_sample_images.png)

![Mười cặp ảnh gốc và ảnh phủ mặt nạ](eda/figures/05_segmentation_overlays.png)

## Tỷ lệ vùng nghi tổn thương

Tỷ lệ lấy từ `lesion_shape_area_ratio`. Các giá trị 0 trong archive được giữ khi tính tứ phân vị và trung vị. Cột khối bằng 0 yêu cầu cả 135 đặc trưng vùng nghi tổn thương cùng bằng 0.

| Lớp | Trung vị diện tích trên train | Khối bằng 0 trên train | Số cũ, chưa rõ phạm vi |
| --- | ---: | ---: | ---: |
| `healthy` | 4,22% | 9 | 12 |
| `leaf_miner` | 9,77% | 6 | 8 |
| `rust` | 5,85% | 19 | 21 |
| `brown_spot` | 6,79% | 1 | 1 |
| `cercospora` | 5,76% | 2 | 2 |
| **Tổng** | — | **37** | **44** |

Hai tổng **cần phân định phạm vi**. EDA chỉ đếm train. Các tài liệu ghi 44 được giữ nguyên.

![Tỷ lệ diện tích vùng nghi tổn thương theo lớp](eda/figures/06_lesion_ratio_by_class.png)

## Tương quan và PCA

Có 62 cột hằng, tất cả bằng 0. Trong 341 cột còn lại, 237 trên 57.970 cặp có Pearson |r| ≥ 0,9. Có 146 cột, tương đương 42,82%, xuất hiện trong ít nhất một cặp như vậy. Đây là một lý do thực nghiệm để không ưu tiên Naive Bayes. Tương quan gộp chưa trực tiếp kiểm tra giả định độc lập có điều kiện theo lớp.

![Tương quan Pearson của đặc trưng train theo nhóm và vùng](eda/figures/07_feature_correlation.png)

Sau `StandardScaler`, PC1 giải thích 19,72% phương sai và PC2 giải thích 11,56%, tổng 31,28%. Cần 63 thành phần để đạt 90%. `cercospora` chồng lấn với các lớp khác, gồm `rust` và `leaf_miner`, trên phép chiếu hai chiều. Hình chỉ giữ một phần biến thiên của vector.

![PCA hai chiều của đặc trưng train đã chuẩn hóa](eda/figures/08_pca_scatter.png)

## Câu hỏi khi bảo vệ

| Câu hỏi | Số liệu trả lời |
| --- | --- |
| Mất cân bằng lớp bao nhiêu? | 465 ảnh `rust` và 136 ảnh `cercospora`, gấp 3,42 lần. |
| Một nhãn chính bỏ qua bao nhiêu dòng có nhiều cờ bệnh? | 315 trong 1.685 dòng có mã hợp lệ. |
| Bao nhiêu cột train có phương sai bằng 0? | 62 trong 403 cột. |
| Hai trục PCA giữ được bao nhiêu thông tin biến thiên? | 31,28% phương sai. |
| Có bao nhiêu ảnh train không có khối vùng nghi tổn thương hữu dụng? | 37 trong 1.006 ảnh có cả khối bằng 0. |

## Giới hạn

Dự án giữ một nhãn chính dù lá có thể bật nhiều cờ bệnh. Severity là tham chiếu yếu. Mặt nạ chỉ là vùng nghi tổn thương theo màu và thiếu mặt nạ gán tay để kiểm chứng. Dữ liệu không có ID cây hoặc phiên chụp, nên chưa xác nhận mức độ độc lập giữa các ảnh. [Giới hạn dữ liệu](BRACOL-dataset.md#hạn-chế)
