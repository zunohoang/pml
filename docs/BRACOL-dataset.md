# Phân bố và quy tắc xử lý BRACOL

Trang này dành cho người đọc kết quả, trình bày phân bố nhãn, phép chia tập và giới hạn của dữ liệu. Dự án dùng phần ảnh toàn lá của BRACOL v1, phát hành năm 2019 theo giấy phép CC BY 4.0. [Nguồn BRACOL](https://doi.org/10.17632/yy2k5y8mxg.1)

Dùng `predominant_stress` để gán một nhãn cho mỗi ảnh toàn lá. Đối chiếu mỗi `id` với `data/raw/leaf/images/<id>.jpg`. Bốn cột bệnh và `severity` không tham gia gán nhãn. Mã `5` nằm ngoài bảng ánh xạ và bị bỏ qua. [Bảng ánh xạ](../configs/default.yaml#L17) · [Điều kiện giữ ảnh](../src/coffee_leaf/data/loader.py#L39)

| `predominant_stress` | Nhãn | Dòng CSV | Có ảnh |
| ---: | --- | ---: | ---: |
| 0 | `healthy` | 272 | 142 |
| 1 | `leaf_miner` | 387 | 253 |
| 2 | `rust` | 531 | 465 |
| 3 | `brown_spot` | 348 | 346 |
| 4 | `cercospora` | 147 | 136 |
| 5 | Bỏ qua | 62 | 59 |
| **Tổng** | | **1.747** | **1.401** |

Có 1.342 ảnh toàn lá có mã `0`–`4` và có tệp ảnh tương ứng.

## Bảng chéo trên toàn bộ CSV

Mỗi tổ hợp gồm bốn chữ số theo thứ tự `miner`, `rust`, `phoma`, `cercospora`. Các cột `0`–`5` là giá trị `predominant_stress`. Bảng liệt kê các tổ hợp xuất hiện trong `data/raw/leaf/dataset.csv`. [Cách đặt dữ liệu](dataset.md)

| Tổ hợp | 0 | 1 | 2 | 3 | 4 | 5 | Tổng |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0000` | 272 | 0 | 0 | 0 | 0 | 2 | 274 |
| `0001` | 0 | 0 | 0 | 0 | 87 | 1 | 88 |
| `0010` | 0 | 0 | 0 | 341 | 0 | 1 | 342 |
| `0011` | 0 | 0 | 0 | 4 | 3 | 1 | 8 |
| `0100` | 0 | 0 | 348 | 0 | 0 | 1 | 349 |
| `0101` | 0 | 0 | 112 | 0 | 52 | 30 | 194 |
| `0110` | 0 | 0 | 1 | 2 | 0 | 0 | 3 |
| `1000` | 0 | 322 | 0 | 0 | 0 | 0 | 322 |
| `1001` | 0 | 19 | 0 | 0 | 5 | 3 | 27 |
| `1010` | 0 | 1 | 0 | 1 | 0 | 0 | 2 |
| `1100` | 0 | 45 | 68 | 0 | 0 | 20 | 133 |
| `1101` | 0 | 0 | 2 | 0 | 0 | 3 | 5 |
| **Tổng** | **272** | **387** | **531** | **348** | **147** | **62** | **1.747** |

Ý nghĩa của mã `5` chưa rõ. Mã này xuất hiện trên 2 dòng toàn `0`, 3 dòng có đúng một cột bệnh bằng `1` và 57 dòng có từ hai cột bệnh bằng `1`. Dự án loại mã `5` trong cả ba trường hợp.

## Xem mẫu ảnh

Đã xem 50 ảnh toàn lá, gồm 6 ảnh mỗi lớp, 10 ảnh mã `5` phủ cả 9 tổ hợp cờ của mã này và 5 cặp ID liền nhau. Ảnh có nền sáng với độ sáng và bóng đổ thay đổi. Mẫu `leaf_miner` có mảng nâu, còn cả mẫu `rust` và `cercospora` đều có thể có vết vàng hoặc nâu. Trong hai ảnh mã `5` có tổ hợp `0000`, ID `639` có một đốm tròn nhỏ, còn ID `671` không có đốm nổi bật ở ảnh xem mẫu.

## Chia train và test

Hàm `split_train_test` dùng `train_test_split` của scikit-learn, phân tầng theo nhãn, tỷ lệ 75/25 và seed `42`. Mỗi mẫu giữ ID gốc để nối với CSV và ảnh. [Mã chia tập](../src/coffee_leaf/data/split.py)

| Nhãn | Train | Test |
| --- | ---: | ---: |
| `brown_spot` | 259 | 87 |
| `cercospora` | 102 | 34 |
| `healthy` | 106 | 36 |
| `leaf_miner` | 190 | 63 |
| `rust` | 349 | 116 |
| **Tổng** | **1.006** | **336** |

Dùng 5 fold phân tầng theo nhóm ảnh với seed `42` trong 1.006 ảnh train để chọn cấu hình. `StandardScaler` và mô hình được fit trên phần train của từng fold. Tập 336 ảnh test dành cho báo cáo sau khi chốt mô hình. [Cấu hình dự án](../configs/default.yaml)

## Hạn chế

- Dự án giữ một nhãn chính. Trong 1.685 dòng mã `0`–`4`, 315 dòng có từ hai cờ bệnh bật. Riêng lớp `cercospora`, 52 trong 147 dòng còn bật cờ `rust`.
- Metadata không có ID cây hoặc phiên chụp. Mỗi ảnh là một nhóm trong CV, nên cả CV lẫn test có thể hơi lạc quan khi các ảnh liên quan lọt vào các tập khác nhau. Kiểm tra SHA-256 trên 1.342 JPG không thấy ảnh trùng byte. Cặp `630`–`631` có nền và bố cục tương tự, nhưng ID liền nhau không xác nhận cùng cây hay phiên chụp.
- `severity` là nhãn tham chiếu yếu vì được suy ra từ phân vùng tự động. Mặt nạ của pipeline chỉ đánh dấu vùng nghi tổn thương, nhạy với cách gom cụm và thiếu mặt nạ gán tay để đối chiếu.
- Có 44 ảnh mà vùng nghi tổn thương chiếm dưới 1% diện tích lá, nên khối 135 đặc trưng của vùng này bằng 0. Theo lớp, số ảnh là `brown_spot` 1, `cercospora` 2, `healthy` 12, `leaf_miner` 8 và `rust` 21.

## Tài liệu tham khảo

- Krohling, R. A., Esgario, G. J. M., và Ventura, J. A. (2019). *BRACOL*, Mendeley Data v1. [DOI 10.17632/yy2k5y8mxg.1](https://doi.org/10.17632/yy2k5y8mxg.1)
- Esgario, J. G. M., Krohling, R. A., và Ventura, J. A. (2020). *Deep learning for classification and severity estimation of coffee leaf biotic stress*. Computers and Electronics in Agriculture, 169, 105162. [Bài báo](https://www.sciencedirect.com/science/article/pii/S0168169919313225)
