# Cách đặt dữ liệu BRACOL

Trang này dành cho người chuẩn bị dữ liệu đầu vào, hướng dẫn đặt CSV và ảnh BRACOL để script tìm đúng tệp.

## Nguồn và vị trí

Dự án dùng BRACOL của Krohling, Esgario và Ventura, Mendeley Data v1 (2019), giấy phép CC BY 4.0. [DOI 10.17632/yy2k5y8mxg.1](https://doi.org/10.17632/yy2k5y8mxg.1)

Đặt bộ dữ liệu sao cho có `data/raw/leaf/images/` và `data/raw/leaf/dataset.csv`. Có thể giải nén tại đó hoặc tạo symlink `data/raw/leaf` trỏ tới bộ dữ liệu đã có. Không đưa ảnh dữ liệu vào Git.

## Kết quả khảo sát

- `dataset.csv` có 1.747 dòng dữ liệu.
- Các cột: `id`, `predominant_stress`, `miner`, `rust`, `phoma`, `cercospora`, `severity`.
- `images/` có 1.401 file `.jpg`, tên theo mẫu `<id>.jpg`.
- Có 346 dòng CSV không có file ảnh khớp theo `id` và tên `<id>.jpg`.
- Năm lớp trong cấu hình là `brown_spot`, `cercospora`, `healthy`, `leaf_miner`, `rust`.

Dự án giữ 1.342 ảnh có nhãn hợp lệ và chia phân tầng thành 1.006 ảnh train, 336 ảnh test. Quy tắc nhãn, phân bố từng lớp và giới hạn metadata được ghi trong [Dữ liệu BRACOL và quy tắc xử lý](BRACOL-dataset.md).
