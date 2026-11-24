# Coffee Leaf ML

Dự án phân loại ảnh toàn lá cà phê theo năm nhãn `healthy`, `leaf_miner`, `rust`, `brown_spot` và `cercospora`. Tài liệu này tóm tắt cách đặt dữ liệu, kiểm thử và đọc kết quả của hai mô hình SVM và Decision Tree.

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
| --- | --- |
| `configs/default.yaml` | Nhãn, kích thước ảnh, phép chia, lưới tham số và thư mục kết quả |
| `data/raw/leaf/`, `data/processed/` | CSV và ảnh gốc, hai tệp đặc trưng train và test |
| `src/coffee_leaf/` | Đọc dữ liệu, tiền xử lý, trích đặc trưng, mô hình và đánh giá |
| `scripts/` | Các lệnh EDA, trích đặc trưng, huấn luyện, đánh giá và tính độ quan trọng |
| `notebooks/` | Notebook EDA và quy trình huấn luyện trên Colab |
| `models/artifacts/` | Pipeline đã fit, CSV kết quả và biểu đồ |
| `tests/` | Kiểm thử dữ liệu, phép xử lý và các bước của pipeline |

## Môi trường và kiểm thử

Môi trường bàn giao dùng Python `3.12.1`. Cột yêu cầu đọc từ [requirements.txt](requirements.txt). Cột phiên bản ghi môi trường đã kiểm tra.

| Gói | Yêu cầu | Phiên bản bàn giao |
| --- | --- | --- |
| NumPy | `>=1.26` | `2.5.5` |
| SciPy | `>=1.18.1` | `1.18.1` |
| scikit-learn | `>=1.4` | `1.9.1` |
| opencv-python | `>=4.9` | `5.1.0.95` |
| scikit-image | `>=0.22` | `0.26.0` |
| pandas | `>=2.1` | `3.1.1` |
| joblib | `>=1.3` | `1.6.0` |
| threadpoolctl | `>=3.8.0` | `3.8.0` |
| PyYAML | `>=6.0` | `6.0.4` |
| Streamlit | `>=1.31` | `1.67.0` |
| pytest | `>=8.0` | `9.2.0` |
| Pillow | `>=13.0.0` | `13.0.0` |
| Matplotlib, dùng cho biểu đồ | `>=3.8` | `3.11.2` |

Đặt CSV tại `data/raw/leaf/dataset.csv` và ảnh tại `data/raw/leaf/images/<id>.jpg`. Có thể giải nén tại đó hoặc đặt symlink `data/raw/leaf` trỏ đến dữ liệu đã giải nén. Kho mã không kèm ảnh.

Từ thư mục gốc dự án, dùng Python 3.12.1 để tạo môi trường và chạy test sau khi đặt dữ liệu.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pytest
```

## Kết quả chính từ Colab

1. Tập dữ liệu dùng 1.342 ảnh, gồm 1.006 train và 336 test. Mỗi ảnh có 403 đặc trưng.
2. SVM `balanced` được chọn bằng CV với macro-F1 trung bình 0,821.
3. Trên test, mô hình được chọn đạt accuracy 0,860 và macro-F1 0,828.
4. `cercospora` có F1 thấp nhất cả trong CV và trên test, lần lượt 0,5784 và 0,6076.
5. Với cây `decision_tree_none`, nhóm màu chiếm 0,717 tổng độ quan trọng đặc trưng.

## Tài liệu

| Tài liệu | Nội dung |
| --- | --- |
| [Đặt dữ liệu](docs/dataset.md) · [BRACOL](docs/BRACOL-dataset.md) | Nguồn, nhãn, phân bố và cách chia |
| [Báo cáo EDA](docs/eda-report.md) · [Notebook EDA](notebooks/01_eda.ipynb) | Số liệu nhãn, ảnh và đặc trưng train, cùng bảng và biểu đồ đã xuất |
| [Pipeline và đặc trưng](docs/pipeline.md) | Từ ảnh gốc đến vector 403 chiều |
| [Huấn luyện mô hình](docs/model-training.md) | Lưới tham số, CV và tệp kết quả |
| [Kết quả đánh giá](docs/evaluation-results.md) | Điểm train, CV, test và phân tích lỗi |
| [Hướng dẫn chạy](docs/running-guide.md) | Thứ tự chạy cục bộ và trên Colab |
| [Cấu trúc mã nguồn](docs/code-structure.md) | Cây tệp, import nội bộ và thứ tự đọc code |
| [Hướng dẫn từng module](docs/module-guide.md) | Vai trò, đầu vào, đầu ra và lỗi của từng hàm |
| [Luồng dữ liệu](docs/data-flow.md) | Shape, kiểu dữ liệu, cấu hình và schema tệp đầu ra |
| [Giải thích tách vùng](docs/segmentation-explained.md) | Từng bước khoanh lá, vùng nghi tổn thương và giới hạn |
| [Danh mục đặc trưng](docs/feature-catalog.md) | Cách tính và tên theo thứ tự của 403 cột |
| [Câu hỏi khi bảo vệ](docs/faq.md) | Các quyết định thiết kế và cách đọc kết quả |

## Phạm vi hiện tại

| Trạng thái | Thành phần |
| --- | --- |
| Đã có | Đọc và chia dữ liệu, tiền xử lý, tách vùng, trích đặc trưng, EDA, huấn luyện, đánh giá, độ quan trọng đặc trưng và biểu đồ |
| Chưa có triển khai | Suy luận ảnh mới (`src/coffee_leaf/inference/predictor.py`) và giao diện Streamlit (`app/streamlit_app.py`). Hai tệp này đang rỗng. |
