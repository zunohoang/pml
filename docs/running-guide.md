# Hướng dẫn chạy

Tài liệu này hướng dẫn chuẩn bị dữ liệu cục bộ, chạy huấn luyện trên Colab và dùng các pipeline đã lưu để xuất báo cáo.

## Cục bộ

1. Từ thư mục gốc dự án, tạo môi trường bằng Python `3.12.1` và cài package.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e .
   ```

2. Đặt `dataset.csv` và thư mục `images/` dưới `data/raw/leaf/`. Mỗi ảnh có tên `<id>.jpg`. Có thể dùng symlink tới thư mục dữ liệu đã giải nén. [Chi tiết dữ liệu](dataset.md)
3. Trích đặc trưng và chạy kiểm thử từ thư mục gốc.

   ```bash
   python scripts/extract_features.py
   python -m pytest
   ```

Script trích đặc trưng ghi `data/processed/features_train.npz` và `data/processed/features_test.npz`. Đặt dữ liệu gốc trước khi chạy toàn bộ test vì một số test kiểm tra trực tiếp dữ liệu và ảnh. [Môi trường bàn giao](../README.md#môi-trường-và-kiểm-thử)

## Khảo sát EDA

Sau khi trích đặc trưng, từ thư mục gốc chạy `python scripts/run_eda.py` để ghi CSV, PNG và JSON vào `docs/eda/`. Chương trình chỉ mở ảnh theo ID trong archive train và suy ra số test bằng phép trừ. [Báo cáo EDA](eda-report.md)

Để chạy [notebook EDA](../notebooks/01_eda.ipynb), mở notebook bằng môi trường có hỗ trợ Jupyter, chọn kernel Python dùng trình thông dịch `.venv/bin/python`, rồi chạy các ô theo thứ tự. Ô đầu tự tìm thư mục gốc từ thư mục làm việc hiện tại.

## Trên Colab

Chép vào thư mục dự án trên Google Drive các đường dẫn `scripts/`, `src/`, `configs/`, `requirements.txt`, `pyproject.toml`, `notebooks/02_train_colab.ipynb` và hai tệp `data/processed/features_train.npz`, `data/processed/features_test.npz`. Khi code đổi, cập nhật trọn `scripts/`, `src/` và notebook để các bản cùng phiên bản. Cập nhật cấu hình và tệp cài đặt nếu chúng thay đổi. Huấn luyện và đánh giá đọc các tệp đặc trưng, nên các bước này không cần chép ảnh gốc lên Drive.

| Mục notebook | Thao tác |
| --- | --- |
| 1. Kết nối Google Drive | Mount Drive và cấp quyền theo lời nhắc của Colab. |
| 2. Vào thư mục dự án | Đặt `PROJECT_DIR` tới thư mục trên Drive và kiểm tra notebook thấy tệp đặc trưng train. |
| 3. Cài đặt | Chạy ô ghim scikit-learn `1.9.1` và cài package editable từ gốc dự án. |
| 4. Huấn luyện | Chạy `scripts/train.py` để chọn ứng viên bằng CV và lưu đủ bốn pipeline. |
| 5–6. Xem kết quả và biểu đồ | Đọc CSV và vẽ biểu đồ CV từ `models/artifacts/`. |
| 7. Đánh giá test | Sau khi chốt ứng viên, chạy `scripts/evaluate.py` và đọc báo cáo test. |
| 8. Độ quan trọng đặc trưng | Chạy `scripts/feature_importance.py`, đọc bảng và vẽ biểu đồ của cây được chọn theo CV. |

CSV, JSON và pipeline được ghi vào `models/artifacts/` bên trong dự án trên Drive. Các ô biểu đồ ghi PNG vào `models/artifacts/figures/`. [Notebook Colab](../notebooks/02_train_colab.ipynb) · [Danh sách tệp kết quả](model-training.md)

## Chạy riêng hai báo cáo

Chạy huấn luyện trước, hoặc chép kết quả huấn luyện từ Drive về `models/artifacts/` và dùng cùng phiên bản scikit-learn với môi trường đã lưu pipeline. Thư mục `candidates/` cùng các tệp CV phải có trước khi chạy hai lệnh sau. Hai báo cáo có thể chạy độc lập sau huấn luyện.

| Lệnh từ thư mục gốc | Đầu vào cần có | Đầu ra |
| --- | --- | --- |
| `python scripts/evaluate.py` | `data/processed/features_test.npz`, `cv_results.csv`, `best_model_info.json` và đủ bốn pipeline trong `candidates/` | `test_results.csv`, `test_confusion_matrix.csv`, `test_per_class.csv` |
| `python scripts/feature_importance.py` | Metadata trong `data/processed/features_train.npz`, `cv_results.csv` và pipeline cây được chọn trong `candidates/` | `tree_feature_importance.csv` và ba bảng tổng hợp theo nhóm, vùng, nhóm–vùng |
