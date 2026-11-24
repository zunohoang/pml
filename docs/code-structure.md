# Cấu trúc mã nguồn

Tài liệu này dành cho người đọc biết Python và học máy, cần định vị các tệp trong dự án và biết nên đọc từ đâu. [Hướng dẫn từng module](module-guide.md) ghi hợp đồng của từng hàm.

## Cây tệp Python

```text
.
├── app/
│   └── streamlit_app.py        Điểm vào Streamlit, tệp rỗng.
├── scripts/
│   ├── extract_features.py     Chia ảnh, tính vector và ghi hai tệp NPZ.
│   ├── run_eda.py               Tổng hợp CSV nhãn và đặc trưng train, xuất báo cáo EDA.
│   ├── train.py                Tìm siêu tham số và lưu bốn ứng viên cùng báo cáo CV.
│   ├── evaluate.py             Chấm bốn ứng viên đã fit trên test.
│   └── feature_importance.py   Xuất độ quan trọng của cây chọn theo CV.
├── src/
│   └── coffee_leaf/
│       ├── __init__.py         Đánh dấu package.
│       ├── config.py           Đọc YAML.
│       ├── analysis/
│       │   ├── __init__.py     Đánh dấu package EDA.
│       │   ├── summaries.py    Tính thống kê, tương quan và PCA.
│       │   └── plots.py        Dựng tám hình EDA từ dữ liệu đã tính.
│       ├── data/
│       │   ├── __init__.py     Đánh dấu package dữ liệu.
│       │   ├── loader.py       Lọc mã, gán nhãn, tra ảnh theo ID.
│       │   └── split.py        Chia train/test phân tầng.
│       ├── preprocessing/
│       │   ├── __init__.py     Đánh dấu package xử lý ảnh.
│       │   ├── image.py        Sửa hướng, thu nhỏ, tìm lóa và chỉnh màu.
│       │   └── segmentation.py Tách lá và vùng nghi tổn thương.
│       ├── features/
│       │   ├── __init__.py     Đánh dấu package đặc trưng.
│       │   ├── channels.py     Đổi không gian màu và lượng tử hóa.
│       │   ├── color.py        Tính histogram và thống kê màu.
│       │   ├── extractor.py    Ghép vector và danh sách tên theo cùng thứ tự.
│       │   ├── feature_names.py Phân nhóm và vùng từ tên đặc trưng.
│       │   ├── gradient.py     Tính histogram độ biến đổi màu theo không gian.
│       │   ├── masks.py        Kiểm tra, kết hợp và co mặt nạ.
│       │   ├── shape.py        Đo hình học của các vùng nghi tổn thương.
│       │   └── texture.py      Tính LBP và GLCM.
│       ├── models/
│       │   ├── __init__.py     Đánh dấu package mô hình.
│       │   ├── builders.py     Dựng Pipeline có scaler và bộ phân loại.
│       │   ├── persistence.py  Ghi và nạp đối tượng joblib.
│       │   └── search.py        Dựng lưới, chia fold, tìm tham số và dự đoán CV.
│       ├── evaluation/
│       │   ├── __init__.py     Đánh dấu package đánh giá.
│       │   ├── feature_importance.py Chọn cây và tổng hợp độ quan trọng.
│       │   ├── metrics.py       Tính điểm, lập bảng và xếp hạng ứng viên.
│       │   └── plots.py         Vẽ Figure từ bảng đã đọc.
│       └── inference/
│           ├── __init__.py     Đánh dấu package suy luận.
│           └── predictor.py    Điểm chứa mã suy luận, tệp rỗng.
└── tests/
    ├── test_data.py               Kiểm tra lọc ảnh và chia tập lặp lại được.
    ├── test_eda.py                Kiểm tra thống kê và hình EDA bằng dữ liệu tổng hợp.
    ├── test_evaluation.py         Kiểm tra đầu vào test, điểm, báo cáo và cờ chọn CV.
    ├── test_extract_features.py   Kiểm tra archive, thứ tự mẫu và xử lý lỗi ảnh.
    ├── test_feature_importance.py Kiểm tra chọn cây, xếp hạng và bốn CSV.
    ├── test_features.py           Kiểm tra vector 403 số, tên cột, vùng và không sửa đầu vào.
    ├── test_features_basic.py     Kiểm tra kênh màu, mặt nạ, hình dạng và gradient.
    ├── test_metrics.py            Đối chiếu điểm và ma trận với ví dụ tính tay.
    ├── test_models.py             Kiểm tra pipeline, CV, chọn ứng viên và lưu mô hình.
    ├── test_plots.py              Kiểm tra bốn Figure và khả năng lưu PNG.
    ├── test_predictor.py          Tệp rỗng.
    ├── test_preprocessing.py      Kiểm tra dạng ảnh, mặt nạ và kết quả lặp lại.
    └── test_texture.py            Kiểm tra LBP, GLCM và cặp pixel hợp lệ.
```

## Import nội bộ

Mũi tên đi từ nơi import đến module được import. `features`, `models` và `evaluation` viết tắt ba package con của `coffee_leaf` trong sơ đồ.

```text
scripts/extract_features.py  → config, data.loader, data.split
                             → features.extractor, features.feature_names
scripts/run_eda.py           → config, data.loader, features.extractor
                             → preprocessing.image, preprocessing.segmentation, analysis.summaries, analysis.plots
scripts/train.py             → config, models.builders, models.search, models.persistence
                             → evaluation.metrics
scripts/evaluate.py          → scripts/train.py, config, models.persistence, evaluation.metrics
scripts/feature_importance.py → config, models.persistence, evaluation.feature_importance
notebooks/02_train_colab.ipynb → evaluation.plots

features.extractor → preprocessing.image, preprocessing.segmentation
                   → features.channels, features.color, features.gradient,
                     features.masks, features.shape, features.texture
features.color    → features.channels, features.masks
features.gradient → features.channels, features.color, features.masks
features.texture  → features.channels, features.color, features.masks
features.shape    → features.masks
models.builders, models.search → config
```

Ví dụ, script trích đặc trưng giao việc tạo mặt nạ cho extractor. Extractor gọi tách vùng rồi truyền mặt nạ vào các hàm đo màu và kết cấu. [Luồng dữ liệu theo từng bước](data-flow.md)

## Quy ước đọc và viết code

| Quy ước | Cách đọc |
| --- | --- |
| Không dùng comment và docstring theo yêu cầu của đề | Đọc ý nghĩa qua tên hàm, tên biến đầy đủ và bảng hợp đồng hàm. |
| Một hàm làm một việc | Đọc `main` như một chuỗi bước, rồi mở hàm phụ cần xem chi tiết. |
| Hằng số viết hoa ở đầu tệp | Xem kích thước, ngưỡng, nhóm cột và tên kết quả trước khi đọc thuật toán. |
| Kiểm tra đầu vào bằng `ValueError` | Ví dụ, mã nhãn không nhận diện được, vector sai shape hoặc có số không hữu hạn. Lỗi I/O theo loại ngoại lệ của thao tác. |
| Seed cố định | Phép chia tập và CV lấy seed 42 từ YAML. K-means chọn seed 42 từ hằng số trong module tách vùng. Khi đầu vào và môi trường giữ nguyên, các bước có seed có thể lặp lại. |
| Tính đặc trưng không sửa mảng ảnh và mặt nạ đầu vào | Các bước xử lý tạo đầu ra riêng. Loader có hai hàm gán cột tại chỗ trên DataFrame đã lọc, còn hàm vẽ cập nhật Axes. |

## Thứ tự đọc

1. Đọc `configs/default.yaml` và `scripts/extract_features.py` để thấy dữ liệu đầu vào, tập giữ lại và hai tệp đầu ra.
2. Đọc `data/loader.py` và `data/split.py` trong package để hiểu gán nhãn và chia ảnh.
3. Đọc `features/extractor.py` trước, rồi lần theo `preprocessing/image.py`, `preprocessing/segmentation.py` và bốn nhóm hàm tính đặc trưng. [Tách vùng](segmentation-explained.md) · [Danh mục đặc trưng](feature-catalog.md)
4. Đọc `scripts/train.py` cùng `models/builders.py`, `models/search.py` và `evaluation/metrics.py` để theo dõi scaler, tìm lưới và chọn ứng viên.
5. Đọc `scripts/evaluate.py`, `scripts/feature_importance.py` và các test cùng nhóm để hiểu kết quả và điều kiện lỗi.
