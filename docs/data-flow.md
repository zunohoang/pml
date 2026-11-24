# Luồng dữ liệu và định dạng tệp

Tài liệu này dành cho người đọc code hoặc dùng lại kết quả của pipeline, cần biết mỗi bước nhận và trả dữ liệu gì. [Pipeline tổng quan](pipeline.md) giải thích mục đích của các phép xử lý.

Ký hiệu `H` và `W` là chiều cao và chiều rộng sau thu nhỏ. Ví dụ, ảnh ngang 256 × 128 px có shape RGB `(128, 256, 3)`. Mặt nạ của ảnh đó có shape `(128, 256)`, với một giá trị bool cho mỗi pixel.

## Các bước chính

| Bước | Đầu vào | Hàm xử lý | Đầu ra và kiểu dữ liệu |
| --- | --- | --- | --- |
| Đọc dữ liệu | `data/raw/leaf/dataset.csv`, 1.747 dòng trong bản dữ liệu dùng | `read_dataset_table` | DataFrame có `id`, `predominant_stress`, các cờ `miner`, `rust`, `phoma`, `cercospora` và `severity`. pandas suy kiểu khi đọc. |
| Lọc và gán nhãn | DataFrame dữ liệu và đường dẫn `data/raw/leaf/images/` | `build_image_table` | DataFrame `(1342, 3)` với `id` nguyên, `label` dạng chuỗi và `image_path` là đối tượng Path. |
| Chia tập | Bảng ảnh đã lọc và cấu hình | `split_train_test`, `load_samples` | Hai bảng `(1006, 3)` và `(336, 3)`. Khi trích đặc trưng, bảng đầy đủ có thêm cột bool `is_train`. |
| Đọc và chỉnh ảnh | Một JPG và cạnh tối đa | `load_resized_rgb`, `preprocess_rgb` | Hai ảnh RGB `uint8` đều có shape `(H, W, 3)`, trước và sau chỉnh màu. |
| Khoanh vùng | Ảnh RGB trước và sau chỉnh | `find_glare_mask`, `find_leaf_mask`, `find_lesion_mask`, `build_lesion_roi_mask` | Mặt nạ bool `(H, W)`. Hai kênh Lab của pixel xét dùng cho K-means có shape `(N, 2)` và là số thực. |
| Trích từng ảnh | Hai ảnh RGB, mặt nạ lá, vùng nghi tổn thương và lóa | `extract_features`, qua `extract_features_from_path` | Vector `float64` hữu hạn `(403,)`, gồm 135 số của vùng nghi tổn thương và 268 số toàn lá. |
| Ghi tập đặc trưng | Mỗi ảnh một vector, bảng nhãn và cờ chia | `build_feature_archive`, `split_feature_arrays`, `save_split_feature_files` | Hai NPZ nén, ma trận `float64` lần lượt `(1006, 403)` và `(336, 403)`. |
| CV và fit | Đặc trưng, ID nhãn và ID ảnh từ NPZ train | `run_grid_search`, `select_best_candidate`, `build_cv_predictions` | Bốn Pipeline đã fit, một ứng viên chốt và vector dự đoán ngoài fold `(1006,)` kiểu nguyên. |
| Chấm test | NPZ test, điểm CV, metadata mô hình và bốn pipeline đã fit | `load_test_data`, `score_candidates`, `write_test_diagnostics` | Bốn kết quả với nhãn dự đoán `(336,)` kiểu nguyên và điểm số thực. Ma trận và bảng theo lớp chỉ lấy ứng viên chốt bằng CV. |
| Giải thích cây | Ba mảng metadata từ NPZ train, điểm CV và pipeline cây | `load_feature_metadata`, `load_best_tree_pipeline`, `compute_importance_tables` | Vector độ quan trọng dạng số thực `(403,)` và bốn DataFrame chi tiết hoặc tổng hợp. |

## Khóa trong hai tệp NPZ

`scripts/extract_features.py` ghi `data/processed/features_train.npz` và `data/processed/features_test.npz` theo cùng schema. Số hàng `N` là 1.006 cho train hoặc 336 cho test. Bảng dưới lấy từ logic ghi archive.

| Khóa | Shape | Kiểu | Ý nghĩa |
| --- | --- | --- | --- |
| `features` | `(N, 403)` | `float64` | Mỗi hàng là vector của một ảnh. |
| `label_ids` | `(N,)` | `int64` | Mã nhãn, cũng là chỉ số trong `label_names`. |
| `image_ids` | `(N,)` | Số nguyên suy từ ID CSV | ID ảnh cùng thứ tự hàng với `features`. |
| `label_names` | `(5,)` | Unicode NumPy | Thứ tự `healthy`, `leaf_miner`, `rust`, `brown_spot`, `cercospora` theo mã tăng dần. |
| `feature_names` | `(403,)` | Unicode NumPy | Tên từng cột, theo thứ tự ghép vector. |
| `feature_groups` | `(403,)` | Unicode NumPy | `color`, `texture`, `shape` hoặc `gradient` của từng cột. |
| `feature_regions` | `(403,)` | Unicode NumPy | `lesion` hoặc `leaf` của từng cột. |

`is_train` chỉ dùng trong bundle trước khi tách mảng. Hai archive đầu ra bỏ khóa này và giữ cùng metadata cột.

## Cấu hình YAML hiện tại

| Khóa | Ý nghĩa | Giá trị |
| --- | --- | --- |
| `seed` | Seed chia tập, CV và dựng bộ phân loại | 42 |
| `data_root` | Thư mục CSV và ảnh | `data/raw/leaf` |
| `image_max_size` | Cạnh dài tối đa của ảnh RGB | 256 px |
| `classes` | Danh sách lớp trong cấu hình | `brown_spot`, `cercospora`, `healthy`, `leaf_miner`, `rust` |
| `folds` | Số fold CV | 5 |
| `svm` → `C` | Lưới mức phạt lỗi SVM | 0,1 / 1 / 10 / 100 |
| `svm` → `gamma` | Lưới hệ số kernel RBF | `scale` / 0,003 / 0,001 / 0,0003 / 0,0001 |
| `decision_tree` → `max_depth` | Lưới chiều sâu cây | `null` / 5 / 10 / 20 |
| `decision_tree` → `min_samples_leaf` | Lưới số mẫu tối thiểu tại nút lá | 1 / 5 / 10 |
| `label_by_code` | Nhãn theo mã, quyết định thứ tự trong NPZ | 0 `healthy`, 1 `leaf_miner`, 2 `rust`, 3 `brown_spot`, 4 `cercospora` |
| `ignored_codes` | Mã bị loại khi đọc dữ liệu | Danh sách chứa 5 |
| `test_size` | Tỷ lệ giữ lại cho test | 0,25 |
| `model_dir` | Nơi ghi pipeline và báo cáo | `models/artifacts` |

`classes` và `label_by_code` cùng chứa năm nhãn, theo hai thứ tự khác nhau. Mã trích đặc trưng dùng thứ tự mã của `label_by_code` để tạo `label_names`. [Cấu hình thực tế](../configs/default.yaml)

## Đầu ra trong `models/artifacts/`

Đường dẫn trong bảng tính từ `models/artifacts/`. [Mục đích từng báo cáo](model-training.md)

| Tệp | Nơi ghi | Schema hoặc kiểu dữ liệu |
| --- | --- | --- |
| `best_model.joblib` | `scripts/train.py` | Pipeline đã fit của ứng viên CV được chọn. |
| `best_model_info.json` | `scripts/train.py` | Object với `model_name`, `class_weight`, `best_params`, `best_cv_macro_f1`, `cv_std`, `train_macro_f1`, `train_accuracy`, `cv_accuracy`, `feature_count`, `train_rows`, `class_names`, `sklearn_version`. |
| `candidates/svm_none.joblib`, `candidates/svm_balanced.joblib`, `candidates/decision_tree_none.joblib`, `candidates/decision_tree_balanced.joblib` | `scripts/train.py` | Bốn Pipeline đã fit trên train. |
| `cv_results.csv` | `scripts/train.py` | 4 dòng. `model_name`, `class_weight` nhận diện ứng viên. `best_params` là chuỗi JSON. Điểm và độ lệch chuẩn nằm trong `train_accuracy`, `cv_accuracy`, `cv_accuracy_std`, `train_macro_f1`, `best_cv_macro_f1`, `cv_std`. `fit_seconds` là thời gian fit và tìm lưới. |
| `test_results.csv` | `scripts/evaluate.py` | 4 dòng. `model_name`, `class_weight`, `selected_by_cv`, `train_accuracy`, `cv_accuracy`, `test_accuracy`, `train_macro_f1`, `cv_macro_f1`, `test_macro_f1`. Cờ xác định ứng viên đã chốt bằng CV. |
| `cv_confusion_matrix.csv`, `test_confusion_matrix.csv` | Lần lượt `scripts/train.py`, `scripts/evaluate.py` | 5 hàng số đếm, cột đầu `actual_class`, 5 cột còn lại là tên lớp dự đoán. |
| `cv_per_class.csv`, `test_per_class.csv` | Lần lượt `scripts/train.py`, `scripts/evaluate.py` | 5 dòng, theo thứ tự lớp. `class_name`, `precision`, `recall`, `f1`, `support`. Ba điểm được ghi 4 chữ số thập phân, support là số mẫu. |
| `tree_feature_importance.csv` | `scripts/feature_importance.py` | 403 dòng, xếp giảm dần. `feature_name`, `group`, `region`, `importance`. |
| `tree_importance_by_group.csv` | `scripts/feature_importance.py` | 4 dòng. `group`, `importance`, `feature_count`. |
| `tree_importance_by_region.csv` | `scripts/feature_importance.py` | 2 dòng. `region`, `importance`, `feature_count`. |
| `tree_importance_by_group_region.csv` | `scripts/feature_importance.py` | 6 dòng. `group`, `region`, `importance`, `feature_count`. |
| `figures/train_vs_validation.png`, `figures/confusion_matrix.png`, `figures/per_class_scores.png`, `figures/tree_feature_importance.png` | `notebooks/02_train_colab.ipynb` | Bốn ảnh PNG vẽ từ các CSV. |

## Chặn rò rỉ dữ liệu

```text
dataset.csv + ảnh JPG
    → lọc ID và gán một nhãn
    → chia phân tầng train/test trước huấn luyện
        ├─ train → vector từng ảnh → features_train.npz
        │   → CV 5 fold theo nhóm image_id
        │   → fit StandardScaler và bộ phân loại trong Pipeline trên train của từng fold
        │   → chọn tham số, chọn ứng viên theo CV, fit lại trên toàn bộ train
        │   → lưu candidates, ứng viên chốt, bảng CV
        │   → metadata train + cây đã fit → bảng độ quan trọng
        └─ test  → vector từng ảnh → features_test.npz
            → chỉ evaluate.py đọc để predict bằng pipeline đã fit
            → gắn cờ theo ứng viên đã chốt → ghi bảng test
```

Các bước tạo vector chỉ xử lý trong từng ảnh, ví dụ K-means tìm cụm màu bên trong một lá. Với scaler học từ nhiều mẫu, Pipeline fit scaler trên phần train của từng fold. Đánh giá test dùng scaler và mô hình đã lưu. Rủi ro do thiếu ID cây hoặc phiên chụp được trình bày trong [kết quả đánh giá](evaluation-results.md#hạn-chế).
