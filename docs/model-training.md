# Huấn luyện mô hình

Tài liệu này ghi cấu hình của SVM và Decision Tree, cách chọn ứng viên bằng cross-validation và ý nghĩa các tệp kết quả.

## Mô hình và lưới tham số

Mỗi thuật toán có hai ứng viên. `class_weight=None` dùng trọng số mặc định và ghi là `none` trong CSV. `class_weight="balanced"` tăng trọng số cho lớp ít mẫu theo tần suất lớp trong dữ liệu fit.

| Mô hình | Cấu hình | Lưới trong `configs/default.yaml` |
| --- | --- | --- |
| SVM (`SVC`) | Kernel RBF, seed 42 | `C` nhận 0,1 / 1 / 10 / 100. `gamma` nhận `scale` / 0,003 / 0,001 / 0,0003 / 0,0001. |
| `DecisionTreeClassifier` | Seed 42 | `max_depth` nhận `null` / 5 / 10 / 20. `min_samples_leaf` nhận 1 / 5 / 10. |

`StandardScaler` đứng trước bộ phân loại trong cả hai `Pipeline`. Với SVM, chuẩn hóa giúp tránh để cột có thang đo lớn chi phối khoảng cách. Đặt scaler trong pipeline khiến nó chỉ học từ phần train của mỗi fold. [Lưới tham số](../configs/default.yaml) · [Mã dựng pipeline](../src/coffee_leaf/models/builders.py)

Đơn vị tiến độ là lần fit của một cấu hình trên một fold. SVM giải bài toán tối ưu và cây chia nhánh theo tiêu chí phân loại, nên luồng này không có tham số epoch. Các chỉ số dùng nhãn từ `predict`. Dự án tắt `probability=True` để tránh bước hiệu chỉnh xác suất khi chấm điểm chỉ cần nhãn.

## Cross-validation và quy tắc chọn

1. `scripts/train.py` đọc `data/processed/features_train.npz` với 1.006 hàng và 403 cột đặc trưng.
2. `GridSearchCV` dùng `StratifiedGroupKFold` với 5 fold, xáo trộn và seed 42. Một `image_id` là một nhóm để cùng ID không nằm ở cả hai phía của một fold. Tính accuracy và macro-F1, tức trung bình F1 với trọng số bằng nhau cho các lớp.
3. Với mỗi ứng viên, grid search chọn lưới theo macro-F1 validation trung bình và fit lại pipeline trên toàn tập train. Khi chọn giữa bốn ứng viên, dùng macro-F1 CV cao nhất, rồi độ lệch chuẩn thấp hơn nếu hòa. Nếu vẫn hòa, giữ ứng viên xuất hiện trước theo thứ tự SVM `none`, SVM `balanced`, cây `none`, cây `balanced`.
4. Pipeline được chốt có thêm báo cáo gộp từ dự đoán ngoài fold. Mỗi ảnh được dự đoán khi nằm trong fold validation. Điểm gộp toàn tập có thể hơi khác trung bình điểm của 5 fold.

Tập test chỉ dùng ở bước báo cáo, sau khi chốt ứng viên. Phần giải thích cây chọn ứng viên trong hai cây bằng macro-F1 CV, rồi độ lệch chuẩn thấp hơn, rồi ưu tiên `none` khi vẫn hòa. [Grid search](../src/coffee_leaf/models/search.py) · [Chọn ứng viên](../src/coffee_leaf/evaluation/metrics.py) · [Chọn cây](../src/coffee_leaf/evaluation/feature_importance.py)

## Chạy trên Colab

Mở [notebook Colab](../notebooks/02_train_colab.ipynb), kết nối Drive, vào thư mục dự án và cài package editable. Notebook ghim scikit-learn `1.9.1`. Chạy mục 4 để huấn luyện, mục 5–6 để đọc và vẽ kết quả CV, mục 7 để chấm test và mục 8 để giải thích cây. [Các bước chuẩn bị và chạy](running-guide.md)

## Kết quả trong `models/artifacts/`

| Tệp | Nội dung |
| --- | --- |
| `best_model.joblib` | Pipeline đã fit của ứng viên được chọn bằng CV |
| `best_model_info.json` | Tên, trọng số lớp, tham số được chọn, điểm train và CV, số hàng, số đặc trưng, thứ tự lớp và phiên bản scikit-learn |
| `candidates/svm_none.joblib`, `candidates/svm_balanced.joblib` | Hai pipeline SVM đã fit để chấm test |
| `candidates/decision_tree_none.joblib`, `candidates/decision_tree_balanced.joblib` | Hai pipeline cây đã fit để chấm test và phân tích độ quan trọng |
| `cv_results.csv` | Bốn ứng viên với `model_name`, `class_weight`, `best_params`, `train_accuracy`, `cv_accuracy`, `cv_accuracy_std`, `train_macro_f1`, `best_cv_macro_f1`, `cv_std`, `fit_seconds` |
| `cv_confusion_matrix.csv`, `test_confusion_matrix.csv` | Ma trận của ứng viên được chọn, hàng là lớp thật trong `actual_class` và cột là lớp dự đoán |
| `cv_per_class.csv`, `test_per_class.csv` | `class_name`, `precision`, `recall`, `f1`, `support` của ứng viên được chọn |
| `test_results.csv` | `model_name`, `class_weight`, `selected_by_cv`, `train_accuracy`, `cv_accuracy`, `test_accuracy`, `train_macro_f1`, `cv_macro_f1`, `test_macro_f1` |
| `tree_feature_importance.csv` | 403 dòng với `feature_name`, `group`, `region`, `importance` |
| `tree_importance_by_group.csv`, `tree_importance_by_region.csv`, `tree_importance_by_group_region.csv` | Tổng `importance` và `feature_count` theo nhóm, vùng hoặc cặp nhóm–vùng |
| `figures/train_vs_validation.png`, `figures/confusion_matrix.png`, `figures/per_class_scores.png`, `figures/tree_feature_importance.png` | Biểu đồ được ghi khi chạy các ô tương ứng trong notebook |
