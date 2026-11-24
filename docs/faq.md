# Câu hỏi khi bảo vệ

Tài liệu này dành cho người chuẩn bị trình bày dự án trước hội đồng có kiến thức lập trình và học máy. Mỗi câu tập trung vào một quyết định thiết kế hoặc một cách đọc kết quả.

### Vì sao dùng SVM và Decision Tree?

Hai mô hình học trên cùng vector 403 chiều để so sánh hai cách phân loại. SVM kernel RBF dùng quan hệ khoảng cách để tạo ranh giới phi tuyến. Decision Tree dùng các điều kiện chia nhánh và cho phép xem cột nào được dùng. [Cấu hình mô hình](model-training.md)

### Vì sao dùng đặc trưng thủ công thay vì mạng nơ-ron?

Phạm vi thí nghiệm là đo tín hiệu màu, kết cấu, hình dạng và gradient trên từng ảnh. Mỗi cột có định nghĩa và vùng đo rõ ràng, ví dụ một ô đếm pixel theo màu của toàn lá. Vector cố định cho phép so sánh hai bộ phân loại trên đúng cùng đầu vào. [Danh mục đặc trưng](feature-catalog.md)

### Có epoch không?

SVM và Decision Tree trong dự án không có tham số epoch. Phần chạy lặp là cross-validation 5 fold: mỗi cấu hình được fit một lần trên phần train của từng fold rồi chấm trên phần validation. [Cross-validation](model-training.md#cross-validation-và-quy-tắc-chọn)

### Chọn tham số và mô hình thế nào?

Grid search thử lưới trên tập train và dùng macro-F1 CV trung bình để chọn tham số mỗi ứng viên. Sau đó, dự án chọn ứng viên theo cùng chỉ số. Khi hòa, ưu tiên độ lệch chuẩn thấp hơn rồi dùng thứ tự ứng viên cố định. Test dành cho báo cáo sau khi chốt. [Quy tắc chọn](model-training.md#cross-validation-và-quy-tắc-chọn)

### `class_weight="balanced"` để làm gì?

Nó tăng ảnh hưởng của lớp có ít mẫu trong dữ liệu fit để lỗi của những lớp này được chú ý hơn. Trong lần chạy đã ghi, hai phiên bản SVM chỉ chênh khoảng 0,001 macro-F1 CV, nhỏ hơn dao động giữa các fold. [So sánh ứng viên](evaluation-results.md#phân-tích)

### 403 đặc trưng là gì và vì sao nhiều số?

Có 135 số đo ở vùng nghi tổn thương và 268 số đo trên toàn lá. Một phép đo có thể tạo nhiều cột. Ví dụ, histogram màu 10 × 10 đếm pixel theo 100 ô màu, cho 100 số. [Danh mục đặc trưng](feature-catalog.md)

### Vì sao dùng 1.342 ảnh thay vì 1.747?

1.747 là số dòng CSV. Dự án chỉ giữ dòng có mã `predominant_stress` từ 0 đến 4 và có ảnh khớp với ID. Hai điều kiện giữ lại 1.342 ảnh. [Bảng số dòng theo mã](BRACOL-dataset.md)

### Chia dữ liệu thế nào?

Chia phân tầng theo lớp với tỷ lệ 75/25, seed 42, thành 1.006 train và 336 test. CV 5 fold chạy bên trong tập train và dùng mỗi `image_id` làm một nhóm. [Chia train và test](BRACOL-dataset.md#chia-train-và-test)

### Có quá khớp không?

SVM được chọn có macro-F1 train 0,984, validation 0,821 và test 0,828, cho thấy mức quá khớp vừa phải. CV giúp đo chênh lệch, `StandardScaler` trong `Pipeline` chỉ học trên phần train của mỗi fold, còn tham số và mô hình được chọn trong tập train. [Phân tích kết quả](evaluation-results.md#phân-tích)

### Lớp nào khó nhất và vì sao?

`cercospora` có F1 thấp nhất, 0,5784 trên CV và 0,6076 trên test. Mô hình hay nhầm lớp này với `rust` và `leaf_miner`. Trong CSV gốc, 52 trong 147 dòng mang nhãn chính `cercospora` còn bật cờ `rust`, gợi ý một phần ảnh có dấu hiệu đồng thời. [Ma trận và phân tích lỗi](evaluation-results.md)

### Đặc trưng nào quan trọng và nên đọc con số ra sao?

Ở cây `decision_tree_none`, nhóm màu chiếm 0,717. Cột đầu là `lesion_lab_std_l` với 0,192. Đây là độ quan trọng của một cây. Cột tương quan có thể chia sẻ điểm và hình dạng 0,000 cho biết cây này không dùng nhóm đó. Giá trị của nhóm hình dạng với SVM cần được đo riêng. [Độ quan trọng của cây](evaluation-results.md#độ-quan-trọng-đặc-trưng-của-cây)

### Hạn chế chính là gì?

Dữ liệu giữ một nhãn chính dù có ảnh bật nhiều cờ bệnh. Thiếu ID cây và phiên chụp làm khó việc xác nhận mô hình tổng quát hóa sang cây mới. Mặt nạ nghi tổn thương dựa trên màu và chưa có mặt nạ gán tay để đối chiếu. (evaluation-results.md#hạn-chế)

### Vì sao SVM không bật `probability=True`?

Báo cáo dùng nhãn từ `predict` để tính accuracy, F1 và ma trận nhầm lẫn. Quy trình này không cần bước hiệu chỉnh xác suất bổ sung của SVC. [Cấu hình SVM](model-training.md#mô-hình-và-lưới-tham-số)
