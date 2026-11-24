# Hướng dẫn đọc từng module

Tài liệu này dành cho người đọc biết Python và học máy, cần tra đầu vào, đầu ra và điều kiện lỗi của mỗi hàm. Có thể xem thứ tự đọc và các import nội bộ ở [Cấu trúc mã nguồn](code-structure.md).

Cột `ValueError` ghi các điều kiện kiểm tra của dự án và các lỗi thư viện quan trọng mà hàm truyền lên. Ký hiệu — chỉ hàm không có nhánh kiểm tra riêng cho lỗi này. Mặt nạ là mảng bool chọn pixel, ví dụ `True` tại điểm thuộc lá. RGB trước chỉnh màu trong tài liệu là ảnh đã sửa hướng và thu nhỏ.

## Package `coffee_leaf`

### `src/coffee_leaf/__init__.py`

Đánh dấu package gốc. [Mã nguồn](../src/coffee_leaf/__init__.py)

### `src/coffee_leaf/config.py`

Nạp tham số YAML để các bước dùng chung một cấu hình. [Mã nguồn](../src/coffee_leaf/config.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `load_config` | Đọc và giải mã YAML bằng safe_load. | Đường dẫn YAML. | Đối tượng YAML, với cấu hình này là dict. | — |

### `src/coffee_leaf/data/__init__.py`

Đánh dấu package đọc và chia dữ liệu. [Mã nguồn](../src/coffee_leaf/data/__init__.py)

### `src/coffee_leaf/data/loader.py`

Đọc nhãn và nối mỗi ID với ảnh có thật, theo quy tắc trong tài liệu BRACOL. [Mã nguồn](../src/coffee_leaf/data/loader.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `read_dataset_table` | Đọc dataset.csv từ thư mục dữ liệu. | Thư mục dữ liệu. | DataFrame các cột CSV. | — |
| `remove_ignored_codes` | Lọc mã cấu hình yêu cầu bỏ và trả bản sao. | DataFrame có predominant_stress, tập mã bỏ. | DataFrame đã lọc. | — |
| `assign_labels` | Ánh xạ mã thành nhãn và gán cột label tại chỗ. | DataFrame đã lọc, dict ánh xạ. | None, DataFrame đầu vào được cập nhật. | Còn mã không có trong ánh xạ. |
| `add_image_paths` | Ghép images và tên ảnh từ ID rồi gán cột image_path tại chỗ. | DataFrame có id, đường dẫn dữ liệu. | None, cột chứa các đối tượng Path. | — |
| `remove_missing_images` | Giữ các dòng có image_path trỏ tới tệp. | DataFrame có image_path. | DataFrame đã lọc. | — |
| `build_image_table` | Gọi read_dataset_table → remove_ignored_codes → assign_labels → add_image_paths → remove_missing_images, rồi lấy ba cột. | Dict cấu hình. | DataFrame id, label, image_path. | Mã còn lại không có nhãn, qua assign_labels. |

### `src/coffee_leaf/data/split.py`

Chia tập giữ lại để chấm test. [Mã nguồn](../src/coffee_leaf/data/split.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `split_train_test` | Gọi train_test_split phân tầng theo nhãn và đánh lại chỉ số mỗi tập. | DataFrame ảnh, cấu hình seed và test_size. | Hai DataFrame train và test. | Thư viện báo lỗi nếu nhãn hoặc kích thước không cho phép chia phân tầng. |

### `src/coffee_leaf/preprocessing/__init__.py`

Đánh dấu package xử lý ảnh. [Mã nguồn](../src/coffee_leaf/preprocessing/__init__.py)

### `src/coffee_leaf/preprocessing/image.py`

Sửa hướng, thu nhỏ, đánh dấu lóa và điều chỉnh màu, độ sáng. [Mã nguồn](../src/coffee_leaf/preprocessing/image.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `load_resized_rgb` | Áp dụng EXIF, đổi RGB và dùng thumbnail Lanczos. | Đường dẫn ảnh, cạnh tối đa. | RGB uint8 (H, W, 3). | — |
| `find_glare_mask` | Đánh dấu pixel rất sáng và ít bão hòa trong HSV. | RGB trước cân bằng trắng. | Mặt nạ bool (H, W). | — |
| `balance_white` | Tính và áp dụng hệ số gray-world có giới hạn. | RGB. | RGB uint8 cùng shape. | — |
| `equalize_lightness` | Chạy CLAHE trên kênh L rồi chuyển Lab về RGB. | RGB uint8. | RGB uint8 cùng shape. | — |
| `preprocess_rgb` | Gọi balance_white rồi equalize_lightness. | RGB uint8. | RGB uint8 đã chỉnh màu và tương phản. | — |

### `src/coffee_leaf/preprocessing/segmentation.py`

Tách lá bằng màu rồi đánh dấu vùng nghi tổn thương bằng cụm màu. Ngưỡng và diễn giải nằm trong tài liệu tách vùng. [Mã nguồn](../src/coffee_leaf/preprocessing/segmentation.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `calculate_excess_green` | Tính 2g − r − b sau khi chuẩn hóa màu từng pixel. | RGB trước chỉnh màu. | Mảng chỉ số thực (H, W). | — |
| `find_leaf_color_candidates` | Lọc khoảng sắc độ và ngưỡng bão hòa, độ sáng. | Ảnh HSV dạng float. | Mặt nạ ứng viên bool. | — |
| `find_leaf_seed_mask` | Giữ pixel xanh dư và đủ bão hòa làm mốc chọn vùng. | Excess Green và ảnh HSV. | Mặt nạ mốc bool. | — |
| `clean_leaf_candidates` | Mở rồi đóng mặt nạ bằng kernel vuông. | Mặt nạ ứng viên. | Mặt nạ bool đã dọn. | — |
| `select_leaf_region` | Chọn vùng có nhiều pixel mốc nhất, dùng diện tích nếu không có mốc. | Mặt nạ ứng viên sạch và mặt nạ mốc. | Mặt nạ bool của vùng được chọn, hoặc mặt nạ ứng viên khi chỉ có nền. | — |
| `validate_leaf_area` | Kiểm tra diện tích lá tối thiểu. | Mặt nạ lá. | None nếu đạt. | Dưới LEAF_MIN_PIXELS. |
| `find_leaf_mask` | Đổi HSV, tính Excess Green, tạo ứng viên và mốc, dọn, chọn vùng, lấp lỗ rồi gọi validate_leaf_area. | RGB trước chỉnh màu. | Mặt nạ lá bool. | Vùng lá dưới ngưỡng diện tích. |
| `extract_lesion_chroma` | Lấy hai kênh Lab a, b trong vùng đang xét. | RGB đã chỉnh, mặt nạ vùng xét. | Mảng (N, 2) màu Lab. | — |
| `cluster_leaf_colors` | Fit KMeans lên màu a, b. | Mảng (N, 2). | KMeans đã fit. | Thư viện từ chối dữ liệu không hợp lệ hoặc ít mẫu hơn số cụm. |
| `select_lesion_cluster` | Dùng cụm đông nhất làm mốc và chọn cụm có tâm xa nhất nếu đủ xa. | Mã cụm từng pixel, tọa độ tâm cụm. | ID cụm, hoặc None. | — |
| `remove_tiny_lesions` | Loại vùng nhỏ bằng remove_small_objects, chọn tham số theo phiên bản thư viện. | Mặt nạ nghi tổn thương. | Mặt nạ bool đã lọc. | — |
| `find_lesion_mask` | Lấy lá trừ lóa, kiểm tra số pixel, trích màu, gom cụm, chọn cụm và lọc vùng nhỏ. | RGB đã chỉnh, mặt nạ lá và lóa. | Mặt nạ bool, toàn False khi không đủ pixel hoặc không chọn được cụm. | Lỗi dữ liệu từ KMeans có thể truyền lên. |

### `src/coffee_leaf/features/__init__.py`

Đánh dấu package tạo và đặt tên đặc trưng. [Mã nguồn](../src/coffee_leaf/features/__init__.py)

### `src/coffee_leaf/features/channels.py`

Chuyển kênh màu và lượng tử hóa theo khoảng hữu hạn. [Mã nguồn](../src/coffee_leaf/features/channels.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `validate_rgb_image` | Kiểm tra kiểu, số kênh và kích thước ảnh. | Mảng RGB. | None nếu đạt. | Không phải ndarray uint8 (H, W, 3) hoặc cạnh rỗng. |
| `convert_to_hsv` | Đổi RGB sang HSV và đổi H sang độ. | RGB uint8. | HSV thực (H, W, 3). | RGB sai chuẩn. |
| `convert_to_lab` | Đổi RGB sang Lab bằng rgb2lab. | RGB uint8. | Lab thực (H, W, 3). | RGB sai chuẩn. |
| `convert_to_gray` | Nhân trọng số ba kênh RGB và làm tròn. | RGB uint8. | Ảnh xám uint8 (H, W). | RGB sai chuẩn. |
| `quantize_channel` | Cắt theo khoảng, ánh xạ tuyến tính sang 0–255 và làm tròn. | Kênh hữu hạn, hai cận low và high. | Kênh uint8 cùng shape. | Khoảng không hữu hạn, high không lớn hơn low hoặc dữ liệu chứa NaN hay Inf. |

### `src/coffee_leaf/features/masks.py`

Kiểm tra mặt nạ, tạo giao vùng và co mặt nạ để tránh biên. [Mã nguồn](../src/coffee_leaf/features/masks.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `validate_mask` | Kiểm tra mảng bool hai chiều và shape nếu cung cấp. | Mặt nạ, tên và shape tùy chọn. | None nếu đạt. | Sai kiểu, sai số chiều, cạnh rỗng hoặc lệch shape yêu cầu. |
| `validate_nonempty_mask` | Kiểm tra mặt nạ và yêu cầu có pixel được chọn. | Mặt nạ, tên và shape tùy chọn. | None nếu đạt. | Mặt nạ sai hoặc toàn False. |
| `build_lesion_roi_mask` | Lấy giao vùng nghi tổn thương với lá và loại pixel lóa. | Ba mặt nạ cùng shape. | Mặt nạ bool vùng nghi tổn thương hợp lệ. | Mặt nạ sai hoặc lệch shape. |
| `shrink_mask` | Co mặt nạ với phần tử cấu trúc vuông, coi ngoài ảnh là False. | Mặt nạ bool, cạnh cửa sổ nguyên dương. | Mặt nạ bool cùng shape. | Mặt nạ sai hoặc cạnh cửa sổ không phải số nguyên dương. |

### `src/coffee_leaf/features/color.py`

Tạo mô tả phân bố màu của hai vùng trên ảnh đã chỉnh. [Mã nguồn](../src/coffee_leaf/features/color.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `normalize_histogram` | Chia số đếm cho tổng hoặc 1 khi tổng bằng 0. | Mảng số đếm. | Mảng float64 chuẩn hóa. | Tổng không hữu hạn hoặc có số đếm âm. |
| `compute_channel_histograms` | Tính histogram từng kênh và nối theo thứ tự truyền vào. | Mảng pixel theo kênh, cấu hình khoảng. | Vector histogram chuẩn hóa. | Lỗi khoảng hoặc số đếm không hợp lệ có thể truyền lên. |
| `compute_lesion_color_features` | Ghép histogram HSV, Lab và thống kê Lab. | RGB đã chỉnh, mặt nạ vùng nghi tổn thương. | Vector (102,). | RGB sai hoặc mặt nạ sai, lệch shape hay rỗng. |
| `compute_leaf_color_features` | Tính histogram chung a, b, trải phẳng, chuẩn hóa rồi lấy căn. | RGB đã chỉnh, mặt nạ lá. | Vector (100,). | RGB sai hoặc mặt nạ sai, lệch shape hay rỗng. |
| `list_channel_histogram_names` | Đặt tên các ô histogram kênh màu. | Tiền tố, cấu hình tên và khoảng kênh. | Danh sách tên. | — |
| `list_lesion_color_names` | Liệt kê histogram HSV, Lab rồi mean và std Lab. | — | Danh sách 102 tên. | — |
| `list_leaf_color_names` | Liệt kê theo vòng ngoài a và vòng trong b. | — | Danh sách 100 tên. | — |

### `src/coffee_leaf/features/gradient.py`

Mô tả độ biến đổi màu lân cận trên toàn lá. [Mã nguồn](../src/coffee_leaf/features/gradient.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `compute_gradient_histograms` | Tính độ lớn gradient Lab, gom histogram, chuẩn hóa rồi lấy căn. | Ảnh Lab, mặt nạ lá đã co. | Vector (24,). | Lỗi shape, gradient hoặc histogram có thể truyền lên. |
| `compute_gradient_features` | Đổi Lab, kiểm tra ảnh và mặt nạ, co lá rồi tính các histogram. | RGB trước chỉnh, mặt nạ lá. | Vector (24,). | RGB sai, mặt nạ sai, rỗng hoặc lệch shape, hay cạnh ảnh dưới 2 pixel. |
| `list_gradient_names` | Đặt tên các ô lần lượt theo L, a, b. | — | Danh sách 24 tên. | — |

### `src/coffee_leaf/features/shape.py`

Đo hình dạng, kích thước và vị trí của các vùng nghi tổn thương. [Mã nguồn](../src/coffee_leaf/features/shape.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `validate_shape_masks` | Kiểm tra lá khác rỗng và vùng nghi tổn thương nằm trong lá. | Hai mặt nạ bool. | None nếu đạt. | Mặt nạ sai, lệch shape, lá rỗng hoặc vùng nằm ngoài lá. |
| `compute_region_features` | Đo số vùng, diện tích tương đối, độ lệch tâm và độ đặc. | Mặt nạ vùng hợp lệ khác rỗng, diện tích lá. | Vector (5,). | — |
| `compute_spatial_features` | Đo khoảng cách trung bình đến nền lá và tỷ lệ diện tích. | Mặt nạ vùng, mặt nạ lá, diện tích lá. | Vector (2,). | — |
| `compute_shape_features` | Kiểm tra mặt nạ rồi nối đặc trưng vùng và vị trí. | Mặt nạ vùng, mặt nạ lá. | Vector (7,), toàn 0 nếu vùng rỗng. | Mặt nạ không hợp lệ hoặc kết quả không hữu hạn. |
| `list_shape_names` | Lấy tên theo thứ tự vector hình dạng. | — | Danh sách 7 tên. | — |

### `src/coffee_leaf/features/texture.py`

Đo kết cấu từ so sánh láng giềng và cặp mức xám. [Mã nguồn](../src/coffee_leaf/features/texture.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `validate_channel` | Kiểm tra kênh ảnh nguyên, hai chiều và có kích thước dương. | Kênh ảnh. | None nếu đạt. | Không phải ndarray 2D nguyên hoặc có cạnh rỗng. |
| `validate_positive_integer` | Kiểm tra một tham số khoảng cách hoặc bán kính. | Giá trị, tên tham số. | None nếu đạt. | Không phải số nguyên dương hoặc là bool. |
| `reduce_to_levels` | Chia nguyên giá trị byte thành 16 mức. | Kênh uint8. | Kênh uint8 trong 0–15. | Kênh không hợp lệ hoặc dtype khác uint8. |
| `compute_lbp_histogram` | Tính LBP uniform tại tâm hợp lệ và chuẩn hóa số đếm. | Kênh nguyên, mặt nạ, bán kính. | Vector (10,), toàn 0 nếu không có tâm hợp lệ. | Kênh, mặt nạ hoặc bán kính không hợp lệ. |
| `get_offset_slices` | Tạo hai lát cắt tương ứng hai đầu của độ dời. | Chiều dài trục, độ dời. | Hai đối tượng slice. | — |
| `list_valid_pairs` | Lấy các cặp mức mà cả hai pixel nằm trong mặt nạ. | Ảnh mức xám, mặt nạ, độ dời hàng và cột. | Mảng (N, 2) có dtype của ảnh mức xám. | Kênh, mặt nạ không hợp lệ hoặc độ dời không nguyên. |
| `read_glcm_properties` | Lấy sáu chỉ số graycoprops và chặn correlation không hữu hạn. | Ma trận GLCM. | Vector float64 (6,). | Lỗi ma trận sai từ thư viện có thể truyền lên. |
| `compute_glcm_properties_for_offset` | Dựng GLCM cho một độ dời hoặc dùng giá trị mặc định khi không có cặp. | Ảnh mức xám, mặt nạ và độ dời. | Vector (6,). | Đầu vào cho cặp không hợp lệ hoặc mức ngoài 0–15. |
| `compute_glcm_features` | Lấy trung bình chỉ số GLCM qua bốn hướng. | Ảnh mức, mặt nạ, khoảng cách. | Vector (6,). | Khoảng cách, ảnh mức hoặc mặt nạ không hợp lệ. |
| `compute_lesion_texture_features` | Ghép GLCM và LBP của ảnh xám trong vùng nghi tổn thương. | RGB đã chỉnh, mặt nạ vùng. | Vector (26,). | RGB hoặc mặt nạ không hợp lệ. |
| `quantize_lab_texture_channels` | Đổi Lab và lượng tử hóa riêng từng kênh màu. | RGB trước chỉnh, mặt nạ lá. | Danh sách ba kênh uint8. | RGB sai hoặc mặt nạ sai, rỗng hay lệch shape. |
| `compute_leaf_lbp_features` | Ghép các histogram LBP toàn lá đã lấy căn. | RGB trước chỉnh, mặt nạ lá. | Vector (90,). | RGB hoặc mặt nạ lá không hợp lệ. |
| `compute_leaf_glcm_features` | Ghép chỉ số GLCM theo các kênh Lab và khoảng cách. | RGB trước chỉnh, mặt nạ lá. | Vector (54,). | RGB hoặc mặt nạ lá không hợp lệ. |
| `list_lbp_names` | Đặt tên từng ô LBP theo bán kính. | Tiền tố, danh sách bán kính. | Danh sách tên. | — |
| `list_glcm_names` | Ghép tiền tố với sáu tên chỉ số GLCM viết thường. | Tiền tố. | Danh sách sáu tên. | — |
| `list_lesion_texture_names` | Nối tên GLCM rồi LBP vùng nghi tổn thương. | — | Danh sách 26 tên. | — |
| `list_leaf_lbp_names` | Liệt kê theo kênh Lab rồi theo bán kính. | — | Danh sách 90 tên. | — |
| `list_leaf_glcm_names` | Liệt kê theo kênh Lab rồi theo khoảng cách. | — | Danh sách 54 tên. | — |

### `src/coffee_leaf/features/feature_names.py`

Ánh xạ tên cột sang vùng và nhóm báo cáo. [Mã nguồn](../src/coffee_leaf/features/feature_names.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `describe_feature` | Tách tên thành vùng, họ kỹ thuật và phần chi tiết. | Tên cột dạng chuỗi. | Cặp (region, group). | Không phải chuỗi hoặc cú pháp, vùng hay họ kỹ thuật không được nhận diện. |

### `src/coffee_leaf/features/extractor.py`

Điều phối tiền xử lý và kết hợp các khối thành vector cố định. [Mã nguồn](../src/coffee_leaf/features/extractor.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `list_feature_names` | Nối danh sách tên theo đúng thứ tự ghép vector. | — | Danh sách 403 tên. | — |
| `validate_feature_inputs` | Kiểm tra hai ảnh RGB cùng shape và mặt nạ lá. | Hai ảnh RGB và mặt nạ lá. | None nếu đạt. | RGB sai, shape khác nhau hoặc mặt nạ sai, lệch shape hay rỗng. |
| `compute_lesion_features` | Kiểm tra tỷ lệ vùng trên lá, tính màu, kết cấu và hình dạng nếu đủ lớn. | RGB đã chỉnh, mặt nạ vùng, mặt nạ lá hợp lệ. | Vector (135,), toàn 0 nếu vùng dưới ngưỡng. | Các phép tính đặc trưng có thể báo lỗi đầu vào không hợp lệ. |
| `compute_leaf_features` | Nối màu, LBP, gradient và GLCM toàn lá. | Hai ảnh RGB và mặt nạ lá. | Vector (268,). | Các phép tính đặc trưng có thể báo lỗi đầu vào không hợp lệ. |
| `validate_feature_vector` | Kiểm tra đúng số chiều và tất cả giá trị hữu hạn. | Vector đặc trưng. | None nếu đạt. | Shape khác (403,) hoặc có NaN hay Inf. |
| `extract_features` | Kiểm tra đầu vào, dựng vùng hợp lệ, tính hai khối và kiểm tra vector ghép. | Hai ảnh RGB, mặt nạ lá, vùng nghi tổn thương và lóa. | Vector float64 (403,). | Ảnh, mặt nạ hoặc vector kết quả không hợp lệ. |
| `extract_features_from_path` | Đọc và thu nhỏ, tìm lóa, chỉnh màu, tìm lá và vùng nghi tổn thương rồi gọi extract_features. | Đường dẫn ảnh, cạnh tối đa. | Vector float64 (403,). | Lá quá nhỏ hoặc đầu vào hay vector đặc trưng không hợp lệ. |

### `src/coffee_leaf/models/__init__.py`

Đánh dấu package mô hình. [Mã nguồn](../src/coffee_leaf/models/__init__.py)

### `src/coffee_leaf/models/builders.py`

Dựng pipeline có scaler rồi đến bộ phân loại. [Mã nguồn](../src/coffee_leaf/models/builders.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `validate_class_weight` | Kiểm tra hai lựa chọn trọng số lớp được hỗ trợ. | None hoặc chuỗi. | None nếu đạt. | Giá trị ngoài None và balanced. |
| `get_model_seed` | Đọc seed từ cấu hình truyền vào hoặc nạp cấu hình mặc định. | Dict cấu hình hoặc None. | Seed cho bộ phân loại. | — |
| `build_svm_pipeline` | Kiểm tra trọng số và ghép StandardScaler với SVC kernel RBF. | Trọng số lớp, cấu hình tùy chọn. | Pipeline chưa fit. | Trọng số lớp không hợp lệ. |
| `build_decision_tree_pipeline` | Kiểm tra trọng số và ghép StandardScaler với DecisionTreeClassifier. | Trọng số lớp, cấu hình tùy chọn. | Pipeline chưa fit. | Trọng số lớp không hợp lệ. |

### `src/coffee_leaf/models/search.py`

Cấu hình không gian tìm kiếm, CV theo nhóm và dự đoán ngoài fold. [Mã nguồn](../src/coffee_leaf/models/search.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `get_search_configuration` | Lấy cấu hình truyền vào hoặc đọc YAML mặc định. | Dict hoặc None. | Dict cấu hình. | — |
| `build_parameter_grid` | Thêm tiền tố bước pipeline vào các khóa lưới. | Tên mô hình, cấu hình tùy chọn. | Dict tham số grid search. | Tên mô hình ngoài svm và decision_tree. |
| `build_group_cross_validation` | Dựng StratifiedGroupKFold theo cấu hình. | Dict hoặc None. | Đối tượng bộ chia CV. | Thư viện báo lỗi nếu số fold không hợp lệ. |
| `run_grid_search` | Dựng bộ chia, tạo GridSearchCV chấm accuracy và macro-F1, fit lại theo macro-F1. | Pipeline, lưới, ma trận, nhãn, nhóm, cấu hình tùy chọn. | GridSearchCV đã fit. | Lỗi cấu hình CV, lưới hoặc dữ liệu từ thư viện truyền lên. |
| `build_cv_predictions` | Chạy cross_val_predict với cùng cách dựng fold. | Pipeline, ma trận, nhãn, nhóm, cấu hình. | Mảng nhãn ngoài fold theo hàng mẫu. | Lỗi chia fold hoặc dự đoán từ thư viện có thể truyền lên. |

### `src/coffee_leaf/models/persistence.py`

Ghi và nạp mô hình qua joblib. [Mã nguồn](../src/coffee_leaf/models/persistence.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `save_model` | Tạo thư mục cha và ghi mô hình. | Mô hình, đường dẫn. | None, tệp joblib. | — |
| `load_model` | Kiểm tra tệp tồn tại rồi nạp mô hình. | Đường dẫn tệp. | Đối tượng đã lưu. | Đường dẫn không phải tệp. |

### `src/coffee_leaf/evaluation/__init__.py`

Đánh dấu package đánh giá. [Mã nguồn](../src/coffee_leaf/evaluation/__init__.py)

### `src/coffee_leaf/evaluation/metrics.py`

Tính chỉ số dùng chung cho CV và test, xếp hạng ứng viên và định dạng dòng điểm. [Mã nguồn](../src/coffee_leaf/evaluation/metrics.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `summarize_candidate` | Lấy điểm tại cấu hình được chọn trong grid search. | Tên mô hình, trọng số, search đã fit, thời gian. | Dict ứng viên, tham số, estimator và điểm. | Có điểm hoặc thời gian không hữu hạn. |
| `format_cv_summary_row` | Đổi None thành none, mã hóa tham số JSON và định dạng điểm 4 chữ số. | Dict ứng viên. | Dict một dòng CV. | JSON chứa số không hữu hạn. |
| `get_candidate_rank` | Dựng khóa xếp hạng theo điểm CV rồi độ lệch chuẩn. | Dict ứng viên. | Cặp (macro-F1 CV, độ lệch chuẩn đổi dấu). | — |
| `select_best_candidate` | Lấy ứng viên lớn nhất theo khóa xếp hạng, giữ thứ tự khi hòa hoàn toàn. | Danh sách ứng viên. | Dict ứng viên được chọn. | Danh sách rỗng. |
| `read_cv_score_row` | Ép bốn điểm CV sang float và đổi tên trường macro-F1 CV. | Một dòng CSV ở dạng dict. | Dict bốn điểm. | Giá trị không chuyển được sang float hoặc không hữu hạn. |
| `format_test_result_row` | Giữ các cột kết quả và định dạng điểm 4 chữ số. | Dict kết quả của ứng viên. | Dict một dòng test. | — |
| `compute_accuracy_and_macro_f1` | Tính accuracy và F1 trung bình đều theo các lớp truyền vào. | Nhãn thật, nhãn dự đoán, ID các lớp. | Cặp điểm. | Nhãn không hợp lệ theo kiểm tra của scikit-learn. |
| `build_confusion_matrix_table` | Đếm thật–dự đoán và gắn tên hàng theo đúng thứ tự lớp. | Hai dãy nhãn, ID lớp, tên lớp. | DataFrame số nguyên với index actual_class. | Scikit-learn hoặc pandas báo lỗi với nhãn hoặc shape không tương thích. |
| `build_per_class_table` | Tính precision, recall, F1 và số mẫu từng lớp. | Hai dãy nhãn, ID lớp, tên lớp. | DataFrame class_name, precision, recall, f1, support. | Scikit-learn hoặc pandas báo lỗi với nhãn hoặc shape không tương thích. |

### `src/coffee_leaf/evaluation/feature_importance.py`

Chọn cây bằng CV và nhóm độ quan trọng của các cột. [Mã nguồn](../src/coffee_leaf/evaluation/feature_importance.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `select_best_tree_name` | Xếp hạng các cây theo macro-F1 CV, độ lệch chuẩn rồi ưu tiên none. | DataFrame CV. | Tên ứng viên cây. | Thiếu cột hoặc ứng viên cây, trọng số ngoài quy ước, hay điểm không hợp lệ hoặc không hữu hạn. |
| `find_tree_classifier` | Tìm bước DecisionTreeClassifier đầu tiên trong pipeline. | Pipeline. | Bộ phân loại cây. | Pipeline không có bước cây tương ứng. |
| `extract_tree_importances` | Lấy bản sao vector độ quan trọng của cây đã fit. | Pipeline, số cột kỳ vọng. | Vector độ quan trọng. | Thiếu cây, cây chưa fit, sai shape, có số âm, số không hữu hạn hoặc tổng lệch 1 quá ngưỡng. |
| `build_feature_table` | Ghép metadata với số đo và sắp giảm ổn định. | Tên, nhóm, vùng và độ quan trọng. | DataFrame feature_name, group, region, importance. | Bốn dãy không có cùng độ dài. |
| `summarize_importances` | Cộng độ quan trọng và đếm số cột theo khóa nhóm. | Bảng đặc trưng, danh sách cột nhóm. | DataFrame kèm importance và feature_count. | — |
| `summarize_by_column` | Gọi summarize_importances cho một cột. | Bảng đặc trưng, tên cột. | DataFrame tổng hợp theo nhóm hoặc vùng. | — |
| `summarize_by_group_and_region` | Gọi summarize_importances theo cặp nhóm và vùng. | Bảng đặc trưng. | DataFrame tổng hợp theo group và region. | — |

### `src/coffee_leaf/evaluation/plots.py`

Dựng Figure từ DataFrame đã đọc. Notebook ghi tệp và gọi show. [Mã nguồn](../src/coffee_leaf/evaluation/plots.py)

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `draw_validation_axis` | Vẽ cột train, CV và thanh sai số trên một trục. | Axes, vị trí, nhãn, ba dãy điểm, tiêu đề. | None, cập nhật Axes. | — |
| `plot_validation_scores` | Tạo hai trục accuracy và macro-F1 rồi gọi draw_validation_axis. | Bảng CV. | Figure hai trục. | — |
| `draw_matrix` | Vẽ một ma trận màu có tên lớp và số trong mỗi ô. | Axes, ma trận, tên lớp, tiêu đề, định dạng số. | None, cập nhật Axes. | — |
| `plot_confusion_matrices` | Tính tỷ lệ theo hàng và vẽ hai ma trận qua draw_matrix. | Bảng nhầm lẫn có tên lớp làm index. | Figure số đếm và tỷ lệ. | — |
| `draw_per_class_scores` | Vẽ ba cột điểm cạnh nhau cho mỗi lớp. | Axes, bảng theo lớp. | None, cập nhật Axes. | — |
| `plot_per_class_scores` | Tạo trục và gọi draw_per_class_scores. | Bảng theo lớp. | Figure một trục. | — |
| `draw_importance_axis` | Vẽ cột ngang cho một bảng độ quan trọng. | Axes, bảng, tên cột nhãn, tiêu đề. | None, cập nhật Axes. | — |
| `plot_tree_importance` | Lấy 15 dòng đầu và gọi draw_importance_axis cho ba bảng. | Bảng từng cột, theo nhóm, theo vùng. | Figure ba trục. | — |

### `src/coffee_leaf/inference/__init__.py`

Đánh dấu package suy luận. [Mã nguồn](../src/coffee_leaf/inference/__init__.py)

### `src/coffee_leaf/inference/predictor.py`

Tệp rỗng, không có hàm suy luận. [Mã nguồn](../src/coffee_leaf/inference/predictor.py)

## Các script ở gốc dự án

Package `analysis` tách phép tính trong `summaries.py` khỏi các hàm trả Figure trong `plots.py`. `scripts/run_eda.py` nạp CSV và archive train, dùng lại tiền xử lý và tách vùng, rồi ghi bảng, hình và JSON vào `docs/eda/`. [Notebook EDA](../notebooks/01_eda.ipynb) · [Báo cáo EDA](eda-report.md)

### `scripts/extract_features.py`

Chia mẫu, trích vector theo ảnh và ghi hai archive có cùng schema. [Mã nguồn](../scripts/extract_features.py)

Trình tự `main` là `load_config` → `load_samples` → `get_worker_count` → `compute_all_features`. Khi có lỗi ảnh, gọi `print_image_errors` và trả mã 1. Khi thành công, gọi `build_feature_archive` → `save_split_feature_files` và trả mã 0.

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `load_samples` | Dựng bảng ảnh, chia tập và đánh dấu is_train theo ID. | Dict cấu hình. | DataFrame ảnh có cột is_train. | Lỗi ánh xạ nhãn hoặc chia phân tầng có thể truyền lên. |
| `get_worker_count` | Lấy n_jobs hoặc mặc định số CPU trừ một, tối thiểu 1. | Dict cấu hình. | Số tiến trình. | — |
| `compute_feature_row` | Trích một vector và bắt lỗi để báo theo ID. | Dòng mẫu có id và image_path, cạnh tối đa. | Bộ ba ID, vector hoặc None, chuỗi lỗi hoặc None. | — |
| `collect_feature_rows` | Thu kết quả theo thứ tự và in tiến độ định kỳ. | Iterator kết quả, tổng số mẫu. | Cặp danh sách vector và lỗi. | — |
| `compute_all_features` | Chạy công việc theo ảnh bằng joblib và chuyển cho collect_feature_rows. | Bảng mẫu, cạnh tối đa, số tiến trình. | Cặp danh sách vector và lỗi. | joblib có thể báo lỗi tham số song song không hợp lệ. |
| `encode_label_ids` | Đổi tên nhãn sang chỉ số trong danh sách lớp. | Series nhãn mẫu, danh sách lớp. | Mảng int64. | Có nhãn không nằm trong label_names. |
| `build_feature_metadata` | Tạo tên cột, vùng và nhóm theo thứ tự vector. | — | Dict ba mảng chuỗi. | Tên sinh ra không hợp lệ có thể làm describe_feature báo lỗi. |
| `build_feature_archive` | Ghép ma trận, nhãn, ID, metadata và cờ chia tập. | Bảng mẫu, vector theo ảnh, danh sách lớp. | Dict mảng có cờ is_train. | Ma trận sai shape, không hữu hạn hoặc mã hóa nhãn lỗi. |
| `split_feature_arrays` | Tách các mảng theo mẫu và dùng chung metadata. | Dict mảng theo mẫu, metadata và cờ is_train. | Hai dict cho train và test. | Cờ không phải mảng bool 1D hoặc số hàng mảng mẫu không khớp. |
| `write_feature_archives` | Nén các dict mảng vào tệp tạm rồi chuyển tới đích. | Thư mục, các cặp tên tệp và dict. | None, các tệp NPZ. | —, đích là thư mục gây IsADirectoryError. |
| `save_feature_file` | Dựng archive chung và ghi một tệp theo tên được truyền. | Đường dẫn, bảng mẫu, vector, danh sách lớp. | None, một tệp NPZ có is_train. | Lỗi dữ liệu từ build_feature_archive. |
| `save_split_feature_files` | Tách archive, tạo hai tên có hậu tố train và test rồi ghi. | Đường dẫn cơ sở, dict chứa is_train. | None, hai tệp NPZ. | Lỗi cờ hoặc số hàng từ split_feature_arrays. |
| `print_image_errors` | In danh sách ảnh lỗi kèm thông báo. | Danh sách cặp ID và chuỗi lỗi. | None, stdout. | — |
| `main` | Gọi load_config → load_samples → get_worker_count → compute_all_features. Nếu thành công, dựng và ghi hai archive. | Hai đường dẫn tùy chọn. | Mã thoát 0 thành công, 1 nếu có ảnh lỗi. | Lỗi cấu hình dữ liệu hoặc dựng archive có thể truyền lên. |

### `scripts/train.py`

Huấn luyện bốn pipeline trên train, chọn theo CV và lưu báo cáo, mô hình. [Mã nguồn](../scripts/train.py)

Trình tự `main` là `load_config` → `load_training_data` → `validate_label_names` → `build_candidate_parameter_grids` → `train_candidate_models` → `write_cv_summary` → `print_training_summary` → `select_best_candidate` → `write_cv_diagnostics` → `save_candidate_models` → `save_best_model`.

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `validate_training_features` | Kiểm tra ma trận và danh sách tên cột. | Ma trận, tên đặc trưng. | None nếu đạt. | Ma trận rỗng hoặc sai 403 cột, tên lệch shape, dtype không phải số thực hoặc có NaN hay Inf. |
| `validate_training_sample_arrays` | Kiểm tra số hàng, kiểu ID và thứ tự mã hợp lệ. | ID nhãn, ID ảnh, tên lớp, số hàng. | None nếu đạt. | ID sai shape hoặc dtype, tên lớp không phải vector Unicode khác rỗng, hoặc ID nhãn ngoài phạm vi. |
| `load_training_data` | Đọc năm khóa cần cho huấn luyện, không cho pickle, rồi kiểm tra mảng. | Đường dẫn NPZ train, có mặc định. | Dict năm mảng. | Thiếu khóa, shape hoặc dữ liệu không hợp lệ. |
| `validate_label_names` | Đối chiếu thứ tự tên lớp trong NPZ với dict nhãn. | Tên lớp, ánh xạ mã. | None nếu khớp. | Mã không liên tục từ 0 hoặc tên, thứ tự không khớp. |
| `build_candidate_parameter_grids` | Dựng hai lưới thông qua build_parameter_grid. | Dict cấu hình. | Dict gồm lưới SVM và cây. | — |
| `train_candidate_model` | Dựng pipeline, chạy grid search, in thời gian và tóm tắt ứng viên. | Mô hình, trọng số, lưới, dữ liệu train, cấu hình. | Dict ứng viên với estimator và điểm. | Lỗi từ pipeline, grid search hoặc điểm không hữu hạn. |
| `train_candidate_models` | Lặp bốn tổ hợp mô hình và trọng số theo thứ tự cố định. | Dữ liệu train, lưới theo mô hình, cấu hình tùy chọn. | Danh sách bốn dict ứng viên. | Lỗi của train_candidate_model có thể truyền lên. |
| `write_cv_summary` | Định dạng các dòng và ghi CSV qua tệp tạm. | Bốn ứng viên, thư mục kết quả. | None, cv_results.csv. | Không đúng bốn ứng viên hoặc tham số JSON không hợp lệ. |
| `print_training_summary` | In điểm train, validation và chênh lệch macro-F1. | Danh sách ứng viên. | None, bảng stdout. | — |
| `write_confusion_matrix_file` | Ghi bảng nhầm lẫn có index giữ tên lớp thật. | Đường dẫn, DataFrame ma trận. | None, CSV. | — |
| `write_per_class_file` | Ghi bảng theo lớp với bốn chữ số cho các điểm. | Đường dẫn, DataFrame theo lớp. | None, CSV. | — |
| `save_cv_diagnostic_files` | Ghi hai báo cáo CV qua thư mục tạm. | Thư mục, bảng ma trận và bảng theo lớp. | None, hai CSV CV. | — |
| `print_confusion_matrix` | Canh nhãn và số đếm thành bảng chữ. | Tên lớp, ma trận số nguyên. | None, stdout. | Tên lớp rỗng làm phép max báo lỗi. |
| `write_cv_diagnostics` | Tạo dự đoán ngoài fold, dựng hai bảng, ghi tệp rồi in điểm và ma trận. | Ứng viên được chọn, dữ liệu train, cấu hình và thư mục. | None, hai CSV cùng điểm và ma trận stdout. | Lỗi chia fold, dự đoán hoặc nhãn có thể truyền lên. |
| `build_best_model_info` | Lấy metadata ứng viên, thứ tự lớp theo classes_ và phiên bản scikit-learn. | Ứng viên được chọn, dữ liệu train. | Dict ghi thành JSON. | — |
| `save_best_model` | Ghi pipeline được chọn và metadata qua thư mục tạm. | Ứng viên, thư mục, dữ liệu train. | None, tệp joblib và JSON. | Metadata JSON chứa số không hữu hạn. |
| `build_candidate_model_path` | Đặt tên tệp từ mô hình và trọng số lớp. | Thư mục, tên mô hình, trọng số. | Path dưới candidates. | — |
| `save_candidate_models` | Ghi từng pipeline qua thư mục tạm, rồi đặt vào candidates. | Danh sách ứng viên, thư mục kết quả. | None, các tệp joblib ứng viên. | — |
| `main` | Nạp cấu hình và train, kiểm tra nhãn, dựng lưới và fit, ghi CV, chọn ứng viên, ghi chẩn đoán và lưu các pipeline. | — | None, báo cáo CV, JSON và pipeline. | Lỗi kiểm tra dữ liệu hoặc huấn luyện có thể truyền lên. |

### `scripts/evaluate.py`

Chấm mô hình đã fit trên tập giữ lại. Đọc tên ứng viên đã chốt bằng CV và gắn cờ kết quả theo tên đó. [Mã nguồn](../scripts/evaluate.py)

Trình tự `main` là `load_config` → `load_best_model_info` → `load_test_data` → `load_selected_candidate_name` → `score_candidates` → `write_test_results` → `write_test_diagnostics` → `print_test_summary`.

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `load_best_model_info` | Đọc metadata ứng viên được chọn. | Thư mục kết quả. | Dict từ JSON. | JSON sai cú pháp hoặc gốc không phải object. |
| `validate_test_label_names` | So nhãn test với cấu hình và tên lớp trong metadata mô hình. | Tên lớp test, ánh xạ, dict metadata. | None nếu khớp. | Mã hay thứ tự lớp không đồng nhất. |
| `load_test_data` | Đọc năm mảng test rồi tái sử dụng các bộ kiểm tra shape và ID. | Đường dẫn NPZ test, ánh xạ, metadata. | Dict năm mảng. | Thiếu khóa, mảng sai hoặc tên lớp không khớp. |
| `load_selected_candidate_name` | Lấy cặp định danh đã lưu từ CV. | Dict metadata mô hình. | Cặp tên mô hình và trọng số. | Thiếu trường hoặc tổ hợp không nhận diện được. |
| `load_cv_scores` | Đọc và kiểm tra đủ bốn dòng CV hợp lệ rồi lập bảng tra. | Thư mục kết quả. | Dict điểm theo cặp ứng viên. | Thiếu cột, sai số dòng, trọng số sai, ứng viên lặp hoặc điểm không hợp lệ. |
| `score_candidate` | Nạp một pipeline, dự đoán và tính điểm test dùng chung. | Đường dẫn mô hình, ma trận test, nhãn, ID lớp. | Dict predictions, test_accuracy, test_macro_f1. | Tệp mô hình thiếu, dự đoán sai shape hoặc nhãn ngoài lớp, hay dữ liệu bị pipeline từ chối. |
| `score_candidates` | Duyệt bốn pipeline, gắn điểm CV và cờ selected_by_cv. | Dict dữ liệu test, cặp ứng viên chốt, thư mục. | Danh sách bốn dict kết quả. | Lỗi CSV CV hoặc lỗi khi chấm một ứng viên. |
| `write_test_results` | Định dạng và ghi bảng kết quả qua tệp tạm. | Danh sách kết quả, thư mục. | None, test_results.csv. | — |
| `write_test_diagnostics` | Dựng và ghi hai bảng từ dự đoán ứng viên chốt. | Kết quả, dữ liệu test, thư mục. | Ma trận số nguyên, hai CSV test. | Lỗi dữ liệu nhãn từ hàm thống kê có thể truyền lên. |
| `print_test_summary` | In điểm từng ứng viên rồi in ma trận của ứng viên chốt. | Kết quả, tên lớp, ma trận. | None, stdout. | — |
| `main` | Nạp cấu hình, metadata và test, tra ứng viên, dự đoán bốn mô hình, ghi hai loại báo cáo rồi in tóm tắt. | — | None, ba CSV test và stdout. | Lỗi dữ liệu, metadata hay mô hình có thể truyền lên. |

### `scripts/feature_importance.py`

Nạp cây đã fit để tổng hợp độ quan trọng theo metadata đặc trưng train. [Mã nguồn](../scripts/feature_importance.py)

Trình tự `main` là `load_config` → `load_feature_metadata` → `load_cv_results` → `load_best_tree_pipeline` → `compute_importance_tables` → `write_importance_files` → `prepare_importance_summary` → `print_importance_summary`.

| Hàm | Vai trò | Đầu vào | Đầu ra | Khi nào có `ValueError` |
| --- | --- | --- | --- | --- |
| `load_feature_metadata` | Đọc riêng tên, nhóm và vùng bằng np.load không cho pickle. | Đường dẫn NPZ train. | Dict ba mảng Unicode (403,). | Thiếu khóa, sai shape hoặc dtype không phải Unicode. |
| `load_cv_results` | Đọc bảng CV và giữ giá trị chữ none. | Thư mục kết quả. | DataFrame CV. | — |
| `load_best_tree_pipeline` | Chọn tên cây bằng CV rồi nạp joblib trong candidates. | DataFrame CV, thư mục kết quả. | Cặp tên ứng viên và pipeline. | Bảng CV không hợp lệ hoặc tệp mô hình thiếu. |
| `compute_importance_tables` | Lấy độ quan trọng, dựng bảng chi tiết và ba bảng tổng hợp. | Pipeline cây, dict metadata. | Dict bốn DataFrame. | Lỗi cây hoặc metadata không khớp. |
| `write_importance_files` | Ghi bốn CSV, làm tròn 4 chữ số ở bước xuất. | Dict bảng, thư mục kết quả. | None, bốn CSV. | — |
| `prepare_importance_summary` | Lấy 10 dòng đầu và hai bảng tổng hợp cho phần in. | Dict bốn DataFrame. | Bộ ba danh sách dòng. | — |
| `print_importance_summary` | In tên cây, 10 đặc trưng, bảng theo nhóm và vùng. | Tên cây, ba danh sách dòng. | None, stdout. | — |
| `main` | Nạp cấu hình, metadata train, điểm CV và cây, tính và ghi báo cáo, chuẩn bị rồi in tóm tắt. | — | None, bốn CSV và stdout. | Lỗi metadata, CV hoặc mô hình có thể truyền lên. |
