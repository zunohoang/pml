# Giải thích cách tách vùng ảnh

Tài liệu này dành cho người biết lập trình và học máy nhưng mới làm quen với xử lý ảnh. Tài liệu giải thích cách dự án khoanh lá và vùng nghi tổn thương, cùng các tình huống mà mỗi bước có thể sai.

Mặt nạ là mảng bool có cùng chiều cao và chiều rộng với ảnh. Ví dụ, mặt nạ lá đánh dấu `True` ở pixel được xem là thuộc lá. RGB trước chỉnh màu ở đây là ảnh đã sửa hướng EXIF và thu nhỏ. [Đọc và chỉnh ảnh](pipeline.md#tiền-xử-lý-và-tách-vùng)

## Tách lá

HSV mô tả sắc độ theo góc `H`, độ bão hòa `S` và độ sáng `V`. Ví dụ, một tờ giấy trắng có thể rất sáng nhưng độ bão hòa thấp, còn lá xanh thường có độ bão hòa cao hơn. `find_leaf_mask` dùng HSV và dấu hiệu xanh dư để tách lá khỏi nền.

| Bước và hàm | Cách làm trong code | Có thể sai khi |
| --- | --- | --- |
| Chuyển HSV và chọn ứng viên qua `find_leaf_mask`, `find_leaf_color_candidates` | `rgb2hsv` cho H chuẩn hóa. Nhân H với `HUE_FULL_CIRCLE` = 360 để đổi sang độ. Giữ H từ `LEAF_MIN_HUE` = 8 đến `LEAF_MAX_HUE` = 170, gồm cả hai đầu. Yêu cầu S > `LEAF_MIN_SATURATION` = 0,15 và V > `LEAF_MIN_VALUE` = 0,06. | Lá quá tối, bạc màu hoặc đổi sắc vượt ngưỡng có thể mất pixel. Nền có màu tương tự có thể lọt vào. |
| Tính xanh dư qua `calculate_excess_green` | Chia RGB cho `MAX_RGB_VALUE` = 255, rồi chia mỗi kênh cho tổng ba kênh của cùng pixel, chặn mẫu số dưới ở `MIN_RGB_SUM` = 10⁻⁸. Chỉ số Excess Green là 2g − r − b trên các kênh đã chuẩn hóa. Ví dụ, r = 0,25, g = 0,50, b = 0,25 cho chỉ số 0,50. | Nền xanh cũng có thể có chỉ số cao. Chỉ số phản ánh màu của pixel. |
| Tạo mốc qua `find_leaf_seed_mask` | Đánh dấu pixel có Excess Green > `LEAF_MIN_SEED_EXCESS_GREEN` = 0,04 và S > 0,15. Mốc dùng để xếp hạng vùng ở bước sau. | Lá ngả vàng hoặc nâu có thể không còn mốc xanh dù thuộc vùng lá. |
| Dọn ứng viên qua `clean_leaf_candidates` | Kernel vuông toàn 1, cạnh `LEAF_KERNEL_SIZE` = 3. Phép mở co rồi nở để bỏ đốm rời. Phép đóng nở rồi co để lấp khe nhỏ. Code dùng `cv2.morphologyEx` với `MORPH_OPEN` rồi `MORPH_CLOSE`. | Phần lá hẹp có thể mất sau phép mở. Phép đóng có thể nối nền sát mép lá. |
| Chọn vùng qua `select_leaf_region` | Gán nhãn theo 4 liên thông với `FOUR_CONNECTIVITY` = 1, tức các pixel chung cạnh được nối. Bỏ vùng nền có ID `BACKGROUND_REGION_ID` = 0. Chọn vùng có nhiều mốc nhất. Nếu không có mốc trong các vùng, lấy vùng lớn nhất. Nếu chỉ còn nền, trả mặt nạ ứng viên. | Vật xanh lớn hoặc phần lá được tách rời có thể làm vùng được chọn lệch khỏi lá thật. |
| Lấp lỗ và kiểm tra qua `find_leaf_mask`, `validate_leaf_area` | `binary_fill_holes` lấp các lỗ bên trong vùng để phần bị đổi màu trên lá vẫn có thể nằm trong mặt nạ. Mặt nạ dưới `LEAF_MIN_PIXELS` = 40 pixel gây `ValueError`. | Lỗ thật do lá rách có thể bị lấp. Lá hợp lệ nhưng quá nhỏ có thể bị từ chối. |

## Tách vùng nghi tổn thương

Lab tách độ sáng `L` khỏi hai trục màu `a` và `b`. Ví dụ, hai vùng xanh và vàng có thể sáng ngang nhau nhưng nằm ở hai vị trí khác trong mặt phẳng `a`, `b`. `find_lesion_mask` tìm một cụm màu đủ xa màu phổ biến trong lá.

| Bước và hàm | Cách làm trong code | Có thể sai khi |
| --- | --- | --- |
| Đánh dấu lóa qua `find_glare_mask` | Trên RGB trước chỉnh màu, đổi HSV và giữ V ≥ `GLARE_MIN_VALUE` = 0,98 và S ≤ `GLARE_MAX_SATURATION` = 0,10. | Vùng tổn thương trắng có thể bị đánh dấu lóa. Vệt phản chiếu có màu có thể lọt qua. |
| Giới hạn vùng xét qua `find_lesion_mask` | Lấy phần lá ngoài lóa. Nếu có ít hơn `LESION_CLUSTER_COUNT` = 3 pixel, trả mặt nạ toàn `False`. | Mặt nạ lá thiếu hoặc lóa phủ rộng sẽ làm mất pixel ứng viên. |
| Lấy màu qua `extract_lesion_chroma` | Dùng `rgb2lab` trên ảnh đã chỉnh màu, rồi lấy riêng `a`, `b` của các pixel đang xét. Kết quả gồm một hàng mỗi pixel. | Dấu hiệu chỉ khác độ sáng có thể khó phát hiện vì `L` không tham gia gom cụm. |
| Gom màu qua `cluster_leaf_colors` | K-means tạo 3 cụm bằng `LESION_CLUSTER_COUNT`, với `LESION_KMEANS_RUNS` = 10 lần khởi tạo và `LESION_KMEANS_SEED` = 42. K-means nhóm các điểm gần nhau trong mặt phẳng màu, chẳng hạn gom các sắc xanh gần nhau vào cùng cụm. | Lá có chuyển màu tự nhiên hoặc nhiều kiểu vết có thể không tách thành ba cụm có ý nghĩa. |
| Chọn cụm qua `select_lesion_cluster` | Cụm có nhiều pixel nhất là mốc. Tính khoảng cách Euclid giữa các tâm và đặt khoảng cách của cụm rỗng bằng 0. Giữ cụm xa nhất khi khoảng cách ≥ `LESION_MIN_COLOR_DISTANCE` = 8. Trả `None` nếu không đủ xa. | Khi tổn thương chiếm đa số, cụm mốc có thể là màu tổn thương. Gân lá hoặc mép lá đổi màu có thể trở thành cụm xa nhất. |
| Dọn vùng nhỏ qua `remove_tiny_lesions` | Bỏ vùng 4 liên thông dưới `LESION_MIN_REGION_PIXELS` = 3. Dùng `remove_small_objects` với `max_size` = 2 nếu thư viện hỗ trợ, nếu không dùng `min_size` = 3. | Những vết thật nhỏ hoặc bị đứt đoạn sau khi thu nhỏ có thể bị xóa. |
| Giữ vùng hợp lệ và kiểm tra diện tích qua `build_lesion_roi_mask`, `compute_lesion_features` | Lấy giao vùng nghi tổn thương với lá, loại lóa rồi chia số pixel còn lại cho diện tích lá. Nếu nhỏ hơn `LESION_MIN_AREA_RATIO` = 0,01, đặt `LESION_FEATURE_COUNT` = 135 giá trị vùng này bằng 0. | Vết bệnh nhỏ có thể rơi dưới ngưỡng, nên vector của ảnh chỉ còn tín hiệu toàn lá. |

Trong kết quả đã ghi nhận, 44 ảnh có khối vùng nghi tổn thương bằng 0, gồm `brown_spot` 1, `cercospora` 2, `healthy` 12, `leaf_miner` 8 và `rust` 21. Vùng khoanh được là giả thuyết theo màu và chưa có mặt nạ gán tay để kiểm chứng. [Hạn chế và kết quả](evaluation-results.md#hạn-chế)

[Ngưỡng lóa và ảnh đầu vào](../src/coffee_leaf/preprocessing/image.py) · [Ngưỡng và thuật toán tách vùng](../src/coffee_leaf/preprocessing/segmentation.py) · [Tạo vùng hợp lệ](../src/coffee_leaf/features/masks.py) · [Ngưỡng khi ghép đặc trưng](../src/coffee_leaf/features/extractor.py)
