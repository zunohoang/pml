# Danh mục 403 đặc trưng

Tài liệu này dành cho người đọc vector đặc trưng hoặc biểu đồ độ quan trọng, cần biết mỗi họ cột đo điều gì. Bảng dưới bao phủ toàn bộ 403 cột và giữ thứ tự các khối trong mã.

`lesion` là vùng nghi tổn thương hợp lệ sau khi bỏ lóa, `leaf` là toàn vùng lá. Ví dụ, histogram chia màu thành các ô rồi đếm số pixel trong từng ô. Khi vùng `lesion` nhỏ hơn 1% diện tích lá, toàn bộ 135 cột của vùng này bằng 0. [Tách vùng](segmentation-explained.md)

## Danh mục theo họ

| Vùng | Nhóm | Họ và mẫu tên | Số cột | Ý nghĩa | Công thức hoặc hàm |
| --- | --- | --- | ---: | --- | --- |
| `lesion` | `color` | `lesion_hsv_{h,s,v}_{00…15}` | 48 | Phân bố sắc độ, bão hòa và độ sáng trong vùng. | `rgb2hsv`, H nhân 360. `np.histogram` 16 ô mỗi kênh trong [0; 360], [0; 1], [0; 1]. Dùng RGB đã chỉnh. |
| `lesion` | `color` | `lesion_lab_{l,a,b}_{00…15}` | 48 | Phân bố độ sáng và hai trục màu Lab trong vùng. | `rgb2lab` trên RGB đã chỉnh, 16 ô mỗi kênh trong [0; 100], [−128; 128], [−128; 128]. |
| `lesion` | `color` | `lesion_lab_mean_{l,a,b}` | 3 | Màu và độ sáng trung bình. | Trung bình các pixel hợp lệ theo kênh Lab trên RGB đã chỉnh. |
| `lesion` | `color` | `lesion_lab_std_{l,a,b}` | 3 | Mức phân tán màu hoặc độ sáng trong vùng. | Độ lệch chuẩn theo kênh Lab với `ddof` = 0. |
| `lesion` | `texture` | `lesion_glcm_{chỉ_số}` | 6 | Kiểu kết hợp mức xám giữa hai pixel gần nhau. | `graycomatrix` và `graycoprops`, khoảng cách 1 px, trung bình 4 hướng trên ảnh xám từ RGB đã chỉnh. |
| `lesion` | `texture` | `lesion_lbp_r{1,2}_{00…09}` | 20 | Tần suất các kiểu sáng tối quanh pixel trong vùng. | `local_binary_pattern`, 8 láng giềng, `method` là `uniform`, bán kính 1 và 2 px, histogram 10 ô. Không lấy căn. |
| `lesion` | `shape` | `lesion_shape_region_count` | 1 | Số mảng rời trong mặt nạ. | `label` với `connectivity` = 1, nối pixel chung cạnh. |
| `lesion` | `shape` | `lesion_shape_mean_area_ratio`, `lesion_shape_max_area_ratio` | 2 | Kích thước điển hình và lớn nhất của mảng so với lá. | Trung bình và cực đại diện tích từ `regionprops`, chia diện tích lá. |
| `lesion` | `shape` | `lesion_shape_mean_eccentricity`, `lesion_shape_mean_solidity` | 2 | Mảng có thon và có lồi đầy hay không. | Trung bình độ lệch tâm và độ đặc từ `regionprops`. Độ đặc so diện tích mảng với bao lồi của nó. |
| `lesion` | `shape` | `lesion_shape_mean_border_distance` | 1 | Vùng nằm sát mép hay sâu trong lá. | `distance_transform_edt` trên mặt nạ lá đệm một viền False. Lấy trung bình trong vùng, chia căn diện tích lá. |
| `lesion` | `shape` | `lesion_shape_area_ratio` | 1 | Phần diện tích lá được khoanh. | Số pixel vùng nghi tổn thương chia số pixel của lá. |
| `leaf` | `color` | `leaf_ab_hist_a{00…09}_b{00…09}` | 100 | Phân bố màu kết hợp trên cả lá. | `np.histogram2d` của a và b trên Lab từ RGB đã chỉnh. Khoảng a [−50; 50], b [−30; 70], 10 × 10 ô. Trải phẳng với a là chỉ số ngoài, chuẩn hóa rồi lấy căn. |
| `leaf` | `texture` | `leaf_lbp_{l,a,b}_r{1,2,3}_{00…09}` | 90 | Kiểu kết cấu cục bộ theo từng kênh màu và ba kích thước lân cận. | Lab từ RGB trước chỉnh, lượng tử hóa về 0–255. Dùng LBP uniform với 8 láng giềng, bán kính 1, 2, 3 px. Chuẩn hóa 10 ô mỗi histogram rồi lấy căn. |
| `leaf` | `gradient` | `leaf_grad_{l,a,b}_{00…07}` | 24 | Mức thay đổi của Lab giữa các pixel gần nhau trong lá. | `np.gradient` theo hai trục và `np.hypot` cho độ lớn trên RGB trước chỉnh đã đổi Lab. Co mặt nạ 3 × 3. Histogram 8 ô với L trong [0; 30], a và b trong [0; 20], chuẩn hóa rồi lấy căn. |
| `leaf` | `texture` | `leaf_glcm_{l,a,b}_d{1,2,4}_{chỉ_số}` | 54 | Mức thường gặp của các cặp ở ba khoảng cách, theo từng kênh màu. | Lab từ RGB trước chỉnh, lượng tử hóa, GLCM 16 mức và trung bình bốn hướng cho mỗi khoảng cách 1, 2, 4 px. |

- Các histogram chia số đếm cho tổng hoặc 1 nếu tổng bằng 0. LBP chỉ lấy tâm trong mặt nạ đã co với cửa sổ có cạnh `2r + 1` để bớt ảnh hưởng của pixel ngoài vùng.
- GLCM lượng tử hóa kênh byte về 16 mức và chỉ giữ cặp nếu cả hai đầu nằm trong mặt nạ. Dùng ma trận đối xứng đã chuẩn hóa. Bốn hướng gồm ngang, dọc và hai đường chéo. Sáu hậu tố `{chỉ_số}` theo thứ tự là `contrast`, `dissimilarity`, `homogeneity`, `asm`, `energy`, `correlation`. Với hướng không có cặp, sáu giá trị là 0, 0, 0, 0, 0 và 1.
- Ở kết cấu vùng nghi tổn thương, ảnh xám là tổng có trọng số RGB 0,299, 0,587 và 0,114, làm tròn về byte. Ở kết cấu toàn lá, ba kênh Lab được cắt và lượng tử hóa từ khoảng L [0; 100], a [−64; 64] và b [−64; 64].

## Đối chiếu tổng

| Nhóm | Vùng nghi tổn thương | Toàn lá | Tổng |
| --- | ---: | ---: | ---: |
| Màu | 102 | 100 | 202 |
| Kết cấu | 26 | 144 | 170 |
| Hình dạng | 7 | 0 | 7 |
| Gradient | 0 | 24 | 24 |
| **Tổng** | **135** | **268** | **403** |

## Cách đọc tên đặc trưng

| Tên thật | Cách tách và đọc |
| --- | --- |
| `lesion_lab_std_l` | `lesion` là vùng nghi tổn thương, `lab` là họ hệ màu Lab, `std` là độ lệch chuẩn và `l` là kênh sáng. |
| `leaf_ab_hist_a05_b07` | `leaf` là toàn lá, `ab` là hai trục màu Lab, `hist` là histogram chung, `a05` và `b07` là chỉ số ô bắt đầu từ 0. |
| `leaf_lbp_a_r2_04` | `leaf` là toàn lá, `lbp` là họ kết cấu, `a` là trục màu Lab, `r2` là bán kính 2 px và `04` là chỉ số ô mã LBP. |

`list_feature_names` tạo danh sách theo thứ tự vector và `describe_feature` suy ra cặp vùng, nhóm từ tên. [Mã ghép tên](../src/coffee_leaf/features/extractor.py) · [Mã ánh xạ nhóm](../src/coffee_leaf/features/feature_names.py)

<details>
<summary>Danh sách đủ 403 tên theo thứ tự cột</summary>

```text
lesion_hsv_h_00
lesion_hsv_h_01
lesion_hsv_h_02
lesion_hsv_h_03
lesion_hsv_h_04
lesion_hsv_h_05
lesion_hsv_h_06
lesion_hsv_h_07
lesion_hsv_h_08
lesion_hsv_h_09
lesion_hsv_h_10
lesion_hsv_h_11
lesion_hsv_h_12
lesion_hsv_h_13
lesion_hsv_h_14
lesion_hsv_h_15
lesion_hsv_s_00
lesion_hsv_s_01
lesion_hsv_s_02
lesion_hsv_s_03
lesion_hsv_s_04
lesion_hsv_s_05
lesion_hsv_s_06
lesion_hsv_s_07
lesion_hsv_s_08
lesion_hsv_s_09
lesion_hsv_s_10
lesion_hsv_s_11
lesion_hsv_s_12
lesion_hsv_s_13
lesion_hsv_s_14
lesion_hsv_s_15
lesion_hsv_v_00
lesion_hsv_v_01
lesion_hsv_v_02
lesion_hsv_v_03
lesion_hsv_v_04
lesion_hsv_v_05
lesion_hsv_v_06
lesion_hsv_v_07
lesion_hsv_v_08
lesion_hsv_v_09
lesion_hsv_v_10
lesion_hsv_v_11
lesion_hsv_v_12
lesion_hsv_v_13
lesion_hsv_v_14
lesion_hsv_v_15
lesion_lab_l_00
lesion_lab_l_01
lesion_lab_l_02
lesion_lab_l_03
lesion_lab_l_04
lesion_lab_l_05
lesion_lab_l_06
lesion_lab_l_07
lesion_lab_l_08
lesion_lab_l_09
lesion_lab_l_10
lesion_lab_l_11
lesion_lab_l_12
lesion_lab_l_13
lesion_lab_l_14
lesion_lab_l_15
lesion_lab_a_00
lesion_lab_a_01
lesion_lab_a_02
lesion_lab_a_03
lesion_lab_a_04
lesion_lab_a_05
lesion_lab_a_06
lesion_lab_a_07
lesion_lab_a_08
lesion_lab_a_09
lesion_lab_a_10
lesion_lab_a_11
lesion_lab_a_12
lesion_lab_a_13
lesion_lab_a_14
lesion_lab_a_15
lesion_lab_b_00
lesion_lab_b_01
lesion_lab_b_02
lesion_lab_b_03
lesion_lab_b_04
lesion_lab_b_05
lesion_lab_b_06
lesion_lab_b_07
lesion_lab_b_08
lesion_lab_b_09
lesion_lab_b_10
lesion_lab_b_11
lesion_lab_b_12
lesion_lab_b_13
lesion_lab_b_14
lesion_lab_b_15
lesion_lab_mean_l
lesion_lab_mean_a
lesion_lab_mean_b
lesion_lab_std_l
lesion_lab_std_a
lesion_lab_std_b
lesion_glcm_contrast
lesion_glcm_dissimilarity
lesion_glcm_homogeneity
lesion_glcm_asm
lesion_glcm_energy
lesion_glcm_correlation
lesion_lbp_r1_00
lesion_lbp_r1_01
lesion_lbp_r1_02
lesion_lbp_r1_03
lesion_lbp_r1_04
lesion_lbp_r1_05
lesion_lbp_r1_06
lesion_lbp_r1_07
lesion_lbp_r1_08
lesion_lbp_r1_09
lesion_lbp_r2_00
lesion_lbp_r2_01
lesion_lbp_r2_02
lesion_lbp_r2_03
lesion_lbp_r2_04
lesion_lbp_r2_05
lesion_lbp_r2_06
lesion_lbp_r2_07
lesion_lbp_r2_08
lesion_lbp_r2_09
lesion_shape_region_count
lesion_shape_mean_area_ratio
lesion_shape_max_area_ratio
lesion_shape_mean_eccentricity
lesion_shape_mean_solidity
lesion_shape_mean_border_distance
lesion_shape_area_ratio
leaf_ab_hist_a00_b00
leaf_ab_hist_a00_b01
leaf_ab_hist_a00_b02
leaf_ab_hist_a00_b03
leaf_ab_hist_a00_b04
leaf_ab_hist_a00_b05
leaf_ab_hist_a00_b06
leaf_ab_hist_a00_b07
leaf_ab_hist_a00_b08
leaf_ab_hist_a00_b09
leaf_ab_hist_a01_b00
leaf_ab_hist_a01_b01
leaf_ab_hist_a01_b02
leaf_ab_hist_a01_b03
leaf_ab_hist_a01_b04
leaf_ab_hist_a01_b05
leaf_ab_hist_a01_b06
leaf_ab_hist_a01_b07
leaf_ab_hist_a01_b08
leaf_ab_hist_a01_b09
leaf_ab_hist_a02_b00
leaf_ab_hist_a02_b01
leaf_ab_hist_a02_b02
leaf_ab_hist_a02_b03
leaf_ab_hist_a02_b04
leaf_ab_hist_a02_b05
leaf_ab_hist_a02_b06
leaf_ab_hist_a02_b07
leaf_ab_hist_a02_b08
leaf_ab_hist_a02_b09
leaf_ab_hist_a03_b00
leaf_ab_hist_a03_b01
leaf_ab_hist_a03_b02
leaf_ab_hist_a03_b03
leaf_ab_hist_a03_b04
leaf_ab_hist_a03_b05
leaf_ab_hist_a03_b06
leaf_ab_hist_a03_b07
leaf_ab_hist_a03_b08
leaf_ab_hist_a03_b09
leaf_ab_hist_a04_b00
leaf_ab_hist_a04_b01
leaf_ab_hist_a04_b02
leaf_ab_hist_a04_b03
leaf_ab_hist_a04_b04
leaf_ab_hist_a04_b05
leaf_ab_hist_a04_b06
leaf_ab_hist_a04_b07
leaf_ab_hist_a04_b08
leaf_ab_hist_a04_b09
leaf_ab_hist_a05_b00
leaf_ab_hist_a05_b01
leaf_ab_hist_a05_b02
leaf_ab_hist_a05_b03
leaf_ab_hist_a05_b04
leaf_ab_hist_a05_b05
leaf_ab_hist_a05_b06
leaf_ab_hist_a05_b07
leaf_ab_hist_a05_b08
leaf_ab_hist_a05_b09
leaf_ab_hist_a06_b00
leaf_ab_hist_a06_b01
leaf_ab_hist_a06_b02
leaf_ab_hist_a06_b03
leaf_ab_hist_a06_b04
leaf_ab_hist_a06_b05
leaf_ab_hist_a06_b06
leaf_ab_hist_a06_b07
leaf_ab_hist_a06_b08
leaf_ab_hist_a06_b09
leaf_ab_hist_a07_b00
leaf_ab_hist_a07_b01
leaf_ab_hist_a07_b02
leaf_ab_hist_a07_b03
leaf_ab_hist_a07_b04
leaf_ab_hist_a07_b05
leaf_ab_hist_a07_b06
leaf_ab_hist_a07_b07
leaf_ab_hist_a07_b08
leaf_ab_hist_a07_b09
leaf_ab_hist_a08_b00
leaf_ab_hist_a08_b01
leaf_ab_hist_a08_b02
leaf_ab_hist_a08_b03
leaf_ab_hist_a08_b04
leaf_ab_hist_a08_b05
leaf_ab_hist_a08_b06
leaf_ab_hist_a08_b07
leaf_ab_hist_a08_b08
leaf_ab_hist_a08_b09
leaf_ab_hist_a09_b00
leaf_ab_hist_a09_b01
leaf_ab_hist_a09_b02
leaf_ab_hist_a09_b03
leaf_ab_hist_a09_b04
leaf_ab_hist_a09_b05
leaf_ab_hist_a09_b06
leaf_ab_hist_a09_b07
leaf_ab_hist_a09_b08
leaf_ab_hist_a09_b09
leaf_lbp_l_r1_00
leaf_lbp_l_r1_01
leaf_lbp_l_r1_02
leaf_lbp_l_r1_03
leaf_lbp_l_r1_04
leaf_lbp_l_r1_05
leaf_lbp_l_r1_06
leaf_lbp_l_r1_07
leaf_lbp_l_r1_08
leaf_lbp_l_r1_09
leaf_lbp_l_r2_00
leaf_lbp_l_r2_01
leaf_lbp_l_r2_02
leaf_lbp_l_r2_03
leaf_lbp_l_r2_04
leaf_lbp_l_r2_05
leaf_lbp_l_r2_06
leaf_lbp_l_r2_07
leaf_lbp_l_r2_08
leaf_lbp_l_r2_09
leaf_lbp_l_r3_00
leaf_lbp_l_r3_01
leaf_lbp_l_r3_02
leaf_lbp_l_r3_03
leaf_lbp_l_r3_04
leaf_lbp_l_r3_05
leaf_lbp_l_r3_06
leaf_lbp_l_r3_07
leaf_lbp_l_r3_08
leaf_lbp_l_r3_09
leaf_lbp_a_r1_00
leaf_lbp_a_r1_01
leaf_lbp_a_r1_02
leaf_lbp_a_r1_03
leaf_lbp_a_r1_04
leaf_lbp_a_r1_05
leaf_lbp_a_r1_06
leaf_lbp_a_r1_07
leaf_lbp_a_r1_08
leaf_lbp_a_r1_09
leaf_lbp_a_r2_00
leaf_lbp_a_r2_01
leaf_lbp_a_r2_02
leaf_lbp_a_r2_03
leaf_lbp_a_r2_04
leaf_lbp_a_r2_05
leaf_lbp_a_r2_06
leaf_lbp_a_r2_07
leaf_lbp_a_r2_08
leaf_lbp_a_r2_09
leaf_lbp_a_r3_00
leaf_lbp_a_r3_01
leaf_lbp_a_r3_02
leaf_lbp_a_r3_03
leaf_lbp_a_r3_04
leaf_lbp_a_r3_05
leaf_lbp_a_r3_06
leaf_lbp_a_r3_07
leaf_lbp_a_r3_08
leaf_lbp_a_r3_09
leaf_lbp_b_r1_00
leaf_lbp_b_r1_01
leaf_lbp_b_r1_02
leaf_lbp_b_r1_03
leaf_lbp_b_r1_04
leaf_lbp_b_r1_05
leaf_lbp_b_r1_06
leaf_lbp_b_r1_07
leaf_lbp_b_r1_08
leaf_lbp_b_r1_09
leaf_lbp_b_r2_00
leaf_lbp_b_r2_01
leaf_lbp_b_r2_02
leaf_lbp_b_r2_03
leaf_lbp_b_r2_04
leaf_lbp_b_r2_05
leaf_lbp_b_r2_06
leaf_lbp_b_r2_07
leaf_lbp_b_r2_08
leaf_lbp_b_r2_09
leaf_lbp_b_r3_00
leaf_lbp_b_r3_01
leaf_lbp_b_r3_02
leaf_lbp_b_r3_03
leaf_lbp_b_r3_04
leaf_lbp_b_r3_05
leaf_lbp_b_r3_06
leaf_lbp_b_r3_07
leaf_lbp_b_r3_08
leaf_lbp_b_r3_09
leaf_grad_l_00
leaf_grad_l_01
leaf_grad_l_02
leaf_grad_l_03
leaf_grad_l_04
leaf_grad_l_05
leaf_grad_l_06
leaf_grad_l_07
leaf_grad_a_00
leaf_grad_a_01
leaf_grad_a_02
leaf_grad_a_03
leaf_grad_a_04
leaf_grad_a_05
leaf_grad_a_06
leaf_grad_a_07
leaf_grad_b_00
leaf_grad_b_01
leaf_grad_b_02
leaf_grad_b_03
leaf_grad_b_04
leaf_grad_b_05
leaf_grad_b_06
leaf_grad_b_07
leaf_glcm_l_d1_contrast
leaf_glcm_l_d1_dissimilarity
leaf_glcm_l_d1_homogeneity
leaf_glcm_l_d1_asm
leaf_glcm_l_d1_energy
leaf_glcm_l_d1_correlation
leaf_glcm_l_d2_contrast
leaf_glcm_l_d2_dissimilarity
leaf_glcm_l_d2_homogeneity
leaf_glcm_l_d2_asm
leaf_glcm_l_d2_energy
leaf_glcm_l_d2_correlation
leaf_glcm_l_d4_contrast
leaf_glcm_l_d4_dissimilarity
leaf_glcm_l_d4_homogeneity
leaf_glcm_l_d4_asm
leaf_glcm_l_d4_energy
leaf_glcm_l_d4_correlation
leaf_glcm_a_d1_contrast
leaf_glcm_a_d1_dissimilarity
leaf_glcm_a_d1_homogeneity
leaf_glcm_a_d1_asm
leaf_glcm_a_d1_energy
leaf_glcm_a_d1_correlation
leaf_glcm_a_d2_contrast
leaf_glcm_a_d2_dissimilarity
leaf_glcm_a_d2_homogeneity
leaf_glcm_a_d2_asm
leaf_glcm_a_d2_energy
leaf_glcm_a_d2_correlation
leaf_glcm_a_d4_contrast
leaf_glcm_a_d4_dissimilarity
leaf_glcm_a_d4_homogeneity
leaf_glcm_a_d4_asm
leaf_glcm_a_d4_energy
leaf_glcm_a_d4_correlation
leaf_glcm_b_d1_contrast
leaf_glcm_b_d1_dissimilarity
leaf_glcm_b_d1_homogeneity
leaf_glcm_b_d1_asm
leaf_glcm_b_d1_energy
leaf_glcm_b_d1_correlation
leaf_glcm_b_d2_contrast
leaf_glcm_b_d2_dissimilarity
leaf_glcm_b_d2_homogeneity
leaf_glcm_b_d2_asm
leaf_glcm_b_d2_energy
leaf_glcm_b_d2_correlation
leaf_glcm_b_d4_contrast
leaf_glcm_b_d4_dissimilarity
leaf_glcm_b_d4_homogeneity
leaf_glcm_b_d4_asm
leaf_glcm_b_d4_energy
leaf_glcm_b_d4_correlation
```

</details>
