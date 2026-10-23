# Sinh 3D

> 3D là phương thức mà đòn bẩy từ 2D sang 3D mạnh nhất. Bước đột phá năm 2023 là 3D Gaussian Splatting. Cú đẩy sinh tạo 2024-2026 xếp chồng khuếch tán đa góc nhìn + tái dựng 3D lên trên để tạo ra vật thể và cảnh từ một prompt hoặc một bức ảnh.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 4 (Thị giác máy tính), Giai đoạn 8, Bài 07 (Latent diffusion)
**Time:** ~45 phút

## Vấn đề

Nội dung 3D rất khó chịu:

- **Cách biểu diễn.** Mesh (lưới đa giác), point cloud, lưới voxel, trường khoảng cách có dấu (SDF), neural radiance field (NeRF — trường bức xạ nơ-ron), Gaussian 3D. Mỗi cách đều có đánh đổi.
- **Khan hiếm dữ liệu.** ImageNet có 14M ảnh. Bộ dữ liệu 3D sạch lớn nhất (Objaverse-XL, 2023) có ~10M vật thể, phần lớn chất lượng thấp.
- **Bộ nhớ.** Một lưới voxel 512³ là 128M voxel; một NeRF cảnh dùng được cần 1M mẫu/tia. Sinh tạo khó hơn tái dựng.
- **Giám sát.** Với ảnh 2D bạn có điểm ảnh. Với 3D bạn thường chỉ có vài góc nhìn 2D và phải nâng lên 3D.

Ngăn xếp 2026 tách hai bài toán ra. Đầu tiên, sinh *ảnh multiview (đa góc nhìn) 2D* bằng một mô hình khuếch tán. Thứ hai, khớp một *biểu diễn 3D* (thường là Gaussian splatting (dựng ảnh bằng hạt Gaussian)) vào những ảnh đó.

## Khái niệm

![3D generation (sinh 3D): khuếch tán đa góc nhìn + tái dựng 3D](../assets/3d-generation.svg)

### Cách biểu diễn: 3D Gaussian Splatting (Kerbl et al., 2023)

Biểu diễn một cảnh thành một đám mây ~1M Gaussian 3D. Mỗi hạt có 59 tham số: vị trí (3), hiệp phương sai (6, hoặc quaternion 4 + scale 3), độ đục (1), màu spherical-harmonics (48 ở bậc 3, 3 ở bậc 0).

Render = phép chiếu + alpha-compositing. Nhanh (~100 fps ở 1080p trên 4090). Khả vi. Khớp bằng hạ gradient so với ảnh thật. Một cảnh khớp xong trong 5-30 phút trên GPU tiêu dùng.

Hai cải tiến 2023-2024 chồng lên trên:
- **Gaussian splat sinh tạo.** Các mô hình như LGM, LRM, InstantMesh dự đoán thẳng một đám mây Gaussian từ một hoặc vài ảnh.
- **4D Gaussian Splatting.** Gaussian có độ lệch riêng theo từng khung hình cho cảnh động.

### Khuếch tán đa góc nhìn

Tinh chỉnh một mô hình khuếch tán ảnh đã huấn luyện trước để sinh ra nhiều góc nhìn nhất quán của cùng một vật thể từ một prompt văn bản hoặc một ảnh đơn. Zero123 (Liu et al., 2023), MVDream (Shi et al., 2023), SV3D (Stability, 2024), CAT3D (Google, 2024). Thường xuất ra 4-16 góc nhìn quanh vật thể, rồi nâng lên 3D bằng Gaussian splatting hoặc NeRF.

### Pipeline văn bản sang 3D

| Mô hình | Đầu vào | Đầu ra | Thời gian |
|-------|-------|--------|------|
| DreamFusion (2022) | văn bản | NeRF qua SDS | ~1 giờ mỗi asset |
| Magic3D | văn bản | mesh + texture | ~40 phút |
| Shap-E (OpenAI, 2023) | văn bản | 3D ẩn | ~1 phút |
| SJC / ProlificDreamer | văn bản | NeRF / mesh | ~30 phút |
| LRM (Meta, 2023) | ảnh | triplane | ~5 giây |
| InstantMesh (2024) | ảnh | mesh | ~10 giây |
| SV3D (Stability, 2024) | ảnh | góc nhìn mới | ~2 phút |
| CAT3D (Google, 2024) | 1-64 ảnh | NeRF 3D | ~1 phút |
| TripoSR (2024) | ảnh | mesh | ~1 giây |
| Meshy 4 (2025) | văn bản + ảnh | mesh PBR | ~30 giây |
| Rodin Gen-1.5 (2025) | văn bản + ảnh | mesh PBR | ~60 giây |
| Tencent Hunyuan3D 2.0 (2025) | ảnh | mesh | ~30 giây |

Hướng đi 2025-2026: các mô hình trực tiếp văn bản sang mesh với vật liệu PBR phù hợp cho game engine. Bước trung gian khuếch tán đa góc nhìn vẫn là công thức chạy tốt nhất cho vật thể nói chung.

### NeRF (để có ngữ cảnh)

Neural Radiance Field (Mildenhall et al., 2020). Một MLP nhỏ nhận `(x, y, z, view direction)` và xuất ra `(color, density)`. Render bằng cách tích phân dọc theo các tia. Đánh bại tổng hợp góc nhìn (view synthesis) dựa trên mesh về chất lượng nhưng render chậm hơn 100-1000 lần. Bị Gaussian splatting thay thế trong phần lớn ứng dụng thời gian thực nhưng vẫn thống trị trong nghiên cứu.

```figure
v4-3d-multiview
```

## Tự tay làm

`code/main.py` cài đặt một phép khớp "Gaussian splatting" 2D đồ chơi: biểu diễn một ảnh đích tổng hợp (một dải gradient mượt) thành tổng các hạt Gaussian 2D. Tối ưu vị trí, màu và hiệp phương sai bằng hạ gradient để khớp với đích. Bạn sẽ thấy hai thao tác cốt lõi: render xuôi (splat + alpha-composite) và khớp bằng hạ gradient.

### Bước 1: Gaussian splat 2D

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### Bước 2: render bằng cách cộng các splat

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

3D Gaussian splatting thật sự sắp xếp các Gaussian theo độ sâu rồi alpha-composite theo thứ tự. Đồ chơi 2D của ta chỉ cộng lại.

### Bước 3: khớp bằng hạ gradient

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## Cạm bẫy

- **Góc nhìn không nhất quán.** Nếu bạn sinh 4 góc nhìn độc lập và chúng mâu thuẫn nhau về cấu trúc vật thể, phép khớp 3D sẽ bị mờ. Cách sửa: khuếch tán đa góc nhìn có attention chia sẻ.
- **Ảo giác mặt sau.** Từ một ảnh sang 3D buộc phải bịa ra phần không nhìn thấy. Chất lượng dao động rất mạnh.
- **Bùng nổ splat Gaussian.** Huấn luyện không ràng buộc sẽ phình lên 10M splat và overfit. Các heuristic làm dày + tỉa (từ bài báo 3D-GS gốc) là bắt buộc.
- **Vấn đề topology.** Mesh sinh từ trường ẩn (SDF) thường có lỗ hoặc tự cắt nhau. Chạy một bộ remesh (ví dụ voxel remesh của Blender) trước khi xuất xưởng.
- **Giấy phép dữ liệu huấn luyện.** Objaverse có giấy phép hỗn hợp; dùng thương mại tùy từng mô hình.

## Áp dụng

| Tác vụ | Lựa chọn 2026 |
|------|-----------|
| Tái dựng cảnh từ ảnh chụp | Gaussian splatting (3DGS, Gsplat, Scaniverse) |
| Vật thể văn bản sang 3D cho game | Meshy 4 hoặc Rodin Gen-1.5 (xuất PBR) |
| Ảnh sang 3D | Hunyuan3D 2.0, TripoSR, InstantMesh |
| Tổng hợp góc nhìn mới từ vài ảnh | CAT3D, SV3D |
| Tái dựng cảnh động | 4D Gaussian Splatting |
| Avatar / người mặc quần áo | Gaussian Avatar, HUGS |
| Nghiên cứu / SOTA | Thứ gì mới ra tuần trước |

Để đưa 3D vào sản phẩm trong pipeline game hay thương mại điện tử: Meshy 4 hoặc Rodin Gen-1.5 xuất mesh PBR dùng thẳng được trong Unity / Unreal.

## Đóng gói

Lưu `outputs/skill-3d-pipeline.md`. Skill nhận một bản brief 3D (đầu vào: văn bản / một ảnh / vài ảnh; đầu ra: mesh / splat / NeRF; mục đích dùng: render / game / VR) và xuất ra: pipeline (khuếch tán đa góc nhìn + khớp, hoặc mô hình mesh trực tiếp), mô hình nền, ngân sách vòng lặp, hậu xử lý topology, các kênh vật liệu cần có.

## Bài tập

1. **Dễ.** Chạy `code/main.py` với 4, 16, 64 Gaussian. Báo MSE cuối cùng so với đích.
2. **Trung bình.** Mở rộng sang Gaussian màu (RGB). Xác nhận bản tái dựng khớp với kiểu màu của đích.
3. **Khó.** Dùng gsplat hoặc Nerfstudio, tái dựng một vật thể thật từ bộ chụp 50 ảnh. Báo thời gian khớp và SSIM cuối cùng trên các góc nhìn giữ lại để kiểm tra.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| 3D Gaussian Splatting | "3DGS" | Cảnh là một đám mây Gaussian 3D; render bằng alpha-composite khả vi. |
| NeRF | "Neural radiance field" | MLP xuất ra màu + mật độ tại một điểm 3D; render bằng tích phân theo tia. |
| Triplane | "Ba mặt phẳng 2-D" | Phân rã 3D thành ba lưới đặc trưng 2-D thẳng trục; rẻ hơn thể tích. |
| SDS | "Score distillation sampling" | Huấn luyện mô hình 3D bằng cách dùng score của khuếch tán 2D làm pseudo-gradient. |
| Multi-view diffusion | "Nhiều góc nhìn cùng lúc" | Mô hình khuếch tán xuất ra một batch góc nhìn camera nhất quán. |
| PBR | "Physically-based rendering" | Vật liệu có các kênh albedo, roughness, metallic, normal. |
| Densification | "Làm mọc splat" | Heuristic huấn luyện 3DGS: tách / nhân bản splat ở vùng gradient cao. |

## Ghi chú sản xuất: 3D chưa có nền chung

Khác với ảnh (khuếch tán ẩn + DiT) và video (DiT không-thời gian), 3D không có một runtime thống trị duy nhất trong năm 2026. Cây quyết định sản xuất rẽ nhánh theo cách biểu diễn:

- **NeRF / triplane.** Suy luận là ray-marching + một lượt forward MLP cho mỗi mẫu. Một lần render 512² cần hàng triệu lượt forward MLP. Gom nhóm mẫu tia mạnh tay; SDPA/xformers dùng được.
- **Khuếch tán đa góc nhìn + tái dựng LRM.** Pipeline hai tầng. Tầng 1 (DiT đa góc nhìn) là một diffusion server y như Bài 07. Tầng 2 (transformer LRM) là một lượt forward một phát trên các góc nhìn. Hồ sơ độ trễ tổng thể là "khuếch tán + một phát" — chọn primitive phục vụ riêng cho từng tầng.
- **SDS / DreamFusion.** Tối ưu theo từng asset (score distillation sampling), không phải suy luận. Hãy dựng job, không phải request handler.

Với hầu hết sản phẩm năm 2026, câu trả lời đúng là "chạy một mô hình khuếch tán đa góc nhìn khi có yêu cầu, tái dựng thành 3DGS bất đồng bộ, phục vụ 3DGS để xem thời gian thực". Cách này tách khối lượng công việc gọn gàng giữa một GPU-inference server (nhanh) và một bộ tối ưu ngoại tuyến (chậm).

## Đọc thêm

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF.
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS.
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS.
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123.
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — khuếch tán đa góc nhìn.
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM.
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D.
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d-novel-multi-view-synthesis-and-3d-generation-from-a-single-image-using-latent-video-diffusion) — SV3D.

---

[← Bài trước: Sinh âm thanh](../../11-audio-generation/docs/vi.md) · [Bài sau: Flow Matching & Rectified Flows →](../../13-flow-matching-rectified-flows/docs/vi.md)
