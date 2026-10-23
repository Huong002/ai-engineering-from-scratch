# GAN có điều kiện & Pix2Pix

> Bước mở khoá lớn đầu tiên của giai đoạn 2014-2017 là điều khiển được thứ GAN tạo ra. Gắn thêm một nhãn, một ảnh, hay một câu. Pix2Pix làm phiên bản cho ảnh, và nó vẫn đánh bại mọi mô hình text-to-image tổng quát trên những tác vụ dịch ảnh sang ảnh (image-to-image translation) hẹp.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 8, Bài 03 (GAN), Giai đoạn 4, Bài 06 (U-Net), Giai đoạn 3, Bài 07 (CNN)
**Time:** ~75 phút

## Vấn đề

Một GAN không điều kiện lấy mẫu ra những khuôn mặt tuỳ ý. Hữu ích cho demo, vô dụng trong sản xuất. Bạn muốn: *ánh xạ một bản phác thành một bức ảnh*, *ánh xạ một bản đồ thành ảnh hàng không*, *ánh xạ cảnh ban ngày thành ban đêm*, *tô màu một ảnh xám*. Ở tất cả những việc này, bạn được cho một ảnh đầu vào `x` và phải xuất ra `y` có tương ứng ngữ nghĩa nào đó. Có nhiều `y` hợp lý cho mỗi `x`. Sai số bình phương trung bình san chúng thành một mớ nhoè. Mất mát đối kháng thì không, vì "trông thật" là sắc nét.

Conditional GAN (GAN có điều kiện; Mirza & Osindero, 2014) thêm một điều kiện `c` làm đầu vào cho cả `G` và `D`. Pix2Pix (Isola và cộng sự, 2017) chuyên hoá điều này: điều kiện là một ảnh đầu vào đầy đủ, bộ sinh là U-Net, bộ phân biệt là một bộ phân loại *theo mảng* (patch discriminator — bộ phân biệt theo mảng), tức PatchGAN, và mất mát là đối kháng + L1. Công thức đó vượt các mô hình text-to-image làm từ đầu trên những miền dịch ảnh sang ảnh hẹp, ngay cả năm 2026, bởi vì nó được huấn luyện trên *dữ liệu theo cặp* (paired data) — bạn có đúng tín hiệu mình cần.

## Khái niệm

![Pix2Pix: bộ sinh U-Net, bộ phân biệt PatchGAN](../assets/pix2pix.svg)

**G có điều kiện.** `G(x, z) → y`. Trong Pix2Pix, `z` là dropout bên trong G (không có nhiễu đầu vào — Isola nhận thấy nhiễu tường minh bị bỏ qua).

**D có điều kiện.** `D(x, y) → [0, 1]`. Đầu vào là *cặp* (điều kiện, đầu ra). Đây là khác biệt then chốt: D phải xét xem `y` có nhất quán với `x` không, chứ không chỉ xem `y` có trông thật không.

**Bộ sinh U-Net.** Encoder-decoder với các kết nối tắt (skip connection) bắc ngang cổ chai. Cực kỳ quan trọng với những tác vụ mà đầu vào và đầu ra chia sẻ cấu trúc mức thấp (cạnh, đường viền). Không có skip, chi tiết tần số cao biến mất.

**Bộ phân biệt PatchGAN.** Thay vì xuất một điểm số thật/giả duy nhất, D xuất một lưới `N×N` trong đó mỗi ô đánh giá một trường tiếp nhận khoảng 70×70 điểm ảnh. Rồi lấy trung bình. Đây là giả định trường ngẫu nhiên Markov: tính chân thực mang tính cục bộ. Huấn luyện nhanh hơn nhiều, ít tham số hơn, đầu ra sắc nét hơn.

**Mất mát.**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

Số hạng L1 (mất mát L1) giúp ổn định huấn luyện và kéo G về phía mục tiêu đã biết. L1 cho cạnh sắc hơn L2 (lấy trung vị, không phải trung bình). `λ = 100` là mặc định của Pix2Pix.

## CycleGAN — khi bạn không có cặp dữ liệu

Pix2Pix cần dữ liệu `(x, y)` theo cặp. CycleGAN (Zhu và cộng sự, 2017) bỏ yêu cầu này với cái giá là thêm một mất mát: mất mát *nhất quán chu trình* (cycle consistency). Hai bộ sinh `G: X → Y` và `F: Y → X`. Huấn luyện chúng sao cho `F(G(x)) ≈ x` và `G(F(y)) ≈ y`. Điều này cho phép bạn dịch ngựa thành ngựa vằn, mùa hè thành mùa đông, mà không cần ví dụ theo cặp.

Năm 2026, dịch ảnh sang ảnh không cặp chủ yếu làm bằng diffusion (ControlNet, IP-Adapter) chứ không phải CycleGAN, nhưng ý tưởng nhất quán chu trình vẫn sống trong gần như mọi bài báo về thích ứng miền không cặp.

```figure
gx-patchgan
```

## Tự tay làm

`code/main.py` cài đặt một GAN có điều kiện tí hon trên dữ liệu 1-D. Điều kiện `c` là một nhãn lớp (0 hoặc 1). Nhiệm vụ: tạo ra một mẫu từ phân phối có điều kiện ứng với lớp đã cho.

### Bước 1: nối điều kiện vào đầu vào của cả G và D

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

One-hot là cách đơn giản nhất. Các mô hình lớn hơn dùng embedding học được, điều biến FiLM, hoặc cross-attention.

### Bước 2: huấn luyện có điều kiện

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

Bộ sinh phải khớp phân phối thật *ứng với điều kiện đã cho*, chứ không phải phân phối biên.

### Bước 3: kiểm tra đầu ra theo từng lớp

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## Cạm bẫy

- **Điều kiện bị bỏ qua.** G học cách lấy biên, D không bao giờ phạt vì tín hiệu điều kiện yếu. Cách sửa: đưa điều kiện vào D mạnh hơn (ở lớp sớm, không chỉ lớp cuối), dùng projection discriminator (Miyato & Koyama 2018).
- **Trọng số L1 quá thấp.** G trôi về những đầu ra trông thật tuỳ ý, không trung thành. Khởi đầu λ≈100 cho các tác vụ kiểu Pix2Pix.
- **Trọng số L1 quá cao.** G tạo ra đầu ra nhoè vì L1 vẫn là một chuẩn L_p. Giảm dần sau khi huấn luyện ổn định.
- **Rò rỉ ground-truth trong D.** Nối `(x, y)` làm đầu vào của D, không chỉ `y`. Không có điều này D không thể kiểm tra tính nhất quán.
- **Sụp đổ mode theo từng lớp.** Mỗi lớp có thể sụp đổ độc lập. Hãy chạy kiểm tra đa dạng theo điều kiện lớp.

## Áp dụng

Hiện trạng năm 2026 của các tác vụ dịch ảnh sang ảnh:

| Tác vụ | Cách tốt nhất |
|------|---------------|
| Phác → ảnh, cùng miền, dữ liệu theo cặp | Pix2Pix / Pix2PixHD (vẫn nhanh, vẫn sắc nét) |
| Phác → ảnh, không cặp | ControlNet với mô hình điều kiện Scribble |
| Phân đoạn ngữ nghĩa → ảnh | SPADE / GauGAN2 hoặc SD + ControlNet-Seg |
| Chuyển phong cách | Diffusion với IP-Adapter hoặc LoRA; các phương pháp GAN đã lỗi thời |
| Độ sâu → ảnh | ControlNet-Depth trên Stable Diffusion |
| Siêu phân giải | Real-ESRGAN (GAN), ESRGAN-Plus, hoặc SD-Upscale (diffusion) |
| Tô màu | ColTran, bộ tô màu dựa trên diffusion, hoặc Pix2Pix-color |
| Ban ngày → ban đêm, mùa, thời tiết | CycleGAN hoặc dựa trên ControlNet |

Pix2Pix vẫn là công cụ đúng khi (a) bạn có hàng nghìn ví dụ theo cặp, (b) tác vụ hẹp và lặp lại được, và (c) bạn cần suy luận nhanh. Trên các tác vụ mở tổng quát, diffusion thắng.

## Đóng gói

Lưu `outputs/skill-img2img-chooser.md`. Skill nhận mô tả tác vụ, mức sẵn có của dữ liệu (theo cặp hay không cặp, N mẫu), và ngân sách độ trễ/chất lượng, rồi xuất ra: hướng tiếp cận (Pix2Pix, CycleGAN, biến thể ControlNet, SDXL + IP-Adapter), yêu cầu dữ liệu huấn luyện, chi phí suy luận, và giao thức đánh giá (LPIPS, FID, đặc thù tác vụ).

## Bài tập

1. **Dễ.** Sửa `code/main.py` để thêm lớp thứ ba. Xác nhận G vẫn ánh xạ nhiễu của từng lớp về đúng mode.
2. **Trung bình.** Thay L1 bằng một mất mát kiểu tri giác trong thiết lập 1-D (ví dụ một D nhỏ đóng băng làm bộ trích đặc trưng). Nó có làm thay đổi độ sắc nét của phân phối có điều kiện không?
3. **Khó.** Phác thảo một CycleGAN trong thiết lập 1-D: hai phân phối, hai bộ sinh, mất mát chu trình. Chứng minh nó học được cách ánh xạ giữa chúng mà không có dữ liệu theo cặp.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Conditional GAN | "GAN có nhãn" | G(z, c), D(x, c). Cả hai mạng đều thấy điều kiện. |
| Pix2Pix | "GAN dịch ảnh sang ảnh" | cGAN theo cặp với G là U-Net và D là PatchGAN + mất mát L1. |
| U-Net | "Encoder-decoder có skip" | Mạng tích chập đối xứng; skip giữ lại tần số cao. |
| PatchGAN | "Bộ phân loại chân thực cục bộ" | D xuất điểm số theo từng mảng thay vì điểm số toàn cục. |
| CycleGAN | "Dịch ảnh không cặp" | Hai G + mất mát nhất quán chu trình; không cần dữ liệu theo cặp. |
| SPADE | "GauGAN" | Chuẩn hoá các activation trung gian bằng bản đồ ngữ nghĩa; phân đoạn sang ảnh. |
| FiLM | "Điều biến tuyến tính theo đặc trưng" | Phép biến đổi affine theo từng đặc trưng sinh từ điều kiện; cách đưa điều kiện rẻ. |

## Ghi chú triển khai: Pix2Pix như một baseline bị ràng buộc độ trễ

Khi bạn có dữ liệu theo cặp và một tác vụ hẹp (phác → render, bản đồ ngữ nghĩa → ảnh, ngày → đêm), suy luận một lần của Pix2Pix nhanh hơn diffusion một bậc về độ trễ. So sánh trong triển khai thường là:

| Đường đi | Số bước | Độ trễ điển hình ở 512² trên một L4 |
|------|-------|----------------------------------------|
| Pix2Pix (U-Net forward) | 1 | ~30 ms |
| SD-Inpaint hoặc SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix thắng về thông lượng trong các batch tĩnh (mọi yêu cầu đều cùng FLOPs). Diffusion thắng về chất lượng và khả năng tổng quát hoá. Cách chơi hiện đại thường là đóng gói một mô hình chưng cất kiểu Pix2Pix cho tác vụ hẹp và một diffusion dự phòng cho các đầu vào hiếm.

## Đọc thêm

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — bài báo cGAN.
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix.
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN.
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD.
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN.
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — projection D.

---

[← Bài trước: GAN](../../03-gans-generator-discriminator/docs/vi.md) · [Bài sau: StyleGAN →](../../05-stylegan/docs/vi.md)
