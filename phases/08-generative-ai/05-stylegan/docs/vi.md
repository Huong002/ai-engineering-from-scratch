# StyleGAN

> Hầu hết bộ sinh trộn `z` vào mọi lớp cùng một lúc. StyleGAN tách nó ra: trước tiên ánh xạ `z` sang một `w` trung gian, rồi *chèn* `w` ở mọi mức độ phân giải thông qua AdaIN. Chỉ thay đổi đó đã gỡ rối không gian latent và biến ảnh mặt chân thực như ảnh chụp thành bài toán đã giải trong bảy năm liền.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 8, Bài 03 (GAN), Giai đoạn 4, Bài 08 (Chuẩn hoá), Giai đoạn 3, Bài 07 (CNN)
**Time:** ~45 phút

## Vấn đề

Một DCGAN ánh xạ `z` thành ảnh qua một chồng các phép tích chập chuyển vị. Vấn đề: `z` điều khiển mọi thứ — tư thế, ánh sáng, danh tính, bối cảnh — chồng chéo lẫn nhau. Dịch theo một trục của `z`, cả bốn thứ cùng đổi. Bạn không thể yêu cầu mô hình cho ra "cùng một người, tư thế khác" vì biểu diễn không tách rời theo cách đó.

Karras et al. (2019, NVIDIA) đề xuất: ngừng đưa `z` trực tiếp vào các lớp conv. Thay vào đó, đưa một tensor hằng `4×4×512` làm đầu vào mạng. Học một MLP 8 lớp ánh xạ `z ∈ Z → w ∈ W`. Chèn `w` ở mọi độ phân giải qua *adaptive instance normalization* (AdaIN): chuẩn hoá từng feature map của conv, rồi scale và shift bằng các phép chiếu affine của `w`. Thêm noise injection (chèn nhiễu) theo từng lớp cho chi tiết ngẫu nhiên (lỗ chân lông, sợi tóc).

Kết quả: `W` có các trục gần trực giao cho "phong cách bậc cao" (tư thế, danh tính) so với "phong cách mịn" (ánh sáng, màu sắc). Bạn có thể tráo phong cách giữa hai ảnh bằng cách dùng `w` của ảnh A cho các mức độ phân giải thấp và `w` của ảnh B cho các mức cao. Điều này mở ra khả năng chỉnh sửa, phong cách hoá chéo miền, và cả một dòng nghiên cứu "StyleGAN-inversion".

## Khái niệm

![StyleGAN: mạng ánh xạ + AdaIN + nhiễu theo lớp](../assets/stylegan.svg)

**Mapping network (mạng ánh xạ).** `f: Z → W`, một MLP 8 lớp. `Z = N(0, I)^512`. `W` không bị ép phải là Gaussian — nó học được hình dạng thích ứng với dữ liệu.

**Synthesis network.** Bắt đầu từ một hằng số đã học `4×4×512`. Mỗi khối độ phân giải: `upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`. Độ phân giải tăng gấp đôi: 4, 8, 16, 32, 64, 128, 256, 512, 1024.

**AdaIN.**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

trong đó `y_scale` và `y_bias` đến từ các phép chiếu affine của `w`. Chuẩn hoá theo từng feature map, rồi tái tạo phong cách. "Phong cách" ở đây là thống kê bậc một và bậc hai của feature map.

**Per-layer noise.** Nhiễu Gaussian một kênh được cộng vào mỗi feature map, nhân với một hệ số học được theo từng kênh. Nó điều khiển chi tiết ngẫu nhiên mà không ảnh hưởng cấu trúc toàn cục.

**Truncation trick (thủ thuật cắt ngắn).** Khi suy luận, lấy mẫu `z`, tính `w = mapping(z)`, rồi `w' = ŵ + ψ·(w - ŵ)` với `ŵ` là trung bình `w` trên nhiều mẫu. `ψ < 1` đánh đổi đa dạng lấy chất lượng. Gần như mọi demo StyleGAN đều dùng `ψ ≈ 0.7`.

## StyleGAN 1 → 2 → 3

| Phiên bản | Năm | Đổi mới |
|---------|------|------------|
| StyleGAN | 2019 | Mapping network + AdaIN + noise + progressive growing (phát triển dần). |
| StyleGAN2 | 2020 | Weight demodulation thay AdaIN (sửa lỗi giọt nước); kiến trúc skip/residual; path-length regularization. |
| StyleGAN3 | 2021 | Tích chập alias-free + kernel đẳng biến; loại bỏ việc texture dính vào lưới pixel. |
| StyleGAN-XL | 2022 | Có điều kiện theo lớp, 1024², ImageNet. |
| R3GAN | 2024 | Đổi mới với reg mạnh hơn; thu hẹp khoảng cách với diffusion trên FFHQ-1024 với ít hơn 20 lần tham số. |

Năm 2026, StyleGAN3 vẫn là lựa chọn mặc định cho (a) ảnh chân thực miền hẹp ở FPS cao, (b) thích ứng miền ít mẫu (huấn luyện trên bộ dữ liệu mới với 100 ảnh, đóng băng mapping), (c) chỉnh sửa dựa trên inversion (tìm `w` tái tạo một ảnh thật, rồi chỉnh `w` đó). Với text-to-image miền mở, nó không phải công cụ phù hợp — diffusion là.

```figure
gx-stylegan-mapping
```

## Tự tay làm

`code/main.py` cài đặt một "style-GAN lite" đồ chơi trong 1-D: một MLP ánh xạ, một hàm tổng hợp nhận một vector hằng đã học và điều biến nó bằng scale/bias dẫn xuất từ `w`, cùng nhiễu theo từng lớp. Nó cho thấy rằng chèn `w` qua điều biến affine ngang bằng hoặc tốt hơn việc nối `z` vào đầu vào của bộ sinh.

### Bước 1: mạng ánh xạ

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### Bước 2: chuẩn hoá instance thích ứng

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

Scale và bias theo từng feature map đến từ `w` qua một phép chiếu tuyến tính.

### Bước 3: nhiễu theo lớp

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

Sigma theo từng kênh là học được.

## Cạm bẫy

- **Lỗi giọt nước.** StyleGAN 1 tạo ra một giọt nước dạng cục trong feature map vì AdaIN đưa giá trị trung bình về 0. Weight demodulation của StyleGAN 2 sửa điều này bằng cách scale trọng số tích chập thay vì activation.
- **Texture dính.** Texture của StyleGAN 1 và 2 bám theo toạ độ pixel chứ không theo toạ độ vật thể (thấy rõ khi nội suy). Tích chập alias-free của StyleGAN 3 sửa bằng bộ lọc sinc có cửa sổ.
- **Độ phủ mode.** Cắt ngắn `ψ < 0.7` trông sạch nhưng lấy mẫu từ một nón hẹp; dùng `ψ = 1.0` nếu bạn cần đa dạng.
- **Inversion là mất mát.** Đảo ngược một ảnh thật vào `W` thường được làm qua tối ưu hoá hoặc một encoder (e4e, ReStyle, HyperStyle). Kết quả trôi dần qua nhiều vòng lặp.

## Áp dụng

| Trường hợp dùng | Cách tiếp cận |
|----------|----------|
| Mặt người chân thực (anime, sản phẩm, miền hẹp) | StyleGAN3 FFHQ / tinh chỉnh riêng |
| Chỉnh sửa mặt từ một ảnh chụp | e4e inversion + hướng StyleSpace / InterFaceGAN |
| Hoán đổi mặt / tái hiện động tác | StyleGAN + encoder + blending |
| Pipeline avatar | StyleGAN3 với ADA để tinh chỉnh với ít dữ liệu |
| Thích ứng miền từ vài ảnh | Đóng băng mapping network, tinh chỉnh phần tổng hợp |
| Sinh đa phương thức hoặc theo văn bản | Đừng — hãy dùng diffusion |

Với các demo cấp sản phẩm mà câu trả lời là "ảnh mặt người", StyleGAN thắng diffusion về chi phí suy luận (một lượt lan truyền xuôi, <10ms trên 4090) và độ nét ở cùng mức chất lượng.

## Đóng gói

Lưu `outputs/skill-stylegan-inversion.md`. Skill nhận một ảnh thật và xuất ra: phương pháp inversion (e4e / ReStyle / HyperStyle), latent loss dự kiến, ngân sách chỉnh sửa (bạn có thể dịch xa bao nhiêu trong `W` trước khi xuất hiện lỗi), và danh sách các hướng chỉnh sửa đã biết là tốt (tuổi, biểu cảm, tư thế).

## Bài tập

1. **Dễ.** Chạy `code/main.py` với `adain_on=True` và `adain_on=False`. So sánh độ phân tán của đầu ra với latent cố định so với latent bị nhiễu.
2. **Trung bình.** Cài đặt mixing regularization (điều chuẩn trộn phong cách): với một batch huấn luyện, tính `w_a`, `w_b`, áp dụng `w_a` cho nửa đầu quá trình tổng hợp và `w_b` cho nửa sau. Bộ giải mã có học được các phong cách tách rời không?
3. **Khó.** Lấy một mô hình StyleGAN3 FFHQ đã huấn luyện (ffhq-1024.pkl). Tìm hướng `w` điều khiển "cười" bằng cách huấn luyện một SVM trên các mẫu có nhãn; báo cáo có thể đẩy xa đến đâu trước khi danh tính trôi.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Mapping network | "Cái MLP" | `f: Z → W`, 8 lớp, tách hình học latent khỏi thống kê dữ liệu. |
| W space | "Không gian phong cách" | Đầu ra của mạng ánh xạ; gần như tách rời. |
| AdaIN | "Adaptive instance norm" | Chuẩn hoá feature map, rồi scale + shift bằng chiếu `w`. |
| Truncation trick | "Psi" | `w = mean + ψ·(w - mean)`, ψ<1 đánh đổi đa dạng lấy chất lượng. |
| Path-length regularization | "PL reg" | Phạt thay đổi lớn ở ảnh trên mỗi đơn vị thay đổi của `w`; làm `W` mượt hơn. |
| Weight demodulation | "Bản sửa của StyleGAN2" | Chuẩn hoá trọng số conv thay vì activation; diệt lỗi giọt nước. |
| Alias-free | "Chiêu của StyleGAN3" | Bộ lọc sinc có cửa sổ; loại bỏ việc texture dính vào lưới pixel. |
| Inversion | "Tìm w cho ảnh thật" | Tối ưu hoặc mã hoá `x → w` sao cho `G(w) ≈ x`. |

## Ghi chú vận hành: vì sao StyleGAN vẫn được triển khai năm 2026

StyleGAN3 trên một 4090 sinh một mặt FFHQ 1024² trong dưới 10 ms — `num_steps = 1`, không giải mã VAE, không lượt cross-attention. Theo góc nhìn vận hành, đây là mức latency sàn cho bất kỳ bộ sinh ảnh nào. Một pipeline SDXL 50 bước + giải mã VAE ở cùng độ phân giải mất ~3 giây. Đó là khoảng cách **300×**, và với các sản phẩm miền hẹp (dịch vụ avatar, pipeline giấy tờ tuỳ thân, sinh mặt cho kho ảnh) nó thắng về TCO.

Hai hệ quả vận hành:

- **Không scheduler, không batcher.** Batch tĩnh ở mức chiếm dụng mục tiêu là tối ưu. Continuous batching (thiết yếu cho LLM và diffusion) không mang lại lợi ích nào vì mọi request đều tốn cùng số FLOPs.
- **Cắt ngắn `ψ` là núm an toàn.** `ψ < 0.7` lấy mẫu từ một nón hẹp trong miền giá trị của mapping network. Đây là đòn bẩy duy nhất mà lớp serving có đối với phương sai mẫu. Hạ `ψ` khi tải cao, nâng lên cho người dùng trả phí cao.

## Đọc thêm

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e inversion.
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL.
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — modern minimal GAN recipe.

---

[← Bài trước: GAN có điều kiện & Pix2Pix](../../04-conditional-gans-pix2pix/docs/vi.md) · [Bài sau: Mô hình khuếch tán →](../../06-diffusion-ddpm-from-scratch/docs/vi.md)
