# GAN — Bộ sinh vs Bộ phân biệt

> Chiêu của Goodfellow năm 2014 với GAN (mạng đối kháng sinh tạo) là bỏ qua hoàn toàn mật độ xác suất. Hai mạng. Một mạng tạo đồ giả. Một mạng bắt đồ giả. Chúng đánh nhau cho tới khi đồ giả không phân biệt được với đồ thật. Đáng lẽ nó không hoạt động. Nó thường không hoạt động. Khi nó hoạt động, các mẫu sinh ra vẫn sắc nét nhất trong tài liệu với những miền hẹp.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 3, Bài 02 (Lan truyền ngược), Giai đoạn 3, Bài 08 (Bộ tối ưu), Giai đoạn 8, Bài 02 (VAE)
**Time:** ~75 phút

## Vấn đề

VAE sinh mẫu mờ vì mất mát giải mã MSE của chúng là Bayes-tối ưu cho ảnh *trung bình* — và trung bình của nhiều chữ số hợp lý là một chữ số nhoè. Bạn muốn một mất mát thưởng cho tính *hợp lý*, chứ không phải sự gần gũi từng pixel với bất kỳ một mục tiêu nào. Không có công thức đóng cho tính hợp lý. Bạn phải học nó.

Ý tưởng của Goodfellow: huấn luyện một bộ phân loại `D(x)` để phân biệt ảnh thật với đồ giả. Huấn luyện một bộ sinh (generator) `G(z)` để lừa `D`. Tín hiệu mất mát cho `G` là bất cứ thứ gì `D` hiện cho là làm nên vẻ thật. Tín hiệu này cập nhật khi `G` tiến bộ, đuổi theo một mục tiêu di động. Nếu cả hai mạng hội tụ, `G` đã học được phân phối dữ liệu mà không cần viết ra `log p(x)`.

Đây là huấn luyện đối kháng. Toán học là một trò chơi minimax (cực tiểu-cực đại):

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

Năm 2026 GAN không còn là bộ sinh SOTA (diffusion và flow matching đã lấy vương miện đó). Nhưng StyleGAN 2/3 vẫn là những mô hình khuôn mặt sắc nét nhất từng được phát hành, bộ phân biệt của GAN được dùng làm *mất mát tri giác* trong huấn luyện diffusion, và huấn luyện đối kháng vận hành các phép distillation 1 bước nhanh (SDXL-Turbo, SD3-Turbo, LCM) cho phép bạn triển khai diffusion thời gian thực.

## Khái niệm

![Huấn luyện GAN: bộ sinh và bộ phân biệt trong thế minimax](../assets/gan.svg)

**Bộ sinh `G(z)`.** Ánh xạ một vector nhiễu `z ~ N(0, I)` thành một mẫu `x̂`. Một mạng có hình dạng bộ giải mã (dense hoặc transposed conv).

**Bộ phân biệt `D(x)`.** Ánh xạ một mẫu thành một xác suất vô hướng (hoặc một điểm số). Thật → 1, giả → 0.

**Mất mát.** Hai bước cập nhật xen kẽ:

- **Huấn luyện `D`:** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`. Binary cross-entropy với thật=1, giả=0.
- **Huấn luyện `G`:** `loss_G = -log D(G(z))`. Đây là dạng *không bão hoà* mà Goodfellow đã dùng (dạng gốc `log(1 - D(G(z)))` bão hoà và giết gradient khi `D` tự tin).

**Vòng huấn luyện.** Một bước `D`, một bước `G`. Lặp lại.

**Vì sao nó hoạt động.** Nếu `G` khớp hoàn hảo `p_data`, thì `D` không thể làm tốt hơn mức ngẫu nhiên và cho ra 0.5 ở mọi nơi; `G` không nhận thêm gradient nào nữa. Cân bằng (cân bằng Nash).

**Vì sao nó hỏng.** Sụp đổ chế độ (mode collapse) (`G` tìm ra một chế độ mà `D` không phân loại được rồi sản xuất nó mãi mãi), gradient biến mất (`D` học quá nhanh và `log D` bão hoà), bất ổn huấn luyện (training instability) (learning rate, batch size, bất cứ thứ gì).

## Những biến thể đã làm GAN hoạt động

| Năm | Đổi mới | Cách khắc phục |
|------|------------|-----|
| 2015 | DCGAN | Conv/deconv, batch norm, LeakyReLU — kiến trúc ổn định đầu tiên. |
| 2017 | WGAN, WGAN-GP | Thay BCE bằng khoảng cách Wasserstein + phạt gradient. Khắc phục gradient biến mất. |
| 2017 | Chuẩn hoá phổ | Chặn Lipschitz cho bộ phân biệt. Vẫn được dùng trong bộ phân biệt năm 2026. |
| 2018 | Progressive GAN | Huấn luyện độ phân giải thấp trước, thêm tầng sau. Kết quả megapixel đầu tiên. |
| 2019 | StyleGAN / StyleGAN2 | Mạng ánh xạ + chuẩn hoá instance thích ứng. SOTA cho ảnh chân thực miền cố định. |
| 2021 | StyleGAN3 | Không alias, đẳng biến với tịnh tiến — vẫn là chuẩn vàng cho khuôn mặt năm 2026. |
| 2022 | StyleGAN-XL | Có điều kiện, nhận biết lớp, quy mô lớn hơn. |
| 2024 | R3GAN | Đổi thương hiệu với regularization mạnh hơn; chạy trên 1024² mà không cần mẹo. |

```figure
gan-minimax
```

## Tự tay làm

`code/main.py` huấn luyện một GAN tí hon trên dữ liệu 1-D: hỗn hợp hai phân phối Gauss. Bộ sinh và bộ phân biệt là MLP một tầng ẩn. Chúng ta tự tay cài đặt forward, backward và vòng lặp minimax. Mục tiêu là thấy hai chế độ hỏng then chốt (sụp đổ chế độ + gradient biến mất) ngay khi chúng xảy ra.

### Bước 1: mất mát không bão hoà

Mất mát Goodfellow nguyên bản `log(1 - D(G(z)))` tiến về 0 khi D phân loại đồ giả của G là giả với độ tự tin cao. Lúc đó gradient cho G gần như bằng không — G không thể tiến bộ. Dạng không bão hoà `-log D(G(z))` có tiệm cận ngược lại: nó nổ ra khi D tự tin, cho G một tín hiệu mạnh.

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### Bước 2: một bước bộ phân biệt cho mỗi bước bộ sinh

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

Đồ giả mới cho G, nếu không gradient sẽ bị cũ.

### Bước 3: coi chừng sụp đổ chế độ

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

Triệu chứng kinh điển: một trong hai chế độ thật không còn được sinh ra nữa. Bộ phân biệt thôi sửa lỗi đó vì nó không bao giờ bị coi là đồ giả.

## Cạm bẫy

- **Bộ phân biệt quá mạnh.** Giảm learning rate của D xuống 2-5 lần, hoặc thêm nhiễu instance/layer. Nếu D đạt độ chính xác >95%, G đã chết.
- **Bộ sinh học thuộc lòng một chế độ.** Thêm nhiễu vào đầu vào D, dùng một tầng minibatch-discriminator, hoặc chuyển sang WGAN-GP.
- **Batch norm rò rỉ thống kê.** Batch thật + batch giả đi qua cùng một tầng BN trộn lẫn thống kê của chúng. Thay vào đó hãy dùng instance norm hoặc spectral norm.
- **Đánh lừa Inception score.** FID và IS rất nhiễu khi số mẫu ít. Dùng ≥10k mẫu khi đánh giá.
- **Lấy mẫu một phát là lời nói dối với tác vụ có điều kiện.** Bạn vẫn cần các scale CFG, mẹo truncation và lấy mẫu lại để có đầu ra dùng được.

## Áp dụng

Bộ công cụ GAN năm 2026:

| Tình huống | Chọn |
|-----------|------|
| Mặt người chân thực, tư thế cố định | StyleGAN3 (sắc nét nhất, nhỏ nhất) |
| Mặt anime / cách điệu | StyleGAN-XL hoặc LoRA của Stable Diffusion |
| Dịch ảnh sang ảnh | Pix2Pix / CycleGAN (Giai đoạn 8 · 04) hoặc ControlNet (Giai đoạn 8 · 08) |
| Text-to-image 1 bước nhanh | Distillation đối kháng của diffusion (SDXL-Turbo, SD3-Turbo) |
| Mất mát tri giác bên trong bộ huấn luyện diffusion | Bộ phân biệt GAN nhỏ trên các mảnh ảnh |
| Bất cứ gì đa phương thức, mở | Đừng — hãy dùng diffusion hoặc flow matching |

GAN sắc nét nhưng hẹp. Khi miền của bạn mở ra — ảnh chụp, prompt văn bản tuỳ ý, video — hãy chuyển sang diffusion. Chiêu đối kháng còn sống như một thành phần (mất mát tri giác, distillation), không phải một bộ sinh độc lập.

## Đóng gói

Lưu `outputs/skill-gan-debugger.md`. Skill nhận một lần chạy GAN đang hỏng (đường cong loss, lưới mẫu, kích thước dataset) và xuất ra danh sách xếp hạng các nguyên nhân khả dĩ, cách sửa một dòng, và quy trình chạy lại.

## Bài tập

1. **Dễ.** Chạy `code/main.py` với thiết lập mặc định. Rồi đặt `D_LR = 5 * G_LR` và chạy lại. Loss của G sụp về một hằng số nhanh đến mức nào?
2. **Trung bình.** Thay mất mát BCE của Goodfellow bằng mất mát WGAN: `loss_D = E[D(fake)] - E[D(real)]`, `loss_G = -E[D(fake)]`, và clip trọng số của D về `[-0.01, 0.01]`. Huấn luyện có ổn định hơn không? So sánh thời gian hội tụ thực tế.
3. **Khó.** Mở rộng ví dụ 1-D sang dữ liệu 2-D (hỗn hợp 8 phân phối Gauss trên một vòng tròn). Theo dõi bộ sinh bắt được bao nhiêu trong 8 chế độ ở các bước 1k, 5k, 10k. Cài đặt minibatch discrimination và đo lại.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Generator | "G" | Mạng biến nhiễu thành mẫu, `G: z → x̂`. |
| Discriminator | "D" | Bộ phân loại `D: x → [0, 1]`, thật vs giả. |
| Minimax | "Trò chơi" | `min_G max_D` của một mục tiêu chung. |
| Non-saturating loss | "Cách sửa" | Dùng `-log D(G(z))` cho G thay vì `log(1 - D(G(z)))`. |
| Mode collapse | "G học thuộc một thứ" | Bộ sinh tạo ra ít đầu ra khác biệt dù dữ liệu đa dạng. |
| WGAN | "Wasserstein" | Thay BCE bằng khoảng cách Earth-Mover + phạt gradient; gradient mượt hơn. |
| Spectral norm | "Mẹo Lipschitz" | Ràng buộc chuẩn trọng số của D để chặn độ dốc; ổn định huấn luyện. |
| StyleGAN | "Cái chạy được" | Mạng ánh xạ + AdaIN; tốt nhất trong loại cho khuôn mặt, vẫn vậy năm 2026. |

## Ghi chú sản xuất: suy luận một bước là lợi thế bền lâu của GAN

GAN không còn thắng về chất lượng mẫu cho sinh miền mở, nhưng chúng vẫn thắng về chi phí suy luận. Trong từ vựng của tài liệu suy luận sản xuất, một GAN có:

- **Không có giai đoạn prefill, không có giai đoạn decode.** Một lượt forward `G(z)` duy nhất. TTFT ≈ tổng độ trễ.
- **Không có áp lực KV-cache.** Trạng thái duy nhất là trọng số. Batch size bị chặn bởi bộ nhớ activation, không phải cache.
- **Continuous batching dễ ợt.** Vì mọi request tốn cùng một lượng FLOPs cố định, một batch tĩnh ở mức lấp đầy mục tiêu của server thường là tối ưu. Không cần scheduler trong luồng.

Đây là lý do distillation GAN (SDXL-Turbo, SD3-Turbo, ADD, LCM) là kỹ thuật thống trị cho text-to-image nhanh năm 2026: nó thu gọn một pipeline diffusion 20-50 bước thành 1-4 lượt forward kiểu GAN mà vẫn giữ phân phối của một base diffusion. Mất mát đối kháng tồn tại như một nút xoay lúc huấn luyện để biến bộ sinh chậm thành bộ sinh nhanh.

## Đọc thêm

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — bài báo GAN gốc.
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — kiến trúc ổn định đầu tiên.
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN.
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — SN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo.

---

[← Bài trước: Autoencoder (bộ tự mã hoá) & Variational Autoencoder (VAE)](../../02-autoencoders-vae/docs/vi.md) · [Bài sau: GAN có điều kiện & Pix2Pix →](../../04-conditional-gans-pix2pix/docs/vi.md)
