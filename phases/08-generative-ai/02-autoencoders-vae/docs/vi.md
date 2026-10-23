# Autoencoder (bộ tự mã hoá) & Variational Autoencoder (VAE)

> Một autoencoder thuần nén rồi tái tạo. Nó học thuộc. Nó không sinh ra cái mới. Thêm một mẹo — ép bộ mã trông giống phân phối Gaussian — và bạn có một bộ lấy mẫu. Chính mẹo đó, phép tái tham số hoá (reparameterization trick) `z = μ + σ·ε`, là lý do mọi mô hình ảnh latent-diffusion và flow-matching bạn dùng trong năm 2026 đều có một VAE ở đầu vào.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 3, Bài 02 (Lan truyền ngược), Giai đoạn 3, Bài 07 (CNN), Giai đoạn 8, Bài 01 (Phân loại)
**Time:** ~75 phút

## Vấn đề

Nén một chữ số MNIST 784 pixel thành một bộ mã 16 con số, rồi tái tạo lại. Một autoencoder thuần sẽ đạt MSE tái tạo rất tốt nhưng không gian mã lại là một mớ hỗn độn gồ ghề. Chọn ngẫu nhiên một điểm trong không gian mã, giải mã nó, bạn nhận được nhiễu. Nó không có bộ lấy mẫu. Nó chỉ là một mô hình nén được khoác áo hào nhoáng.

Thứ bạn thật sự muốn là: (a) không gian mã là một phân phối sạch, mượt mà bạn có thể lấy mẫu từ đó — ví dụ một Gaussian đẳng hướng `N(0, I)`, (b) giải mã bất kỳ mẫu nào cũng cho ra một chữ số hợp lý, và (c) encoder và decoder vẫn nén tốt. Ba mục tiêu, một kiến trúc, một hàm mất mát.

VAE của Kingma năm 2013 giải quyết bài này bằng cách huấn luyện encoder xuất ra một *phân phối* `q(z|x) = N(μ(x), σ(x)²)`, kéo phân phối đó về phía prior (phân phối tiên nghiệm) `N(0, I)` thông qua một hình phạt KL divergence (phân kỳ KL), rồi lấy mẫu `z` từ `q(z|x)` trước khi giải mã. Khi suy luận, bỏ encoder đi, lấy mẫu `z ~ N(0, I)`, giải mã. Chính hình phạt KL là thứ buộc không gian mã phải có cấu trúc.

Năm 2026, VAE hiếm khi được triển khai độc lập — chúng đã bị diffusion vượt mặt về chất lượng ảnh thô — nhưng chúng là encoder được chọn cho mọi mô hình latent-diffusion (SD 1/2/XL/3, Flux, AudioCraft). Học VAE là bạn học lớp đầu tiên vô hình của mọi pipeline ảnh mà bạn dùng.

## Khái niệm

![Autoencoder vs VAE: thủ thuật tái tham số hoá](../assets/vae.svg)

**Autoencoder.** `z = encoder(x)`, `x̂ = decoder(z)`, loss = `||x - x̂||²`. Không gian mã không có cấu trúc.

**Encoder của VAE.** Xuất ra hai vector: `μ(x)` và `log σ²(x)`. Hai vector này định nghĩa `q(z|x) = N(μ, diag(σ²))`.

**Thủ thuật tái tham số hoá.** Lấy mẫu từ `q(z|x)` là phép không khả vi. Hãy viết lại mẫu dưới dạng `z = μ + σ·ε` với `ε ~ N(0, I)`. Giờ `z` là một hàm tất định của `(μ, σ)` cộng thêm nhiễu không phải tham số — gradient chảy ngược qua `μ` và `σ`.

**Hàm mất mát.** Evidence Lower BOund (ELBO), gồm hai số hạng:

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

Tái tạo (mất mát tái tạo) kéo `x̂` về phía `x`. KL kéo `q(z|x)` về phía prior. Hai bên đánh đổi nhau. β nhỏ (<1) = mẫu sắc nét hơn, không gian mã kém Gaussian hơn. β lớn (>1) = không gian mã sạch hơn, mẫu mờ hơn. β-VAE (Higgins 2017) làm núm vặn này nổi tiếng và khởi động hướng nghiên cứu disentanglement.

**Lấy mẫu.** Khi suy luận: rút `z ~ N(0, I)`, cho đi qua decoder. Một lượt lan truyền xuôi — không lấy mẫu lặp như diffusion.

```figure
vae-latent-grid
```

## Tự tay làm

`code/main.py` cài một VAE tí hon không dùng numpy hay torch. Đầu vào là dữ liệu tổng hợp 8 chiều rút ra từ một hỗn hợp Gaussian 2 thành phần trong không gian 8 chiều. Encoder và decoder đều là MLP một lớp ẩn. Ta tự cài hàm kích hoạt tanh, lượt lan truyền xuôi, hàm mất mát, và cả lượt lan truyền ngược viết tay. Không phải để chạy production — chỉ để học.

### Bước 1: lan truyền xuôi qua encoder

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

Dùng `log σ²` thay cho `σ` để đầu ra của mạng không bị ràng buộc (lấy softplus của σ là một cái bẫy — gradient chết khi σ ≈ 0).

### Bước 2: tái tham số hoá và giải mã

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### Bước 3: ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

KL có dạng đóng chính xác vì cả hai phân phối đều là Gaussian. Đừng tích phân bằng số. Đến năm 2026 vẫn có người giao code ước lượng KL bằng Monte Carlo — chậm gấp 3 lần mà chẳng để làm gì.

### Bước 4: sinh mẫu

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

Đó chính là mô hình sinh. Năm dòng.

## Cạm bẫy

- **Posterior collapse (sụp đổ hậu nghiệm).** Số hạng KL đẩy `q(z|x) → N(0, I)` mạnh tới mức `z` không còn mang thông tin gì về `x`. Cách sửa: β-annealing (bắt đầu β=0, tăng dần tới 1), free bits, hoặc bỏ KL trên các chiều không hoạt động.
- **Mẫu bị mờ.** Hợp lý của decoder Gaussian ngầm định việc tái tạo theo MSE, mà MSE là tối ưu Bayes cho chuẩn L2 (tức là giá trị trung bình) — trung bình của một tập chữ số hợp lý là một chữ số nhoè. Cách sửa: dùng decoder rời rạc (VQ-VAE, NVAE), hoặc chỉ dùng VAE làm encoder rồi chồng diffusion lên không gian ẩn (đây chính là cách Stable Diffusion làm).
- **β quá lớn, quá sớm.** Xem lại posterior collapse. Hãy bắt đầu ở β≈0.01 rồi tăng dần.
- **Chiều ẩn quá nhỏ.** 16 chiều là đủ cho MNIST, 256 chiều cho ImageNet 256², 2048 chiều cho ImageNet 1024². VAE của Stable Diffusion nén 512×512×3 → 64×64×4 (hệ số giảm mẫu 32 lần theo diện tích không gian, 32 lần theo kênh).

## Áp dụng

Bộ công cụ VAE năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Encoder ảnh sang không gian ẩn cho diffusion | Stable Diffusion VAE (`sd-vae-ft-ema`) hoặc Flux VAE |
| Encoder âm thanh sang không gian ẩn | Encodec (Meta), SoundStream, hoặc DAC (Descript) |
| Không gian ẩn cho video | Các patch không-thời gian của Sora, Latte VAE, WAN VAE |
| Học biểu diễn tách rời (disentangled) | β-VAE, FactorVAE, TCVAE |
| Không gian ẩn rời rạc (để mô hình hoá bằng transformer) | VQ-VAE, RVQ (ResidualVQ) |
| Không gian ẩn liên tục để sinh mẫu | VAE thuần, rồi đặt một mô hình flow/diffusion làm điều kiện trong không gian ẩn đó |

Một mô hình latent-diffusion chính là một VAE với một mô hình diffusion sống chen giữa encoder và decoder. VAE lo phần nén thô, mô hình diffusion gánh phần nặng. Cùng một khuôn mẫu đó áp cho video (VAE + video-diffusion DiT) và âm thanh (Encodec + transformer MusicGen).

## Đóng gói

Lưu `outputs/skill-vae-trainer.md`.

Skill nhận vào: hồ sơ dataset + mục tiêu số chiều ẩn + mục đích phía sau (tái tạo, sinh mẫu, hay làm đầu vào cho latent-diffusion) và xuất ra: lựa chọn kiến trúc (thuần/β/VQ/RVQ), lịch β, số chiều ẩn, hợp lý của decoder (Gaussian hay categorical), và kế hoạch đánh giá (MSE tái tạo, KL theo từng chiều, khoảng cách Fréchet giữa `q(z|x)` và `N(0, I)`).

## Bài tập

1. **Dễ.** Đổi `β` trong `code/main.py` thành `0.01`, `0.1`, `1.0`, `5.0`. Ghi lại MSE tái tạo và KL cuối cùng. Với dữ liệu tổng hợp của bạn, β nào là tốt nhất theo nghĩa Pareto?
2. **Trung bình.** Thay hợp lý của decoder Gaussian bằng hợp lý Bernoulli (mất mát cross-entropy). So sánh chất lượng mẫu trên một phiên bản nhị phân hoá của chính dữ liệu tổng hợp đó.
3. **Khó.** Mở rộng `code/main.py` thành một VQ-VAE mini: thay `z` liên tục bằng phép tra cứu láng giềng gần nhất trong một codebook gồm K=32 mục. So sánh MSE tái tạo và báo cáo xem có bao nhiêu mục của codebook được dùng tới (codebook collapse là chuyện có thật).

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Autoencoder | Mạng mã hoá-giải mã | `x → z → x̂`, học MSE. Không sinh mẫu. |
| VAE | AE có bộ lấy mẫu | Encoder xuất ra một phân phối, hình phạt KL tạo hình dạng cho không gian mã. |
| ELBO | Cận dưới bằng chứng | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`; chặt khi `q = p(z\|x)`. |
| Reparameterization | `z = μ + σ·ε` | Viết lại nút ngẫu nhiên thành phần tất định cộng nhiễu thuần. Nhờ vậy lan truyền ngược xuyên qua phép lấy mẫu được. |
| Prior | `p(z)` | Phân phối đích cho biến ẩn, thường là `N(0, I)`. |
| Posterior collapse | "Số hạng KL thắng" | Encoder bỏ qua `x`, xuất ra đúng prior; decoder buộc phải bịa ra. |
| β-VAE | Trọng số KL chỉnh được | `loss = recon + β·KL`. β cao hơn = tách rời hơn nhưng mờ hơn. |
| VQ-VAE | Biến ẩn rời rạc | Thay `z` liên tục bằng vector gần nhất trong codebook; nhờ đó mô hình hoá được bằng transformer. |

## Ghi chú vận hành: VAE là đường nóng nhất trong một máy chủ diffusion

Trong một pipeline Stable Diffusion / Flux / SD3, VAE được gọi hai lần cho mỗi yêu cầu — một lần để mã hoá (nếu làm img2img / inpainting) và một lần để giải mã. Ở 1024², lượt decoder thường là đỉnh bộ nhớ kích hoạt lớn nhất trong toàn pipeline, vì nó tăng mẫu từ biến ẩn `128×128×16` trở lại `1024×1024×3`. Hai hệ quả thực tế:

- **Cắt lát hoặc chia ô lượt giải mã.** `diffusers` cung cấp `pipe.vae.enable_slicing()` và `pipe.vae.enable_tiling()`. Chia ô đánh đổi một vết đường may nhỏ để lấy bộ nhớ `O(tile²)` thay vì `O(H·W)`. Bắt buộc phải có với 1024²+ trên GPU phổ thông.
- **Decoder bf16, số học fp32 cho lần resize cuối.** VAE của SD 1.x được phát hành ở fp32 và *âm thầm sinh ra NaN* khi bị ép sang fp16 ở 1024²+. SDXL phát hành `madebyollin/sdxl-vae-fp16-fix` — luôn ưu tiên biến thể fp16-fix hoặc dùng bf16.

## Đọc thêm

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — bài báo VAE.
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — β-VAE tách rời biểu diễn.
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE.
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — VAE cho ảnh ở mức state-of-the-art.
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion; VAE làm encoder.
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec, chuẩn mực VAE cho âm thanh.

---

[← Bài trước: Mô hình sinh](../../01-generative-models-taxonomy-history/docs/vi.md) · [Bài sau: GAN →](../../03-gans-generator-discriminator/docs/vi.md)
