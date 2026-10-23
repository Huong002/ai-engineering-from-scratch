# Mô hình khuếch tán — DDPM từ đầu

> Ho, Jain, Abbeel (2020) đưa cho cả lĩnh vực một công thức khó bỏ. Phá huỷ dữ liệu bằng nhiễu qua một nghìn bước nhỏ. Huấn luyện một mạng nơ-ron dự đoán nhiễu. Đảo ngược quá trình khi suy luận. Ngày nay mọi mô hình hình ảnh, video, 3D và âm nhạc chủ lưu đều chạy trên vòng lặp này, có thể kèm flow matching hoặc các mẹo consistency ở trên.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 3, Bài 02 (Lan truyền ngược), Giai đoạn 8, Bài 02 (VAE)
**Time:** ~75 phút

## Vấn đề

Bạn muốn một bộ lấy mẫu cho `p_data(x)`. GAN chơi một trò minimax thường phân kỳ. VAE tạo mẫu mờ từ decoder Gaussian. Thứ bạn thực sự muốn là một mục tiêu huấn luyện (a) một loss ổn định duy nhất (không điểm yên, không minimax), (b) là cận dưới của `log p(x)` (nên bạn có likelihood), và (c) mẫu đạt chất lượng SOTA.

Sohl-Dickstein et al. (2015) đã có câu trả lời lý thuyết: định nghĩa một chuỗi Markov (Markov chain) `q(x_t | x_{t-1})` thêm dần nhiễu Gaussian, và huấn luyện một chuỗi ngược `p_θ(x_{t-1} | x_t)` để khử nhiễu (denoising). Ho, Jain, Abbeel (2020) chỉ ra loss có thể rút gọn thành một dòng — dự đoán nhiễu — và dọn lại phần toán. Năm 2020 đây là một điều tò mò. Năm 2021 nó tạo ra mẫu tốt nhất. Năm 2022 nó thành Stable Diffusion. Năm 2026 nó là nền tảng.

## Khái niệm

![DDPM: nhiễu thuận, khử nhiễu ngược](../assets/ddpm.svg)

**Quá trình thuận (forward process) `q`.** Thêm nhiễu Gaussian trong `T` bước nhỏ. Dạng đóng — lý do toán còn giải được — là bước tích luỹ cũng là Gaussian:

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

trong đó `α̅_t = ∏_{s=1..t} (1 - β_s)` với một lịch nhiễu (noise schedule) `β_t`. Chọn `β_t` từ 1e-4 đến 0.02 tuyến tính qua T=1000 bước và `x_T` xấp xỉ `N(0, I)`.

**Quá trình ngược (reverse process) `p_θ`.** Học một mạng nơ-ron `ε_θ(x_t, t)` dự đoán nhiễu đã được thêm vào. Cho `x_t`, khử nhiễu bằng:

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

trong đó `σ_t` là `sqrt(β_t)` hoặc một phương sai học được. Biểu thức trông xấu nhưng chỉ là đại số — giải ra `x_{t-1}` từ posterior `q(x_{t-1} | x_t, x_0)` và thay `x_0` bằng ước lượng từ dự đoán nhiễu.

**Loss huấn luyện.**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

Lấy `x_0` từ dữ liệu, chọn ngẫu nhiên `t`, lấy mẫu `ε ~ N(0, I)`, tính `x_t` nhiễu trong một lần bằng dạng đóng, rồi hồi quy trên nhiễu. Một loss, không minimax, không KL, không mẹo tái tham số hoá.

**Lấy mẫu (sampling).** Bắt đầu `x_T ~ N(0, I)`. Lặp bước ngược từ `t = T` về `1`. Xong.

## Vì sao nó hiệu quả

Ba trực giác:

1. **Khử nhiễu thì dễ; sinh thì khó.** Ở `t=T`, dữ liệu là nhiễu thuần — mạng phải giải một bài toán tầm thường. Ở `t=0`, mạng chỉ cần dọn vài pixel. Ở `t` trung gian, bài toán khó nhưng mạng có nhiều gradient chảy qua cùng bộ trọng số từ mọi mức nhiễu.

2. **Score matching trá hình.** Vincent (2011) chứng minh rằng dự đoán nhiễu tương đương với ước lượng `∇_x log q(x_t | x_0)`, tức *score*. SDE ngược dùng score này để đi ngược gradient mật độ — một bước đi ngẫu nhiên có dẫn dắt về vùng xác suất cao.

3. **ELBO (cận dưới biến phân) rút gọn thành MSE đơn giản.** Cận dưới biến phân đầy đủ có một số hạng KL mỗi bước thời gian (timestep). Với cách tham số hoá của DDPM, những số hạng KL đó rút gọn thành MSE trên dự đoán nhiễu với hệ số cụ thể; Ho bỏ hệ số (gọi là loss "simple") và chất lượng lại *tốt lên*.

```figure
diffusion-denoise
```

## Tự tay làm

`code/main.py` cài một DDPM 1-D. Dữ liệu là hỗn hợp hai đỉnh. "Mạng" là một MLP nhỏ nhận `(x_t, t)` và xuất nhiễu dự đoán. Huấn luyện là loss một dòng. Lấy mẫu lặp chuỗi ngược.

### Bước 1: lịch thuận (dạng đóng)

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### Bước 2: lấy mẫu `x_t` trong một lần

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### Bước 3: một bước huấn luyện

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### Bước 4: lấy mẫu ngược

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

Với bài 1-D có 40 bước thời gian và MLP 24 đơn vị, nó học được hỗn hợp hai đỉnh trong ~200 epoch.

## Điều kiện hoá thời gian

Mạng cần biết nó đang khử nhiễu ở bước thời gian nào. Hai cách chuẩn:

- **Embedding sin.** Như positional encoding của Transformer. `embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`. Cho qua một MLP, broadcast vào mạng.
- **Điều kiện hoá FiLM / group-norm.** Chiếu embedding thành scale/bias theo kênh (FiLM) ở mỗi block.

Code đồ chơi của chúng ta dùng sin → concat. U-Net thực tế dùng FiLM.

## Cạm bẫy

- **Lịch nhiễu quan trọng rất nhiều.** `β` tuyến tính là mặc định của DDPM nhưng lịch cosine (Nichol & Dhariwal, 2021) cho FID tốt hơn với cùng lượng tính toán. Đổi lịch nếu chất lượng chững lại.
- **Embedding bước thời gian rất mong manh.** Truyền thẳng `t` dạng float chạy được cho 1-D đồ chơi nhưng hỏng với ảnh; luôn dùng embedding tử tế.
- **V-prediction so với ε-prediction.** Với những chế độ hẹp (t rất nhỏ hoặc rất lớn), `ε` có tỷ lệ tín hiệu trên nhiễu kém. V-prediction (`v = α·ε - σ·x`) ổn định hơn; SDXL, SD3 và Flux dùng nó.
- **Classifier-free guidance.** Khi suy luận, tính cả `ε` có điều kiện và không điều kiện, rồi `ε_cfg = (1 + w) · ε_cond - w · ε_uncond` với `w ≈ 3-7`. Nội dung ở Bài 08.
- **1000 bước là nhiều.** Production dùng DDIM (20-50 bước), DPM-Solver (10-20 bước), hoặc distillation (1-4 bước). Xem Bài 12.

## Áp dụng

| Vai trò | Stack điển hình năm 2026 |
|------|-----------------------|
| Diffusion không gian pixel ảnh (nhỏ, đồ chơi) | DDPM + U-Net |
| Diffusion latent cho ảnh | VAE encoder + U-Net hoặc DiT (Bài 07) |
| Diffusion latent cho video | DiT không gian-thời gian (Sora, Veo, WAN) |
| Diffusion latent cho âm thanh | Encodec + diffusion transformer |
| Khoa học (phân tử, protein, vật lý) | Diffusion đẳng biến (EDM, RFdiffusion, AlphaFold3) |

Diffusion là xương sống sinh mẫu phổ quát. Flow matching (Bài 13) là đối thủ 2024-2026 thường thắng về tốc độ suy luận với cùng chất lượng.

## Đóng gói

Lưu `outputs/skill-diffusion-trainer.md`. Skill nhận một dataset + ngân sách tính toán và xuất ra: lịch nhiễu (linear/cosine/sigmoid), mục tiêu dự đoán (ε/v/x), số bước, guidance scale, họ sampler, và một giao thức đánh giá.

## Bài tập

1. **Dễ.** Đổi T từ 40 thành 10 trong `code/main.py`. Chất lượng mẫu (histogram trực quan của đầu ra) giảm thế nào? Ở T nào thì cấu trúc hai đỉnh sụp đổ?
2. **Trung bình.** Chuyển từ ε-prediction sang v-prediction. Tự suy lại bước ngược. So sánh chất lượng mẫu cuối.
3. **Khó.** Thêm classifier-free guidance. Điều kiện theo nhãn lớp `c ∈ {0, 1}`, bỏ nhãn 10% thời gian khi huấn luyện, và khi lấy mẫu dùng `ε = (1+w)·ε_cond - w·ε_uncond`. Đo tỷ lệ trúng đỉnh có điều kiện tại `w = 0, 1, 3, 7`.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Forward process | "Thêm nhiễu" | Chuỗi Markov cố định `q(x_t \| x_{t-1})` phá huỷ dữ liệu. |
| Reverse process | "Khử nhiễu" | Chuỗi học được `p_θ(x_{t-1} \| x_t)` tái dựng dữ liệu. |
| β schedule | "Nấc thang nhiễu" | Phương sai mỗi bước; tuyến tính, cosine hoặc sigmoid. |
| α̅ | "Alpha bar" | Tích luỹ thừa `∏(1 - β)`; cho `x_t` dạng đóng từ `x_0`. |
| Simple loss | "MSE trên nhiễu" | `\|\|ε - ε_θ(x_t, t)\|\|²`; mọi suy diễn biến phân đều quy về đây. |
| ε-prediction | "Dự đoán nhiễu" | Đầu ra là nhiễu được thêm vào; DDPM chuẩn. |
| V-prediction | "Dự đoán vận tốc" | Đầu ra là `α·ε - σ·x`; điều kiện hoá theo t tốt hơn. |
| DDPM | "Bài báo" | Ho et al. 2020; β tuyến tính, 1000 bước, U-Net. |
| DDIM | "Bộ lấy mẫu tất định" | Bộ lấy mẫu phi Markov, 20-50 bước, cùng mục tiêu huấn luyện. |
| Classifier-free guidance | "CFG" | Trộn dự đoán nhiễu có điều kiện và không điều kiện để khuếch đại điều kiện hoá. |

## Ghi chú production: suy luận diffusion là bài toán số bước

Bài báo DDPM chạy T=1000 bước ngược. Không ai ship thứ đó ra production. Mọi stack suy luận thực tế chọn một trong ba chiến lược — và mỗi cái ánh xạ gọn vào cách đặt vấn đề "độ trễ đến từ đâu":

1. **Sampler nhanh hơn, cùng mô hình.** DDIM (20-50 bước), DPM-Solver++ (10-20), UniPC (8-16). Thay thế trực tiếp vòng lặp ngược; trọng số `ε_θ` đã huấn luyện không đổi. Cắt độ trễ 20-50×.
2. **Distillation.** Huấn luyện một student khớp teacher trong ít bước hơn: Progressive Distillation (2 → 1), Consistency Models (tuỳ ý → 1-4), LCM, SDXL-Turbo, SD3-Turbo. Cắt thêm 5-10× độ trễ, cần huấn luyện lại.
3. **Cache và biên dịch.** `torch.compile(unet, mode="reduce-overhead")`, các backend diffusion của TensorRT-LLM, attention `xformers`/SDPA, trọng số bf16. Cắt độ trễ mỗi bước ~2×. Cộng dồn với (1) và (2).

Với một server diffusion production, câu chuyện ngân sách giống như tài liệu production mô tả cho LLM: độ trễ là `num_steps × step_cost + VAE_decode`, thông lượng là `batch_size × (num_steps × step_cost)^-1`. TTFT nhỏ (một bước); tương đương TPOT là toàn bộ thời gian phản hồi vì sinh ảnh là "tất cả cùng lúc" từ góc nhìn người dùng.

## Đọc thêm

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — bài báo diffusion, đi trước thời đại.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM.
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM, ít bước hơn.
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — lịch cosine, phương sai học được.
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — classifier guidance.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — ký pháp thống nhất, công thức gọn nhất.

---

[← Bài trước: StyleGAN](../../05-stylegan/docs/vi.md) · [Bài sau: Latent diffusion (khuếch tán trong không gian ẩn) & Stable Diffusion →](../../07-latent-diffusion-stable-diffusion/docs/vi.md)
