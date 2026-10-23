# Flow Matching & Rectified Flows

> Mô hình khuếch tán cần 20-50 bước lấy mẫu vì chúng đi theo một đường cong từ nhiễu tới dữ liệu. Flow matching (khớp luồng) (Lipman et al., 2023) và rectified flow (luồng nắn thẳng) (Liu et al., 2022) huấn luyện các đường thẳng (straight path). Đường càng thẳng thì càng ít bước, suy luận càng nhanh. Stable Diffusion 3, Flux.1 và AudioCraft 2 đều chuyển sang flow matching trong 2024.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 8, Bài 06 (DDPM), Giai đoạn 1 (Giải tích)
**Time:** ~45 phút

## Vấn đề

Quá trình nghịch của DDPM là một bước đi ngẫu nhiên 1000 bước từ `N(0, I)` quay về phân phối dữ liệu. DDIM rút nó xuống còn 20-50 bước tất định. Bạn muốn ít bước hơn — lý tưởng là một. Rào cản là ODE (phương trình vi phân thường) giải quá trình nghịch bị cứng; đường đi thì cong.

Nếu bạn huấn luyện được mô hình sao cho đường từ nhiễu tới dữ liệu là một *đường thẳng*, thì chỉ một bước Euler từ `t=1` tới `t=0` là đủ. Flow matching dựng thẳng điều đó: định nghĩa một phép nội suy đường thẳng từ `x_1 ∼ N(0, I)` tới `x_0 ∼ data`, huấn luyện một vector field (trường vận tốc) `v_θ(x, t)` để khớp đạo hàm theo thời gian của nó, rồi tích phân (integration) lúc suy luận.

Rectified flow (Liu 2022) đi xa hơn: nắn thẳng các đường một cách lặp lại bằng thủ tục reflow, tạo ra một ODE ngày càng gần tuyến tính. Sau hai vòng reflow, bộ lấy mẫu 2 bước đã đạt chất lượng của DDPM 50 bước.

## Khái niệm

![Flow matching: nội suy đường thẳng giữa nhiễu và dữ liệu](../assets/flow-matching.svg)

### Luồng đường thẳng

Định nghĩa:

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

trong đó `x_0 ~ data` và `x_1 ~ N(0, I)`. Đạo hàm theo thời gian dọc đường thẳng này là hằng số:

```
dx_t / dt = x_1 - x_0
```

Định nghĩa một trường vận tốc nơ-ron `v_θ(x_t, t)` và huấn luyện nó khớp đạo hàm này:

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

Đây là loss **conditional flow matching** (Lipman 2023). Huấn luyện không cần mô phỏng: bạn không bao giờ cuộn ODE ra. Chỉ cần lấy mẫu `(x_0, x_1, t)` và hồi quy.

### Lấy mẫu

Lúc suy luận, tích phân trường vận tốc đã học *ngược* theo thời gian:

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

Bắt đầu tại `x_1 ~ N(0, I)`, bước Euler xuống tới `t=0`.

### Rectified flow (Liu 2022)

Luồng đường thẳng có hiệu quả, nhưng các đường học được *không thực sự thẳng* — chúng cong vì nhiều `x_0` có thể ánh xạ tới cùng một `x_1`. Bước reflow của rectified flow:

1. Huấn luyện mô hình luồng v_1 với các cặp ngẫu nhiên.
2. Lấy mẫu N cặp `(x_1, x_0)` bằng cách tích phân v_1 từ `x_1` tới điểm hạ cánh `x_0`.
3. Huấn luyện v_2 trên các cặp ví dụ đó. Vì các cặp giờ đã "ODE-matched", phép nội suy đường thẳng giữa chúng thực sự phẳng hơn.
4. Lặp lại.

Trên thực tế, 2 vòng reflow đưa bạn tới gần tuyến tính, cho phép suy luận 2-4 bước. SDXL-Turbo, SD3-Turbo, LCM đều là mô hình chưng cất từ flow matching.

### Vì sao cách này thắng cho ảnh năm 2024

Ba lý do:

1. **Huấn luyện không mô phỏng** — không cuộn ODE khi huấn luyện, cài đặt rất đơn giản.
2. **Hình học loss tốt hơn** — đường thẳng có tỉ lệ tín hiệu trên nhiễu ổn định, còn ε-loss của DDPM có SNR kém ở hai đầu lịch.
3. **Suy luận nhanh hơn** — 4-8 bước với chất lượng SDXL-Turbo; 1 bước nhờ consistency distillation.

## Flow matching vs DDPM — mối liên hệ chính xác

Flow matching với đường điều kiện Gaussian chính là khuếch tán *với một lịch nhiễu cụ thể*. Chọn lịch `x_t = α(t) x_0 + σ(t) x_1` thì flow matching thu hồi lại khuếch tán đã viết lại theo Stratonovich với `v = α'·x_0 - σ'·x_1`. Hai cách tương đương về đại số với các đường Gaussian.

Điều flow matching thêm vào: sự *rõ ràng* của mục tiêu (một vận tốc thuần), một loss gọn hơn, và quyền tự do thử nghiệm với các phép nội suy phi Gaussian.

```figure
normalizing-flow
```

## Tự tay làm

`code/main.py` cài đặt flow matching 1-D trên hỗn hợp Gauss hai đỉnh. Trường vận tốc `v_θ(x, t)` là một MLP nhỏ huấn luyện với mục tiêu đường thẳng. Lúc suy luận, tích phân 1, 2, 4 và 20 bước Euler rồi so sánh chất lượng mẫu.

### Bước 1: loss huấn luyện

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### Bước 2: suy luận nhiều bước

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### Bước 3: so sánh số bước

Hãy kỳ vọng bộ lấy mẫu 4 bước đã đạt chất lượng của 20 bước — điều này rất quan trọng cho độ trễ.

## Cạm bẫy

- **Tham số hoá thời gian.** Flow matching dùng `t ∈ [0, 1]` với `t=0` tại dữ liệu, `t=1` tại nhiễu. DDPM dùng `t ∈ [0, T]` với `t=0` tại dữ liệu, `t=T` tại nhiễu. Cùng hướng, khác thang. Các bài báo hay làm sai chỗ này.
- **Chọn lịch.** Đường thẳng của rectified flow là lịch flow matching "mặc định", nhưng bạn có thể dùng lấy mẫu t theo cosine hoặc logit-normal (SD3 làm vậy) để bao phủ thang tốt hơn.
- **Chi phí reflow.** Sinh tập dữ liệu cặp cho reflow là một lượt suy luận đầy đủ cho mỗi mẫu. Chỉ reflow khi bạn thực sự cần suy luận 1-2 bước.
- **Classifier-free guidance vẫn áp dụng.** Chỉ cần đổi ε thành v trong tổ hợp tuyến tính: `v_cfg = (1+w) v_cond - w v_uncond`.

## Áp dụng

| Trường hợp dùng | Stack 2026 |
|----------|-----------|
| Text-to-image, chất lượng tốt nhất | Flow matching: SD3, Flux.1-dev |
| Text-to-image, 1-4 bước | Distilled flow matching: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| Suy luận thời gian thực | Consistency distillation từ một nền tảng khớp luồng (LCM, PCM) |
| Sinh âm thanh | Flow matching: Stable Audio 2.5, AudioCraft 2 |
| Sinh video | Flow matching trộn với diffusion (Sora, Veo, Stable Video) |
| Khoa học / vật lý (quỹ đạo hạt, phân tử) | Flow matching + trường vận tốc đẳng biến |

Khi một bài báo nói "nhanh hơn khuếch tán" trong 2025-2026, gần như luôn là flow matching + chưng cất.

## Đóng gói

Lưu `outputs/skill-fm-tuner.md`. Skill này nhận một spec mô hình kiểu khuếch tán và chuyển thành cấu hình huấn luyện flow matching: chọn lịch, phân phối lấy mẫu thời gian (uniform / logit-normal), optimizer, kế hoạch reflow, số bước mục tiêu, giao thức đánh giá.

## Bài tập

1. **Dễ.** Chạy `code/main.py` và so sánh MSE của 1 bước với 20 bước so với phân phối dữ liệu thật.
2. **Trung bình.** Chuyển từ lấy mẫu `t` đều sang logit-normal (tập trung lấy mẫu ở t giữa). Chất lượng mô hình có tăng không?
3. **Khó.** Cài đặt một vòng reflow: sinh cặp (x_0, x_1) bằng cách tích phân mô hình đầu, huấn luyện mô hình thứ hai trên các cặp đó, và so sánh chất lượng mẫu 1 bước.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Flow matching | "Khuếch tán đường thẳng" | Huấn luyện `v_θ(x, t)` khớp `x_1 - x_0` dọc một phép nội suy. |
| Rectified flow | "Reflow" | Thủ tục lặp nắn thẳng các luồng đã học. |
| Velocity field | "v_θ" | Đầu ra của mô hình — hướng cần dịch chuyển `x_t`. |
| Straight-line interpolant | "Đường đi" | `x_t = (1-t)·x_0 + t·x_1`; đạo hàm mục tiêu tầm thường. |
| Euler sampler | "Bộ giải ODE bậc 1" | Bộ tích phân đơn giản nhất; chạy tốt khi đường đi thẳng. |
| Logit-normal t | "Lấy mẫu SD3" | Tập trung lấy mẫu `t` về vùng giữa nơi gradient mạnh nhất. |
| Consistency distillation | "Bộ lấy mẫu 1 bước" | Huấn luyện mô hình học trò ánh xạ mọi `x_t` trực tiếp về `x_0`. |
| CFG with velocity | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`; cùng mẹo, biến mới. |

## Ghi chú sản xuất: Flux.1-schnell là flow matching ở tốc độ nhanh nhất

Thành quả sản xuất của flow matching là Flux.1-schnell — một DiT khớp luồng được chưng cất xuống 1-4 bước suy luận mà vẫn giữ chất lượng cỡ Flux-dev. Notebook "Run Flux on an 8GB machine" của Niels là công thức triển khai tham chiếu: mã hoá T5 + CLIP, khử nhiễu MMDiT lượng tử hoá (4 bước cho schnell so với 50 cho dev), giải mã VAE. Bài toán chi phí:

| Biến thể | Số bước | Độ trễ ở 1024² trên L4 | Tổng FLOPs (tương đối) |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (raw) | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08× (12× nhanh hơn) |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2 bước | 2 | ~0.3 s | 0.03× |

Quy tắc sản xuất: **nền tảng khớp luồng + chưng cất = mặc định 2026 cho text-to-image nhanh.** Mọi nhà cung cấp lớn đều phát hành combo này: SD3-Turbo (SD3 + flow + chưng cất), Flux-schnell (Flux-dev + nắn thẳng rectified flow), CogView-4-Flash. Nền tảng khuếch tán thuần chỉ còn cho các checkpoint cũ.

## Đọc thêm

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — rectified flow.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, rectified flow ở quy mô lớn.
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — khung tổng quát bao trùm cả FM và diffusion.
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — chưng cất 1 bước cho diffusion / flow.
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — biến thể turbo.
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — flow matching trong sản xuất.

---

[← Bài trước: Sinh 3D](../../12-3d-generation/docs/vi.md) · [Bài sau: Đánh giá →](../../14-evaluation-fid-clip-score/docs/vi.md)
