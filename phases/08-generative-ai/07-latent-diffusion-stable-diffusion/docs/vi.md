# Latent diffusion (khuếch tán trong không gian ẩn) & Stable Diffusion

> Khuếch tán trong không gian điểm ảnh trên ảnh 512×512 là một tội ác tính toán. Rombach và cộng sự (2022) nhận ra rằng bạn không cần đủ 786k chiều để tạo một bức ảnh — bạn chỉ cần đủ để nắm cấu trúc ngữ nghĩa, phần còn lại để một bộ giải mã riêng lo. Hãy chạy khuếch tán bên trong không gian ẩn (latent space) của VAE. Chỉ ý tưởng đó đã là Stable Diffusion.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 8, Bài 02 (VAE), Giai đoạn 8, Bài 06 (DDPM), Giai đoạn 7, Bài 09 (ViT)
**Time:** ~75 phút

## Vấn đề

Khuếch tán trong không gian điểm ảnh ở 512² nghĩa là U-Net chạy trên tensor có shape `[B, 3, 512, 512]`. Mỗi bước lấy mẫu tốn ~100 GFLOPS với U-Net 500M tham số. Năm mươi bước là 5 TFLOPS cho một ảnh. Huấn luyện trên một tỷ ảnh thì hoá đơn tính toán thật vô lý.

Phần lớn số FLOP đó đổ vào việc đẩy những chi tiết không quan trọng về mặt tri giác qua mạng — phần texture tần số cao mà một VAE có mất mát vẫn nén bỏ được. Ý tưởng của Rombach: huấn luyện VAE một lần (*giai đoạn thứ nhất*), đóng băng nó, rồi chạy khuếch tán hoàn toàn trong không gian ẩn 64×64 bốn kênh (*giai đoạn thứ hai*). Vẫn U-Net đó. Còn 1/16 số điểm ảnh. Ít hơn ~64 lần FLOP với chất lượng tương đương.

Đây chính là công thức của Stable Diffusion. SD 1.x / 2.x dùng U-Net 860M trên latent `64×64×4`, SDXL dùng U-Net 2.6B trên `128×128×4`, SD3 thay U-Net bằng Diffusion Transformer (DiT) với flow matching. Flux.1-dev (Black Forest Labs, 2024) dùng DiT-MMDiT 12B tham số. Tất cả đều chạy trên cùng nền hai giai đoạn.

## Khái niệm

![Khuếch tán trong không gian ẩn: VAE nén + khuếch tán trong không gian ẩn](../assets/latent-diffusion.svg)

**Hai giai đoạn, huấn luyện riêng.**

1. **Giai đoạn 1 — VAE.** Encoder `E(x) → z`, decoder `D(z) → x`. Mục tiêu nén: giảm mẫu 8× trên mỗi trục không gian và chỉnh số kênh để tổng kích thước latent còn ~1/16 số điểm ảnh. Loss = tái tạo (L1 + LPIPS cảm nhận tri giác) + KL (trọng số nhỏ để `z` không bị ép quá Gaussian, vì ta không cần lấy mẫu chính xác từ `z`). Thường huấn luyện kèm loss đối kháng để ảnh giải mã ra sắc nét.

2. **Giai đoạn 2 — khuếch tán trên `z`.** Coi `z = E(x_real)` là dữ liệu. Huấn luyện U-Net (hoặc DiT) khử nhiễu `z_t`. Khi suy luận: lấy mẫu `z_0` bằng khuếch tán, rồi `x = D(z_0)`.

**Điều kiện hoá văn bản.** Hai thành phần thêm vào. Một text encoder (bộ mã hoá văn bản) đóng băng (CLIP-L cho SD 1.x, CLIP-L+OpenCLIP-G cho SD 2/XL, T5-XXL cho SD3 và Flux). Một đường tiêm cross-attention (attention chéo): mỗi khối U-Net nhận `[Q = image features, K = V = text tokens]` rồi trộn chúng vào. Token là cách duy nhất để văn bản tác động lên ảnh.

**Hàm loss giống hệt Bài 06.** Vẫn MSE trên nhiễu kiểu DDPM / flow matching. Bạn chỉ đổi miền dữ liệu.

## Các biến thể kiến trúc

| Mô hình | Năm | Backbone | Hình dạng latent | Bộ mã hoá văn bản | Tham số |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 tokens) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + bộ tinh chỉnh | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Chưng cất | 128×128×4 | như trên | lấy mẫu 1-4 bước |
| SD3 | 2024 | MMDiT (DiT đa phương thức) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT chưng cất | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 bước |

Xu hướng: thay U-Net bằng DiT (transformer trên các patch latent), tăng quy mô text encoder (T5 bám prompt tốt hơn CLIP), tăng số kênh latent (4 → 16 cho thêm dư địa chi tiết).

```figure
noise-schedule
```

## Tự tay làm

`code/main.py` xếp một "VAE" 1-D đồ chơi (encoder + decoder đồng nhất, chỉ để minh hoạ; VAE thật sẽ là một conv net) lên trên DDPM của Bài 06 và thêm điều kiện hoá theo lớp với classifier-free guidance (hướng dẫn không cần bộ phân loại). Nó cho thấy cùng một loss khuếch tán vẫn hoạt động dù bạn chạy trên giá trị 1-D thô hay trên giá trị đã mã hoá — đó là điểm mấu chốt.

### Bước 1: encoder/decoder

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

VAE thật có trọng số đã huấn luyện. Về mặt sư phạm, phép ánh xạ tuyến tính này đủ để cho thấy khuếch tán hoạt động trên `z` mà không cần quan tâm tới không gian dữ liệu gốc.

### Bước 2: khuếch tán trong không gian `z`

Vẫn DDPM như Bài 06. Dữ liệu mà mạng thấy là `z = E(x)`. Sau khi lấy mẫu `z_0`, giải mã bằng `D(z_0)`.

### Bước 3: classifier-free guidance

Khi huấn luyện, bỏ nhãn lớp 10% số lần (thay bằng một null token). Khi suy luận, tính cả `ε_cond` và `ε_uncond`, rồi:

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = không hướng dẫn (đa dạng tối đa), `w = 3` = mặc định, `w = 7+` = bão hoà / quá sắc.

### Bước 4: điều kiện hoá văn bản (khái niệm, không phải code)

Thay nhãn lớp bằng đầu ra của một text encoder đóng băng. Đưa embedding văn bản vào U-Net qua cross-attention:

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

Đây là khác biệt thực chất duy nhất giữa một mô hình khuếch tán có điều kiện theo lớp và Stable Diffusion.

## Cạm bẫy

- **Lệch thang VAE.** VAE của SD 1.x có một hằng số co giãn (`scaling_factor ≈ 0.18215`) áp sau khi mã hoá. Quên nó là U-Net sẽ huấn luyện trên latent có phương sai sai lệch khủng khiếp. Checkpoint nào cũng kèm một hằng số như vậy.
- **Text encoder sai mà không báo.** SD3 cần T5-XXL với >=128 token, và việc lùi về chỉ dùng CLIP là mất mát. Luôn kiểm tra `use_t5=True`, nếu không độ trung thành với prompt sẽ tụt thê thảm.
- **Trộn lẫn các không gian ẩn.** SDXL, SD3, Flux dùng VAE khác nhau. Một LoRA huấn luyện trên latent của SDXL sẽ không chạy trên SD3. Hugging Face diffusers 0.30+ từ chối nạp checkpoint không khớp.
- **CFG quá cao.** `w > 10` tạo ảnh bão hoà, bệt như dầu và bám prompt quá mức, đánh đổi bằng đa dạng. Điểm ngọt là `w = 3-7`.
- **Rò rỉ negative prompt.** Negative prompt rỗng trở thành null token; negative prompt có nội dung trở thành `ε_uncond`. Hai thứ này không giống nhau; một số pipeline lặng lẽ mặc định về null.

## Áp dụng

Các stack production năm 2026:

| Mục tiêu | Backbone khuyến nghị |
|--------|----------------------|
| Miền hẹp, dữ liệu có cặp, huấn luyện mô hình từ đầu | Fine-tune SDXL (LoRA / toàn bộ) — nhanh triển khai nhất |
| Text-to-image miền mở, trọng số mở | Flux.1-dev (12B, Apache / phi thương mại) hoặc SD3.5-Large |
| Suy luận nhanh nhất, trọng số mở | Flux.1-schnell (1-4 bước, Apache) hoặc SDXL-Lightning |
| Bám prompt tốt nhất, dạng dịch vụ | GPT-Image / DALL-E 3 (vẫn vậy), Midjourney v7, Imagen 4 |
| Quy trình chỉnh sửa | Flux.1-Kontext (tháng 12/2024) — nhận ảnh + văn bản một cách tự nhiên |
| Nghiên cứu, baseline | SD 1.5 — cổ nhưng được nghiên cứu kỹ |

## Đóng gói

Lưu `outputs/skill-sd-prompter.md`. Skill nhận một prompt văn bản + phong cách đích và xuất ra: mô hình + checkpoint, thang CFG, sampler, negative prompt, độ phân giải, tổ hợp ControlNet/IP-Adapter tuỳ chọn, và checklist QA từng bước.

## Bài tập

1. **Dễ.** Chạy `code/main.py` với guidance `w ∈ {0, 1, 3, 7, 15}`. Ghi lại giá trị trung bình mẫu theo từng lớp. Ở `w` nào thì trung bình lớp lệch vượt qua trung bình dữ liệu thật?
2. **Trung bình.** Thay encoder tuyến tính đồ chơi bằng cặp encoder/decoder tanh-MLP có loss tái tạo. Huấn luyện lại khuếch tán trên latent mới. Chất lượng mẫu có thay đổi không?
3. **Khó.** Dựng suy luận Stable Diffusion thật với diffusers: nạp `sdxl-base`, chạy 30 bước Euler với CFG=7, đo thời gian. Giờ chuyển sang `sdxl-turbo` với 4 bước và CFG=0. Cùng chủ thể, chất lượng khác — mô tả cái gì đã thay đổi và vì sao.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| First stage | "Cái VAE" | Cặp encoder/decoder (bộ tự mã hoá) đã huấn luyện; nén 512² xuống 64². |
| Second stage | "Cái U-Net" | Mô hình khuếch tán trên không gian ẩn. |
| CFG | "Thang guidance" | `(1+w)·ε_cond - w·ε_uncond`; chỉnh độ mạnh điều kiện hoá. |
| Null token | "Embed prompt rỗng" | Embed vô điều kiện dùng cho `ε_uncond`. |
| Cross-attention | "Đường văn bản đi vào" | Mỗi khối U-Net attend tới token văn bản như K và V. |
| DiT | "Diffusion Transformer" | Thay U-Net bằng transformer trên các patch latent; mở rộng quy mô tốt hơn. |
| MMDiT | "DiT đa phương thức" | Kiến trúc của SD3: hai luồng văn bản và ảnh với attention chung. |
| VAE scaling factor | "Con số ma thuật" | Chia latent cho ~5.4 để khuếch tán hoạt động trong không gian phương sai đơn vị. |

## Ghi chú production: chạy Flux-12B trên GPU tiêu dùng 8GB

tích hợp Flux tham chiếu là công thức chuẩn mực cho câu hỏi "tôi có GPU tiêu dùng, tôi triển khai được không?". Mẹo là đúng công thức ba núm mà tài liệu suy luận production liệt kê, áp lên một DiT khuếch tán:

1. **Nạp so le.** Flux có ba mạng không bao giờ cần cùng tồn tại trong VRAM: text encoder T5-XXL (~10 GB ở fp32), CLIP-L (nhỏ), MMDiT 12B, và VAE. Mã hoá prompt trước, *xoá* các encoder, nạp DiT, khử nhiễu, *xoá* DiT, nạp VAE, giải mã. GPU tiêu dùng 8GB chỉ chứa được một giai đoạn một lúc.
2. **Lượng tử hoá 4-bit qua bitsandbytes.** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)` cho cả encoder T5 và DiT. Cắt bộ nhớ 8×, mức giảm chất lượng gần như không nhận ra với text-to-image theo benchmark của Aritra (có link trong notebook).
3. **Offload CPU.** `pipe.enable_model_cpu_offload()` tự hoán đổi module giữa CPU và GPU khi mỗi lượt forward tiến lên. Thêm 10-20% độ trễ nhưng giúp pipeline chạy được.

Bài toán bộ nhớ là: `10 GB T5 / 8 = 1.25 GB` sau lượng tử hoá, `12 B params × 0.5 bytes = ~6 GB` cho DiT đã lượng tử hoá, cộng thêm activation. Theo cách nói của stas00, đây là cực đoan của suy luận TP=1 — không song song hoá mô hình, lượng tử hoá tối đa. Cho production bạn sẽ chạy TP=2 hoặc TP=4 trên H100; còn cho một laptop dev, đây là công thức.

## Đọc thêm

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion.
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL.
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, MMDiT.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 family.
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — reference implementation for every checkpoint above.

---

[← Bài trước: Mô hình khuếch tán](../../06-diffusion-ddpm-from-scratch/docs/vi.md) · [Bài sau: ControlNet, LoRA & điều kiện hoá →](../../08-controlnet-lora-conditioning/docs/vi.md)
