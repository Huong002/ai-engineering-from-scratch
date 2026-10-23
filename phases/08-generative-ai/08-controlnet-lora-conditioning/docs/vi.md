# ControlNet, LoRA & điều kiện hoá

> Chỉ riêng văn bản là một tín hiệu điều khiển vụng về. ControlNet cho phép bạn nhân bản một mô hình diffusion đã huấn luyện trước rồi lái nó bằng bản đồ độ sâu, khung xương tư thế, nét vẽ nguệch ngoạc hay ảnh biên. LoRA cho phép bạn tinh chỉnh (fine-tuning) một mô hình 2 tỷ tham số bằng cách chỉ huấn luyện 10 triệu tham số. Cùng nhau, chúng biến Stable Diffusion từ món đồ chơi thành pipeline ảnh năm 2026 mà mọi agency đều dùng.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 8, Bài 07 (Latent diffusion), Giai đoạn 10 (LLM from scratch — nền tảng LoRA)
**Time:** ~75 phút

## Vấn đề

Một prompt như "a woman in a red dress walking a dog on a busy street" không cho mô hình biết *con chó ở đâu*, *người phụ nữ đang ở tư thế nào*, hay *phối cảnh* của con phố. Văn bản chỉ ghim được khoảng 10% những gì bạn cần để chỉ định một bức ảnh. Phần còn lại là thị giác và không thể mô tả hiệu quả bằng lời.

Huấn luyện một mô hình có điều kiện mới từ đầu cho mỗi tín hiệu (pose, depth, canny, segmentation) là quá tốn kém. Bạn muốn giữ nguyên phần xương sống SDXL 2,6 tỷ tham số, gắn thêm một mạng phụ nhỏ đọc tín hiệu điều kiện, và để nó đẩy nhẹ các đặc trưng trung gian của xương sống. Đó chính là ControlNet.

Bạn cũng muốn dạy mô hình những khái niệm mới (khuôn mặt bạn, sản phẩm của bạn, phong cách của bạn) mà không phải huấn luyện lại toàn bộ mô hình. Bạn muốn một delta nhỏ hơn 100 lần. Đó là LoRA — low-rank adaptation (thích ứng hạng thấp), tức những adapter (bộ nối) cắm vào trọng số attention sẵn có.

ControlNet + LoRA + văn bản = bộ đồ nghề của người làm thực tế năm 2026. Hầu hết pipeline ảnh trong sản xuất xếp 2-5 LoRA, 1-3 ControlNet và một IP-Adapter lên trên nền SDXL / SD3 / Flux.

## Khái niệm

![ControlNet nhân bản encoder; LoRA thêm các delta hạng thấp](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

Lấy một SD đã huấn luyện trước. *Nhân bản* nửa encoder của U-Net. Đóng băng bản gốc. Huấn luyện bản sao để nhận thêm một đầu vào conditioning (điều kiện hoá): biên, độ sâu, tư thế. Nối bản sao trở lại nửa decoder của bản gốc bằng các skip connection *tích chập không* (conv 1×1 khởi tạo bằng 0 — khởi đầu là no-op, rồi học một delta).

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

Khởi tạo zero-conv nghĩa là ControlNet bắt đầu như hàm đồng nhất — chưa huấn luyện cũng không gây hại. Huấn luyện trên 1M bộ ba (prompt, condition, image) với hàm mất mát diffusion chuẩn.

ControlNet theo từng phương thức ra mắt dưới dạng mô hình phụ nhỏ (~360M cho SDXL, ~70M cho SD 1.5). Bạn có thể ghép chúng lúc suy luận:

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021)

Với mọi lớp tuyến tính `W ∈ R^{d×d}` trong mô hình, đóng băng `W` và thêm một delta hạng thấp:

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

với `r << d`. Hạng (rank) 4-16 là chuẩn cho attention, hạng 64-128 cho các tinh chỉnh nặng. Số tham số mới: `2 · d · r` thay vì `d²`. Với attention SDXL có `d=640`, `r=16`: 20k tham số mỗi adapter thay vì 410k — giảm 20 lần. Trên toàn mô hình: một LoRA thường nặng 20-200MB so với 5GB của mô hình gốc.

Lúc suy luận bạn có thể scale LoRA: `W' = W + α · B @ A`. `α = 0.5-1.5` là bình thường. Nhiều LoRA xếp cộng dồn (kèm lưu ý quen thuộc rằng chúng tương tác theo cách phi tuyến).

### IP-Adapter (Ye et al., 2023)

Một adapter rất nhỏ nhận *ảnh* làm conditioning (song song với văn bản). Nó dùng encoder ảnh CLIP để tạo image token, rồi tiêm các token đó vào cross-attention cùng với token văn bản. ~20MB cho mỗi mô hình gốc. Cho phép bạn "tạo một ảnh theo phong cách của ảnh tham chiếu này" mà không cần LoRA.

## Ma trận khả năng ghép

| Công cụ | Điều khiển gì | Kích thước | Dùng khi nào |
|------|------------------|------|-------------|
| ControlNet | Cấu trúc không gian (tư thế, độ sâu, biên) | 70-360MB | Bố cục, thành phần chính xác |
| LoRA | Phong cách, chủ thể, khái niệm | 20-200MB | Cá nhân hoá, phong cách |
| IP-Adapter | Phong cách hoặc chủ thể từ ảnh tham chiếu | 20MB | Khi không văn bản nào tả được vẻ ngoài |
| Textual Inversion | Một khái niệm đơn lẻ thành token mới | 10KB | Cách cũ, phần lớn đã bị LoRA thay thế |
| DreamBooth | Tinh chỉnh toàn phần trên một chủ thể | 2-5GB | Danh tính mạnh, tính toán cao |
| T2I-Adapter | Phương án thay thế ControlNet nhẹ hơn | 70MB | Thiết bị biên, ngân sách suy luận |

ControlNet ≈ spatial control (điều khiển không gian). LoRA ≈ ngữ nghĩa. Dùng cả hai.

```figure
v4-controlnet-zero
```

## Tự tay làm

`code/main.py` mô phỏng hai cơ chế trên không gian 1-D:

1. **LoRA.** Một lớp tuyến tính đã huấn luyện trước `W`. Đóng băng nó. Huấn luyện một `B @ A` hạng thấp sao cho `W + BA` khớp với một lớp tuyến tính đích. Cho thấy `r = 1` là đủ để học hoàn hảo một hiệu chỉnh hạng 1.

2. **ControlNet-lite.** Một bộ dự đoán "nền đóng băng" và một "mạng phụ" đọc thêm tín hiệu. Đầu ra của mạng phụ được gated bởi một scalar học được khởi tạo bằng 0 (phiên bản zero-conv của chúng ta). Huấn luyện và xem cổng (gate) tăng dần.

### Bước 1: Toán của LoRA

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### Bước 2: mạng phụ khởi tạo zero

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

Ở bước 0 đầu ra giống hệt bản nền. Giai đoạn đầu huấn luyện cập nhật `gate` rất chậm — không có trôi lệch thảm khốc.

## Cạm bẫy

- **Scale LoRA quá tay.** `α = 2` hoặc `α = 3` là mẹo "làm nó mạnh hơn" quen thuộc, tạo ra đầu ra quá đậm phong cách hoặc hỏng. Giữ `α ≤ 1.5`.
- **Xung đột trọng số ControlNet.** Dùng ControlNet Pose ở weight 1.0 và ControlNet Depth ở weight 1.0 thường vượt ngưỡng. Tổng weight ≈ 1.0 là mặc định an toàn.
- **LoRA sai nền.** LoRA của SDXL âm thầm không có tác dụng trên SD 1.5 vì số chiều attention không khớp. Diffusers sẽ cảnh báo từ 0.30+.
- **Trôi lệch của Textual Inversion.** Token huấn luyện trên một checkpoint sẽ trôi rất xấu trên checkpoint khác. LoRA dễ mang đi hơn.
- **Gộp trọng số LoRA và lưu trữ.** Bạn có thể nướng một LoRA vào trọng số mô hình gốc để suy luận nhanh hơn (không cộng thêm lúc chạy), nhưng mất khả năng scale `α` lúc chạy. Hãy giữ cả hai phiên bản.

## Áp dụng

| Mục tiêu | Pipeline năm 2026 |
|------|---------------|
| Tái tạo phong cách nghệ thuật của một thương hiệu | LoRA huấn luyện trên ~30 ảnh tuyển chọn ở rank 32 |
| Đưa khuôn mặt tôi vào ảnh sinh ra | DreamBooth hoặc LoRA + IP-Adapter-FaceID |
| Tư thế cụ thể + prompt | ControlNet-Openpose + SDXL + văn bản |
| Bố cục hiểu độ sâu | ControlNet-Depth + SD3 |
| Ảnh tham chiếu + prompt | IP-Adapter + văn bản |
| Bố cục chính xác | ControlNet-Scribble hoặc ControlNet-Canny |
| Thay nền | ControlNet-Seg + Inpainting (Bài 09) |
| Phong cách 1 bước nhanh | LCM-LoRA trên SDXL-Turbo |

## Đóng gói

Lưu `outputs/skill-sd-toolkit-composer.md`. Skill nhận một nhiệm vụ (tài nguyên đầu vào: prompt, ảnh tham chiếu tuỳ chọn, tư thế tuỳ chọn, độ sâu tuỳ chọn, nét vẽ tuỳ chọn) và xuất ra chồng công cụ, các weight, cùng một quy trình seed tái lập được.

## Bài tập

1. **Dễ.** Trong `code/main.py`, cho rank `r` của LoRA chạy từ 1 đến 4. Ở rank nào thì LoRA khớp chính xác một delta đích hạng 2?
2. **Trung bình.** Huấn luyện hai LoRA riêng cho hai phép biến đổi đích. Nạp chúng cùng lúc và cho thấy tương tác cộng dồn của chúng. Khi nào tương tác đó phá vỡ tính tuyến tính?
3. **Khó.** Dùng diffusers để xếp chồng: SDXL-base + Canny-ControlNet (weight 0.8) + một LoRA phong cách (α 0.8) + IP-Adapter (weight 0.6). Đo đánh đổi giữa FID và độ bám prompt khi các weight trong chồng thay đổi.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| ControlNet | "Điều khiển không gian" | Encoder được nhân bản + skip zero-conv; đọc một ảnh điều kiện. |
| Zero convolution | "Khởi đầu như hàm đồng nhất" | Conv 1×1 khởi tạo bằng 0; ControlNet bắt đầu như no-op. |
| LoRA | "Adapter hạng thấp" | `W + B @ A`, `r << d`; ít hơn 100 lần tham số so với tinh chỉnh toàn phần. |
| rank r | "Núm vặn" | Độ nén của LoRA; 4-16 là điển hình, 64+ cho cá nhân hoá nặng. |
| α | "Độ mạnh LoRA" | Scale delta của LoRA lúc chạy. |
| IP-Adapter | "Ảnh tham chiếu" | Adapter điều kiện hoá bằng ảnh nhỏ, qua token ảnh CLIP. |
| DreamBooth | "Tinh chỉnh toàn bộ chủ thể" | Huấn luyện toàn mô hình trên ~30 ảnh của một chủ thể. |
| Textual Inversion | "Token mới" | Chỉ học một embedding từ mới; cách cũ, phần lớn đã bị thay thế. |

## Ghi chú sản xuất: hoán đổi LoRA, các làn ControlNet, phục vụ multi-tenant

Một SaaS text-to-image thật phục vụ hàng trăm LoRA và cả chục ControlNet trên cùng một checkpoint gốc. Bài toán phục vụ này trông rất giống multi-tenancy của LLM (tài liệu sản xuất bàn trường hợp LLM dưới continuous batching và LoRAX / S-LoRA):

- **Hoán đổi nóng LoRA, đừng gộp.** Gộp `W' = W + α·B·A` vào mô hình gốc cho suy luận mỗi bước nhanh hơn ~3-5% nhưng đóng băng `α` và mô hình gốc. Giữ LoRA nóng trong VRAM dưới dạng delta hạng r; diffusers có `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` để bật theo từng request. Chi phí hoán đổi là `2 · d · r · num_layers` trọng số — cỡ MB, dưới một giây.
- **ControlNet như một làn attention thứ hai.** Encoder được nhân bản chạy song song với mô hình gốc. Hai ControlNet cùng weight 1.0 = hai lượt forward thêm mỗi bước, không phải một lượt gộp. Dư địa batch size giảm theo bình phương. Hãy dự trù chi phí mỗi bước ~1,5× cho mỗi ControlNet đang bật.
- **LoRA cũng lượng tử hoá được.** Nếu bạn đã lượng tử hoá mô hình gốc (xem Bài 07, Flux trên 8GB), delta của LoRA cũng lượng tử hoá gọn sang 8-bit hoặc 4-bit. Cách nạp kiểu QLoRA cho phép bạn xếp 5-10 LoRA lên trên một nền Flux 4-bit mà không nổ bộ nhớ.

Riêng Flux: notebook Flux-on-8GB của Niels lượng tử hoá mô hình gốc xuống 4-bit; xếp một LoRA phong cách (`pipe.load_lora_weights("user/style-lora")`) lên nền đã lượng tử hoá đó với `weight_name="pytorch_lora_weights.safetensors"` vẫn chạy được. Đây là công thức mà phần lớn agency SaaS dùng năm 2026.

## Đọc thêm

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet.
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA (ban đầu cho LLM; sau chuyển sang diffusion).
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter.
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — phương án nhẹ hơn ControlNet.
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth.
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — các pipeline tham chiếu.

---

[← Bài trước: Latent diffusion (khuếch tán trong không gian ẩn) & Stable Diffusion](../../07-latent-diffusion-stable-diffusion/docs/vi.md) · [Bài sau: Inpainting, Outpainting & Chỉnh sửa ảnh →](../../09-inpainting-outpainting-editing/docs/vi.md)
