# Sinh video (Video Generation)

> Một ảnh là tensor 2 chiều. Một video là tensor 3 chiều. Lý thuyết giống nhau; chi phí tính toán khó hơn 10-100 lần. Sora của OpenAI (tháng 2/2024) đã chứng minh việc đó là khả thi. Đến 2026, Veo 2, Kling 1.5, Runway Gen-3, Pika 2.0 và WAN 2.2 xuất video thương mại từ văn bản ở 1080p — và ngăn xếp trọng số mở (CogVideoX, HunyuanVideo, Mochi-1, WAN 2.2) chỉ chậm hơn 12 tháng.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 8, Bài 07 (Latent diffusion), Giai đoạn 7, Bài 09 (ViT), Giai đoạn 8, Bài 06 (DDPM)
**Time:** ~45 phút

## Vấn đề

Một video 1080p dài 10 giây ở 24fps là 240 khung hình (frame) với 1920×1080×3 pixel. Tức khoảng 1,5 GB dữ liệu thô cho mỗi clip. Diffusion trong không gian pixel là bất khả thi. Bạn cần:

1. **Nén không-thời gian.** Một VAE mã hoá video, chứ không phải từng khung hình, thành một chuỗi patch không-thời gian.
2. **Nhất quán thời gian.** Các khung hình phải chia sẻ nội dung, ánh sáng và danh tính đối tượng trong nhiều giây. Mạng phải mô hình hoá motion (chuyển động).
3. **Ngân sách tính toán.** Huấn luyện video tốn hơn ảnh 10-100 lần với cùng kích thước mô hình.
4. **Điều kiện hoá.** Văn bản, ảnh (khung đầu), âm thanh, hoặc một video khác. Hầu hết mô hình thương mại nhận cả bốn.

Kiến trúc giải quyết bài toán này là **Diffusion Transformer (DiT)** áp lên các patch không-thời gian, huấn luyện trên tập dữ liệu (prompt, caption, video) khổng lồ. Vẫn dùng loss diffusion như Bài 06.

## Khái niệm

![Diffusion video: patchify, DiT, giải mã](../assets/video-generation.svg)

### Patchify

Mã hoá video bằng một VAE 3D (nén không-thời gian học được). latent (ẩn) có shape `[T_latent, H_latent, W_latent, C_latent]`. Chia thành các patch kích thước `[t_p, h_p, w_p]`. Với các mô hình kiểu Sora, `t_p = 1` (patch theo từng khung) hoặc `t_p = 2` (hai khung một lần). Một video 1080p 10 giây nén xuống còn ~20.000-100.000 patch.

### DiT không-thời gian

Một transformer xử lý chuỗi patch phẳng. Mỗi patch có embedding vị trí 3D (thời gian + y + x). Attention thường được phân rã:

- **Attention không gian** trong các patch của từng khung hình.
- **Attention thời gian** (temporal layer, lớp thời gian) xuyên các khung hình tại cùng vị trí không gian.
- **Attention 3D đầy đủ** tốn hơn 16-100 lần; chỉ dùng ở độ phân giải thấp hoặc trong nghiên cứu.

### Điều kiện hoá bằng văn bản

Cross-attention với một text encoder lớn (T5-XXL cho Sora, CogVideoX-5B cũng dùng T5-XXL). Prompt dài rất quan trọng — tập huấn luyện của Sora có các re-caption dày do GPT sinh, trung bình 200 token mỗi clip.

### Huấn luyện

Loss diffusion chuẩn (dự đoán ε hoặc v) trên các latent không-thời gian. Dữ liệu: video trên web + ~100M clip được tuyển chọn + caption văn bản tổng hợp. Tính toán: hơn 10.000 giờ GPU cho một thử nghiệm nghiên cứu nhỏ; quy mô Sora là hơn 100.000.

## Bối cảnh sản xuất năm 2026

| Mô hình | Thời điểm | Thời lượng tối đa | Độ phân giải tối đa | Trọng số mở? | Đáng chú ý |
|-------|------|--------------|---------|---------------|---------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | Không | Mô hình đầu tiên cho thấy đặc tính máy mô phỏng thế giới ở quy mô lớn |
| Sora Turbo | 2024-12 | 20s | 1080p | Không | Sora bản sản xuất với suy luận nhanh gấp 5 lần |
| Veo 2 (Google) | 2024-12 | 8s | 4K | Không | Chất lượng và vật lý tốt nhất năm 2025 |
| Veo 3 | 2025 Q3 | 15s | 4K | Không | Âm thanh gốc và điều khiển camera mạnh hơn |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | Không | Chuyển động người tốt nhất quý 1/2025 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | Không | Công cụ dựng video chuyên nghiệp chạy bên trên |
| Pika 2.0 | 2024-10 | 5s | 1080p | Không | Nhất quán nhân vật mạnh nhất |
| CogVideoX (THUDM) | 2024 | 10s | 720p | Có (2B, 5B) | Video mở quy mô 5B đầu tiên |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | Có (13B) | SOTA mở cuối 2024 |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | Có (10B) | Giấy phép thoáng nhất |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | Có | Mô hình mở mạnh nhất giữa 2025 |

Trọng số mở đang thu hẹp khoảng cách nhanh hơn trong không gian ảnh: đến giữa 2026, HunyuanVideo + các LoRA của WAN 2.2 đã chạy phần lớn quy trình nguồn mở.

```figure
video-diffusion-denoise
```

## Tự tay làm

`code/main.py` mô phỏng ý tưởng DiT không-thời gian cốt lõi: patchify một video tổng hợp nhỏ, thêm embedding vị trí cho từng patch, rồi denoise toàn bộ chuỗi bằng attention kiểu transformer trên các patch. Không dùng numpy; Python thuần. Chúng ta chỉ ra rằng nhất quán thời gian vẫn xuất hiện ngay cả ở 1-D khi các patch của khung liền kề chia sẻ một denoiser và embedding vị trí.

### Bước 1: patchify một "video" 1-D tổng hợp

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### Bước 2: embedding vị trí cho mỗi khung hình

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### Bước 3: denoiser nhìn thấy toàn bộ chuỗi

Thay vì denoise từng khung hình độc lập, mạng tí hon của chúng ta nối tất cả giá trị khung hình + embedding vị trí của chúng rồi dự đoán nhiễu cho mọi khung hình cùng lúc.

### Bước 4: kiểm tra nhất quán thời gian

Sau khi huấn luyện, lấy mẫu một video. Đo delta giữa các khung liên tiếp. Nếu mô hình đã học được cấu trúc thời gian, các delta sẽ nhỏ hơn so với khi lấy mẫu từng khung độc lập.

## Cạm bẫy

- **Lấy mẫu từng khung độc lập = flicker (nhấp nháy).** Nếu bạn chạy diffusion ảnh trên từng khung riêng rẽ, đầu ra bị nhấp nháy vì nhiễu của mỗi khung là độc lập. Diffusion video sửa điều này bằng cách ghép nối các khung qua attention hoặc nhiễu chia sẻ.
- **Attention 3D ngây thơ = OOM.** Attention 3D đầy đủ trên latent 1080p 10 giây là hàng trăm tỷ phép tính. Hãy phân rã thành không gian + thời gian.
- **Caption dữ liệu quan trọng hơn kích thước.** Nâng cấp chính của Sora so với các công trình trước là huấn luyện trên caption chi tiết hơn ~10 lần (các clip được GPT-4 gắn nhãn lại). Báo cáo kỹ thuật của OpenAI nói rõ điều này.
- **Điều kiện hoá khung đầu.** Hầu hết mô hình thương mại còn nhận một ảnh làm khung đầu. Đây là chế độ "image-to-video"; quá trình huấn luyện có bao gồm biến thể này.
- **Trôi vật lý.** Các clip dài (>10s) tích tụ những điểm không nhất quán tinh vi. Sinh theo cửa sổ trượt + neo keyframe giúp ích.

## Áp dụng

| Trường hợp dùng | Lựa chọn 2026 |
|----------|-----------|
| Text-to-video chất lượng cao nhất, dạng hosted | Veo 3 hoặc Sora |
| Điện ảnh điều khiển bằng camera | Runway Gen-3 với motion brush |
| Nhất quán nhân vật xuyên các clip | Pika 2.0 hoặc Kling 2.1 |
| Trọng số mở, fine-tune nhanh | WAN 2.2 + LoRA |
| Image-to-video | WAN 2.2-I2V, Kling 2.1 I2V, hoặc Runway |
| Lip sync từ âm thanh sang video | Veo 3 (âm thanh gốc) hoặc một mô hình lip-sync chuyên dụng |
| Biên tập video | Runway Act-Two, Kling Motion Brush, Flux-Kontext (khung tĩnh) |

Chi phí cho mỗi giây video ở mức chất lượng tương đương đã giảm 20 lần trong giai đoạn 2024-2026.

## Đóng gói

Lưu `outputs/skill-video-brief.md`. Skill nhận một video brief (thời lượng, tỉ lệ khung, phong cách, kế hoạch camera, độ nhất quán chủ thể, âm thanh) và xuất ra: mô hình + nơi host, bộ khung prompt (ngôn ngữ camera, mô tả chủ thể, mô tả chuyển động), giao thức seed + tái lập, và checklist QA ở mức từng khung hình.

## Bài tập

1. **Dễ.** Trong `code/main.py`, so sánh delta giữa các khung cho (a) lấy mẫu từng khung độc lập, (b) lấy mẫu cả chuỗi cùng lúc. Báo cáo trung bình và phương sai của các delta.
2. **Trung bình.** Thêm điều kiện khung đầu: ghim khung 0 vào một giá trị cho trước rồi lấy mẫu phần còn lại. Đo xem giá trị được ghim lan truyền như thế nào.
3. **Khó.** Dùng HuggingFace diffusers để chạy CogVideoX-2B trên GPU cục bộ. Đo thời gian 20 bước suy luận ở 720p cho một clip 6 giây. Profile attention không-thời gian để tìm điểm nghẽn.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Video VAE | "VAE 3-D" | Encoder nén `(T, H, W, C)` → latent không-thời gian. |
| Patches | "Các token" | Các khối 3-D kích thước cố định của latent; đầu vào của DiT. |
| Factorized attention | "Không gian + thời gian" | Chạy attention trên không gian, rồi trên thời gian; bỏ qua attention 3-D đầy đủ. |
| Image-to-video (I2V) | "Làm ảnh này chuyển động" | Mô hình nhận một ảnh + văn bản, xuất video bắt đầu từ ảnh đó. |
| Keyframe conditioning | "Neo khung hình" | Ghim những khung cụ thể để điều khiển mạch video. |
| Motion brush | "Gợi ý hướng" | Đầu vào UI nơi người dùng vẽ các vector chuyển động lên ảnh. |
| Re-captioning | "Caption dày" | Dùng LLM gắn lại nhãn cho clip huấn luyện bằng prompt chi tiết. |
| Flicker | "Artefact thời gian" | Không nhất quán giữa các khung; sửa bằng denoise có ghép nối. |

## Ghi chú sản xuất: latent video là bài toán băng thông bộ nhớ

Một clip 1080p 10 giây ở 24 fps là 240 khung × 1920 × 1080 × 3 ≈ 1,5 GB pixel thô. Sau khi nén bằng VAE video 4× (`2 × spatial × 2 × temporal`) thì latent còn ~100 MB mỗi request. Chạy qua DiT không-thời gian 30 bước với batch 1 nghĩa là bạn đang di chuyển ~3 GB/bước qua HBM — băng thông bộ nhớ, chứ không phải FLOPs, mới là điểm nghẽn.

Ba núm vặn trong sản xuất, đều lấy từ chương về suy luận trong tài liệu production-inference:

- **TP xuyên DiT.** Các mô hình text-to-video thường ≥10B tham số. TP=4 trên 4 H100 là chuẩn; PP=2 × TP=2 cho các mô hình cỡ 405B. Độ trễ mỗi bước giảm gần như tuyến tính theo TP cho tới khi chạm bức tường all-reduce.
- **Batch theo khung = continuous batching.** Ở thời điểm sinh, video về mặt khái niệm là một batch các khung được nối bằng attention. Continuous batching (lập lịch trong khi chạy) áp dụng được: bắt đầu render khung `t+1` trong khi khung `t-1` đang được trả về, nếu kiến trúc mô hình cho phép sinh theo cửa sổ trượt.
- **Cache prefill ở mức clip.** Với image-to-video, điều kiện hoá khung đầu tương tự prefill prompt của LLM: tính một lần, tái dùng qua các lượt của decoder thời gian. Đây thực chất là KV-cache cho video.

## Đọc thêm

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — báo cáo kỹ thuật Sora.
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX.
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo.
- [Genmo (2024). Mochi 1 preview model card](https://huggingface.co/genmo/mochi-1-preview) — Mochi-1.
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — SOTA mở giữa 2025.
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — bài báo Video Diffusion đầu tiên.
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — tổ tiên của Stable Video Diffusion.

---

[← Bài trước: Inpainting, Outpainting & Chỉnh sửa ảnh](../../09-inpainting-outpainting-editing/docs/vi.md) · [Bài sau: Sinh âm thanh →](../../11-audio-generation/docs/vi.md)
