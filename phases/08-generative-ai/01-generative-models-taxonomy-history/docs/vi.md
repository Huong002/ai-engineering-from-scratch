# Mô hình sinh — Phân loại & Lịch sử

> Mọi mô hình ảnh, mô hình văn bản, mô hình video và mô hình 3D đều rơi vào một trong năm nhóm. Chọn sai nhóm là bạn sẽ vật lộn với phần toán cả mấy tuần. Chọn đúng thì mười hai năm tiến bộ gần nhất của lĩnh vực này sẽ xếp gọn trong đầu bạn.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 2 (ML Fundamentals), Giai đoạn 3 (Deep Learning Core), Giai đoạn 7, Bài 14 (Transformer)
**Time:** ~45 phút

## Vấn đề

Một mô hình sinh (generative model) làm đúng một việc: cho các mẫu huấn luyện rút ra từ một phân phối chưa biết `p_data(x)`, hãy xuất ra các mẫu mới trông như thể cũng đến từ phân phối đó. Khuôn mặt, câu văn, tệp MIDI, cấu trúc protein — nếu nhìn kỹ thì đều là cùng một bài toán.

Vấn đề là `p_data` sống trong một không gian hàng triệu chiều (ảnh RGB 512x512 là khoảng 786k chiều), các mẫu nằm trên một đa tạp mỏng bên trong không gian đó, và bạn chỉ có chừng 10M ví dụ. Vét cạn mật độ là vô vọng. Mọi mô hình sinh đều là một thỏa hiệp: đổi một bài toán khó lấy một bài toán khó ít hơn một chút.

Năm họ mô hình đã sống sót qua mười hai năm vừa rồi. Biết mỗi họ thỏa hiệp điều gì sẽ cho bạn biết vì sao nó thắng ở vài tác vụ và sụp đổ ở những tác vụ khác.

## Khái niệm

![Năm họ mô hình sinh — phân loại (taxonomy) theo thứ chúng mô hình hóa](../assets/taxonomy.svg)

**1. Mật độ tường minh, tính được.** Viết `log p(x)` thành một tổng mà bạn thực sự tính được. Các mô hình autoregressive (tự hồi quy) như PixelCNN, WaveNet, GPT phân rã `p(x) = ∏ p(x_i | x_<i)`. Normalizing flows (RealNVP, Glow) xây `p(x)` như một biến đổi khả nghịch của một phân phối cơ sở đơn giản. Ưu: likelihood (hợp lý) chính xác, hàm mất mát huấn luyện gọn gàng. Nhược: suy luận tự hồi quy là tuần tự (chậm với chuỗi dài), flows cần kiến trúc khả nghịch (hạn chế về kiến trúc).

**2. Mật độ tường minh, xấp xỉ.** Chặn dưới `log p(x)` (ELBO) và tối ưu chặn đó. VAE (Kingma 2013) dùng một encoder-decoder với hậu nghiệm biến phân. Các mô hình diffusion (khuếch tán) như DDPM (Ho 2020) huấn luyện một denoiser tối ưu ngầm một ELBO có trọng số. Diffusion là xương sống thống trị cho ảnh, video và 3D năm 2026.

**3. Mật độ ẩn.** Bỏ qua mật độ hoàn toàn; học một generator `G(z)` sinh mẫu và một discriminator `D(x)` phân biệt thật với giả. GAN (Goodfellow 2014). Nhanh khi suy luận (một lượt forward) nhưng nổi tiếng là không ổn định khi huấn luyện. StyleGAN 1/2/3 vẫn là state of the art cho ảnh chân thực trong miền hẹp (khuôn mặt, phòng ngủ) ngay cả năm 2026.

**4. Dựa trên score / thời gian liên tục.** Học trực tiếp gradient của log-mật độ `∇_x log p(x)` (score). Song & Ermon (2019) chỉ ra score matching tổng quát hóa diffusion thành một SDE. Flow matching (Lipman 2023) là thứ hot 2024-2026: huấn luyện không cần mô phỏng, đường đi thẳng hơn, tốc độ sampling (lấy mẫu) nhanh hơn DDPM 4-10x. Stable Diffusion 3, Flux, AudioCraft 2 đều dùng flow matching.

**5. Autoregressive trên token rời rạc.** Nén dữ liệu nhiều chiều bằng VQ-VAE hoặc một bộ lượng tử hoá dư thành một chuỗi token rời rạc ngắn, rồi dùng Transformer mô hình hóa chuỗi token đó. Parti, MuseNet, AudioLM, VALL-E, patch tokenizer của Sora đều dùng cách này. Đây là nhóm 1 cộng thêm một tokenizer đã học.

## Lịch sử tóm tắt

| Năm | Mô hình | Vì sao quan trọng |
|------|-------|-----------------|
| 2013 | VAE (Kingma) | Mô hình sinh sâu đầu tiên có hàm mất mát huấn luyện dùng được. |
| 2014 | GAN (Goodfellow) | Mật độ ẩn, không likelihood — mẫu sắc nét đến kinh ngạc. |
| 2015 | DRAW, PixelCNN | Sinh ảnh tuần tự. |
| 2017 | Glow, RealNVP | Flow khả nghịch; likelihood chính xác cùng độ sâu. |
| 2017 | Progressive GAN | Khuôn mặt megapixel đầu tiên. |
| 2019 | StyleGAN / StyleGAN2 | Khuôn mặt chân thực vẫn khó bị đánh bại trong đúng miền đó. |
| 2020 | DDPM (Ho) | Diffusion trở nên thực dụng. |
| 2021 | CLIP, DALL-E 1, VQGAN | Text-to-image thành phổ thông. |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | Latent diffusion + điều kiện văn bản = thứ phổ cập. |
| 2022 | ControlNet, LoRA | Điều khiển tinh vi mô hình diffusion đã tiền huấn luyện. |
| 2023 | SDXL, Midjourney v5, Flow matching | Quy mô + động học huấn luyện tốt hơn. |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | Diffusion cho video; flow matching thắng. |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | Video chất lượng sản xuất. |
| 2026 | Consistency + Rectified Flow | Lấy mẫu một bước từ xương sống diffusion. |

## Năm câu hỏi phân loại nhanh

Khi có một bài báo mô hình sinh mới, hãy trả lời năm câu hỏi này trước khi đọc phần phương pháp.

1. **Đang mô hình hóa cái gì?** Điểm ảnh, latents (biến ẩn), token rời rạc, Gaussian 3D, mesh, dạng sóng?
2. **Mật độ là tường minh hay ẩn?** Họ có viết ra `log p(x)` không?
3. **Lấy mẫu: một phát hay lặp?** Lặp nghĩa là suy luận chậm hơn; một phát thường là adversarial (đối kháng) hoặc đã chưng cất.
4. **Điều kiện: không điều kiện, theo lớp, văn bản, ảnh, pose?** Điều này quyết định hàm mất mát và giàn giáo kiến trúc.
5. **Đánh giá: FID, CLIP score, IS, sở thích con người, độ chính xác tác vụ?** Mỗi thứ có kiểu hỏng đã biết (xem Bài 14).

Bạn sẽ trả lời lại năm câu này ở mọi bài trong giai đoạn này. Đến cuối, chúng sẽ thành phản xạ.

```figure
autoencoder-bottleneck
```

## Tự tay làm

Code của bài này là một minh họa nhẹ: khớp một hỗn hợp Gaussian 1 chiều từ các mẫu bằng ba cách tiếp cận đồ chơi (kernel density, histogram rời rạc, và một generator "kiểu GAN" lấy mẫu gần nhất) để bạn thấy khác biệt giữa mật độ tường minh và ẩn trên một bài toán in vừa một màn hình.

Chạy `code/main.py`. Nó rút 2000 mẫu từ hỗn hợp hai đỉnh Gaussian, rồi in:

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

Để ý: hai cách đầu cho bạn hỏi "điểm này có khả năng đến mức nào?" Cách thứ ba thì không. Đây chính là khác biệt *tường minh vs ẩn* sẽ có ý nghĩa ở mọi bài sau này.

## Áp dụng

Họ nào, cho tác vụ nào, vào năm 2026?

| Tác vụ | Họ tốt nhất | Vì sao |
|------|-------------|-----|
| Mặt người chân thực, miền hẹp | StyleGAN 2/3 | Vẫn sắc nét nhất, suy luận nhanh nhất. |
| Text-to-image tổng quát | Latent diffusion + flow matching | SD3, Flux.1, DALL-E 3. |
| Text-to-image nhanh | Rectified flow + chưng cất | SDXL-Turbo, SD3-Turbo, LCM. |
| Text-to-video | Diffusion Transformer + flow matching | Sora, Veo 2, Kling. |
| Giọng nói + âm nhạc | AR dựa trên token (AudioLM, VALL-E, MusicGen) hoặc flow matching (AudioCraft 2) | Token rời rạc mở rộng rẻ. |
| Cảnh 3D | Fit Gaussian Splatting, prior diffusion | 3D-GS để tái dựng, diffusion cho góc nhìn mới. |
| Ước lượng mật độ (không lấy mẫu) | Flows | Họ duy nhất có `log p(x)` chính xác. |
| Mô phỏng / vật lý | Flow matching, score SDE | Đường thẳng, trường vector mượt. |

## Đóng gói

Lưu thành `outputs/skill-model-chooser.md`.

Skill này nhận một mô tả tác vụ và xuất ra: (1) nên dùng họ nào, (2) danh sách xếp hạng gồm ba lựa chọn mã nguồn mở và ba lựa chọn dịch vụ, (3) kiểu hỏng bạn nên cảnh giác, và (4) ngân sách compute/thời gian.

## Bài tập

1. **Dễ.** Với mỗi sản phẩm trong năm sản phẩm sau, xác định họ và xương sống: ChatGPT image, Midjourney v7, Sora, Runway Gen-3, ElevenLabs. Bằng chứng nên lấy từ các báo cáo kỹ thuật công khai.
2. **Trung bình.** Bài báo bạn sắp đọc ngày mai tuyên bố lấy mẫu nhanh hơn diffusion 100x. Viết ra ba câu hỏi để kiểm tra xem tốc độ đó còn đúng khi có điều kiện và độ phân giải cao hay không.
3. **Khó.** Chọn một miền bạn quan tâm (ví dụ cấu trúc protein, CAD, phân tử, quỹ đạo). Trả lời năm câu hỏi phân loại cho mô hình SOTA hiện tại trong miền đó và phác thảo mô hình tốt hơn sẽ thay đổi điều gì.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Generative model | "Nó tạo ra thứ mới" | Học một bộ lấy mẫu cho `p_data(x)`, có thể phơi ra `log p(x)`. |
| Explicit density | "Bạn tính được nó" | Mô hình cung cấp `log p(x)` dạng đóng hoặc tính được. |
| Implicit density | "Kiểu GAN" | Chỉ có bộ lấy mẫu — không cách nào tính `p(x)` của một điểm cho trước. |
| ELBO | "Chặn dưới bằng chứng" | Một chặn dưới tính được của `log p(x)`; VAE và diffusion tối ưu nó. |
| Score | "Gradient của log-mật độ" | `∇_x log p(x)`; diffusion và mô hình SDE học trường này. |
| Manifold hypothesis | "Dữ liệu sống trên một mặt" | Dữ liệu nhiều chiều tập trung trên một đa tạp chiều thấp; đó là lý do giảm chiều hiệu quả. |
| Autoregressive | "Đoán phần tiếp theo" | Phân rã phân phối đồng thời thành tích các phân phối có điều kiện. |
| Latent | "Code đã nén" | Biểu diễn chiều thấp mà từ đó decoder có thể tái dựng đầu vào. |

## Ghi chú sản xuất: năm họ, năm kiểu suy luận

Mỗi họ ứng với một đường chi phí máy chủ suy luận khác nhau. Tài liệu về suy luận trong sản xuất mô tả suy luận LLM thành prefill + decode; cách phân rã tương tự cũng áp dụng ở đây:

- **Autoregressive (nhóm 1 và 5).** Decode tuần tự chi phối độ trễ; KV-cache, continuous batching và speculative decoding đều áp dụng trực tiếp.
- **VAE / diffusion / flow-matching (nhóm 2 và 4).** Không có decode theo nghĩa LLM. Chi phí = `num_steps × step_cost`, và `step_cost` là một lượt forward transformer hoặc U-Net ở độ phân giải latent đầy đủ. Các nút vặn trong sản xuất là số bước (DDIM / DPM-Solver / chưng cất), batch size, và độ chính xác số (bf16 / fp8 / int4).
- **GAN (nhóm 3).** Một lượt forward. Không lịch trình, không KV-cache. TTFT ≈ tổng độ trễ. Đây là lý do StyleGAN vẫn thắng ở UX miền hẹp.

Khi thấy "nhanh hơn diffusion" trong tóm tắt một bài báo, hãy dịch nó thành "ít bước hơn × cùng chi phí mỗi bước" hoặc "cùng số bước × chi phí mỗi bước rẻ hơn". Còn lại đều là marketing.

## Đọc thêm

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — bài báo GAN.
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — bài báo VAE.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — bài báo DDPM.
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — diffusion như một SDE.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — bài báo flow matching.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3.

---

[Bài sau: Autoencoder (bộ tự mã hoá) & Variational Autoencoder (VAE) →](../../02-autoencoders-vae/docs/vi.md)
