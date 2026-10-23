# Transfusion: Văn bản autoregressive (tự hồi quy) + Ảnh diffusion (khuếch tán) trong một Transformer

> Chameleon và Emu3 đặt cược tất cả vào token rời rạc. Chúng chạy được, nhưng nút thắt lượng tử hoá hiện rõ — chất lượng ảnh bão hoà dưới mức của các mô hình diffusion trong không gian liên tục. Transfusion (Meta, Zhou et al., tháng 8/2024) đặt cược ngược lại: giữ ảnh liên tục, bỏ hẳn VQ-VAE, và huấn luyện một transformer với hai loss. Token văn bản nhận dự đoán token kế tiếp. Patch ảnh nhận loss flow matching (khớp luồng) / diffusion. Cả hai mục tiêu cùng tối ưu một bộ trọng số. Kiến trúc đứng sau Stable Diffusion 3 (MMDiT) là họ hàng gần. Bài này đọc luận đề Transfusion, dựng một bộ huấn luyện hai loss đồ chơi, và lần theo attention mask cho phép một transformer làm cả hai việc.

**Type:** Build
**Languages:** Python (stdlib, two-loss trainer on MNIST-scale toy)
**Prerequisites:** Giai đoạn 12, Bài 11 (Chameleon), Giai đoạn 8 (AI sinh tạo)
**Time:** ~180 phút

## Mục tiêu học tập

- Nối dây một transformer chạy hai loss (NTP trên token văn bản, MSE diffusion trên patch ảnh) trên một backbone duy nhất.
- Giải thích vì sao attention hai chiều qua các patch ảnh cộng với attention nhân quả trên token văn bản là lựa chọn mask đúng.
- So sánh kiểu Transfusion (ảnh liên tục, loss diffusion) với kiểu Chameleon (ảnh rời rạc, NTP) về compute, chất lượng và độ phức tạp code.
- Gọi tên đóng góp của MMDiT: trọng số riêng theo modality ở mỗi block, attention chung tại residual stream.

## Vấn đề

Cuộc tranh luận token ảnh rời rạc so với liên tục còn cũ hơn cả LLM. Biểu diễn liên tục (continuous token — token liên tục; pixel thô, latent VAE) bảo toàn chi tiết. Token rời rạc (chỉ số VQ) khớp với từ vựng gốc của transformer nhưng mất chi tiết ngay ở bước lượng tử hoá.

Chameleon / Emu3 đi hướng rời rạc: một loss, một kiến trúc, nhưng độ trung thực ảnh bị chặn trên bởi chất lượng tokenizer.

Mô hình diffusion đi hướng liên tục: chất lượng ảnh vượt trội, nhưng là mô hình tách rời khỏi LLM, kỹ nghệ lịch nhiễu phức tạp, và không tích hợp gọn với sinh văn bản.

Transfusion hỏi: có thể có cả hai không? Giữ ảnh liên tục, vẫn huấn luyện một mô hình, dùng hai loss ghép vào một bước gradient.

## Khái niệm

### Kiến trúc hai loss

Một transformer chỉ gồm decoder xử lý một chuỗi chứa:

- Token văn bản (rời rạc, từ từ vựng BPE).
- Patch ảnh (liên tục, khối 16x16 pixel chiếu vào hidden dim bằng embedding tuyến tính — giống đầu vào của encoder ViT).
- Thẻ `<image>` và `</image>` đánh dấu nơi các patch liên tục nằm.

Lượt forward chạy một lần. Loss chọn một trong hai head cho mỗi token:

- Với token văn bản: cross-entropy chuẩn trên head logits từ vựng.
- Với patch ảnh: loss diffusion trên patch liên tục — dự đoán nhiễu đã được thêm vào mỗi patch.

Gradient chảy qua thân transformer dùng chung. Cả hai loss cùng cải thiện bộ trọng số chia sẻ.

### Attention mask: văn bản nhân quả + ảnh hai chiều

Token văn bản phải nhân quả — không thể để một token văn bản attend tới văn bản tương lai, nếu không teacher forcing sẽ hỏng. Tuy vậy, patch ảnh biểu diễn một ảnh chụp tại một thời điểm; chúng nên attend lẫn nhau theo hai chiều bên trong cùng một khối ảnh.

Mask là:

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

Được hiện thực như một mask tam giác theo khối lúc huấn luyện và suy luận.

### Loss diffusion bên trong transformer

Loss diffusion là chuẩn mực: thêm nhiễu vào một patch ảnh, yêu cầu mô hình dự đoán nhiễu (hoặc patch sạch, tương đương). Phiên bản của Transfusion dùng flow matching — dự đoán trường vận tốc từ nhiễu tới sạch.

Trong huấn luyện:
1. Với mỗi patch ảnh x0, lấy mẫu một timestep (bước thời gian) t ngẫu nhiên.
2. Lấy mẫu nhiễu ε, tính xt = (1-t) * x0 + t * ε (nội suy tuyến tính cho flow matching).
3. Transformer dự đoán v_theta(xt, t); loss = MSE(v_theta(xt, t), ε - x0).
4. Lan truyền ngược song song với các loss NTP văn bản từ cùng chuỗi.

Lúc suy luận, sinh là:
- Token văn bản: lấy mẫu autoregressive chuẩn.
- Patch ảnh: vòng lấy mẫu diffusion (thường 10-30 bước) có điều kiện trên các token văn bản trước đó.

### MMDiT: biến thể của Stable Diffusion 3

Stable Diffusion 3 (Esser et al., tháng 3/2024) tung ra MMDiT (Multimodal Diffusion Transformer) gần cùng thời điểm với Transfusion. Hai kiến trúc là anh em.

Khác biệt chính của MMDiT:

- Trọng số riêng theo modality ở mỗi block. Mỗi block transformer có trọng số Q, K, V và MLP riêng cho token văn bản so với patch ảnh. Attention là chung (liên modality); mọi thứ còn lại riêng theo modality.
- Huấn luyện rectified flow. Một biến thể flow matching cụ thể với cách lấy mẫu đã biết và toán đơn giản hơn DDPM.
- Quy mô. MMDiT là backbone của SD3 (biến thể 2B và 8B tham số). Bài báo Transfusion mở rộng tới 7B.

Cả hai hội tụ về cùng một ý tưởng lõi: một transformer chạy NTP trên văn bản và diffusion trên biểu diễn ảnh liên tục.

### Vì sao cách này hơn kiểu Chameleon

Khoảng cách chất lượng giữa diffusion-liên tục và NTP-rời rạc trong sinh ảnh là đo được. Bài báo Transfusion báo cáo:

- Ở 7B tham số, hơn mô hình kiểu Chameleon cùng cỡ 3-5 điểm FID.
- Không cần huấn luyện tokenizer — encoder ảnh đơn giản hơn (chiếu tuyến tính vào hidden, giống lớp đầu vào của ViT).
- Suy luận có thể song song hoá denoising (khử nhiễu) patch ảnh, khác với token ảnh autoregressive.

Nhược điểm: Transfusion là mô hình hai loss, khiến động lực học huấn luyện khó hơn. Cần tinh chỉnh loss weights (trọng số mất mát). Sự lệch pha giữa NTP và diffusion có thể khiến một head lấn át.

### Những gì nối tiếp

Janus-Pro (Bài 12.15) tinh chỉnh ý tưởng của Transfusion bằng cách tách rời encoder thị giác cho hiểu và cho sinh — SigLIP cho một bên, VQ cho bên kia — trong khi chia sẻ thân transformer. Show-o (Bài 12.14) thay diffusion bằng discrete-diffusion (dự đoán có che). Gia đình sinh hợp nhất phân nhánh nhanh sau Transfusion.

Các VLM sản xuất năm 2026 có khả năng sinh ảnh — Gemini 3 Pro, GPT-5, đường sinh ảnh của Claude Opus 4.7 — gần như chắc chắn dùng một hậu duệ nào đó của gia đình này. Chi tiết là độc quyền.

```figure
cfg-guidance-scale
```

## Áp dụng

`code/main.py` dựng một Transfusion đồ chơi trên bài toán nhỏ cỡ MNIST:

- Caption văn bản là các chuỗi số nguyên ngắn mô tả một chữ số (0-9).
- Ảnh là lưới byte 4x4.
- Một cặp phép chiếu tuyến tính chia sẻ trọng số đóng vai transformer thay thế; loss NTP trên văn bản, loss MSE trên patch nhiễu.
- Vòng huấn luyện luân phiên hai loss, attention mask được viết tường minh.
- Sinh ra một caption văn bản và một ảnh 4x4 trong một lượt forward.

Transformer chỉ là đồ chơi. Phần ống dẫn hai loss, cách dựng attention mask và vòng suy luận mới là artifact thật.

## Đóng gói

Bài này tạo ra `outputs/skill-two-loss-trainer-designer.md`. Cho một tác vụ huấn luyện đa phương thức mới (văn bản + ảnh, văn bản + âm thanh, văn bản + video), nó thiết kế lịch hai loss (loss weights, hình dạng mask, block chia sẻ so với riêng theo modality) và gắn cờ rủi ro triển khai.

## Bài tập

1. Một mô hình kiểu Transfusion huấn luyện 70% token văn bản và 30% patch ảnh. Loss diffusion ảnh lớn hơn loss NTP văn bản khoảng 10x về độ lớn. Dùng loss weights nào để cân bằng chúng?

2. Hiện thực mask tam giác theo khối cho chuỗi: `[T, T, <image>, P, P, P, P, </image>, T]`. Đánh dấu mỗi phần tử 0 hoặc 1.

3. MMDiT có trọng số QKV riêng theo modality. Việc này thêm bao nhiêu overhead số tham số so với transformer chia sẻ hoàn toàn của Transfusion? Ở 7B tham số, có đáng không?

4. Sinh: cho một prompt văn bản, mô hình chạy NTP cho 50 token, rồi gặp `<image>`, rồi chạy diffusion trên 256 patch qua 20 bước khử nhiễu. Tổng cộng bao nhiêu lượt forward?

5. Đọc Mục 3 của bài báo SD3. Mô tả rectified flow và vì sao nó hội tụ trong ít bước suy luận hơn DDPM.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| Two-loss training | "NTP + diffusion" | Một transformer duy nhất tối ưu cả cross-entropy trên token văn bản và MSE trên patch ảnh liên tục trong cùng một bước gradient |
| Flow matching | "Rectified flow" | Biến thể diffusion dự đoán trường vận tốc từ nhiễu tới dữ liệu sạch; toán đơn giản hơn DDPM |
| MMDiT | "Multimodal DiT" | Kiến trúc của Stable Diffusion 3: attention chung, MLP và norm riêng theo modality |
| Block-triangular mask | "Causal text + bidirectional image" | Attention mask nhân quả xuyên văn bản nhưng hai chiều bên trong vùng ảnh |
| Continuous image representation | "No VQ" | Patch ảnh là vector số thực, không phải chỉ số codebook nguyên |
| Velocity prediction | "v-parameterization" | Đầu ra mạng là trường vận tốc giữa nhiễu và dữ liệu, không phải bản thân nhiễu |

## Đọc thêm

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)

---

[← Bài trước: Emu3: Dự đoán token kế tiếp cho sinh ảnh và sinh video](../../12-emu3-next-token-for-generation/docs/vi.md) · [Bài sau: Show-o và các mô hình hợp nhất khuếch tán rời rạc →](../../14-show-o-discrete-diffusion-unified/docs/vi.md)
