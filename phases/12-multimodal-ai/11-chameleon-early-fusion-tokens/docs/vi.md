# Chameleon và các mô hình đa phương thức chỉ dùng token với early fusion (hợp nhất sớm)

> Mọi VLM ta đã gặp tới giờ đều tách riêng ảnh và văn bản. Token thị giác đến từ một bộ mã hoá thị giác, chảy vào projector, rồi gặp văn bản bên trong LLM. Từ vựng thị giác và từ vựng văn bản không bao giờ trùng nhau. Chameleon (Meta, tháng 5/2024) đặt câu hỏi: nếu chúng trùng nhau thì sao? Huấn luyện một VQ-VAE biến ảnh thành chuỗi token rời rạc (discrete token) lấy từ một từ vựng chung. Mỗi tài liệu đa phương thức (modality, phương thức) giờ chỉ là một chuỗi — token văn bản và token hình ảnh xen kẽ, một loss tự hồi quy (autoregressive) duy nhất. Tác dụng phụ: mô hình có thể sinh đầu ra đa phương thức — xen kẽ token văn bản và token ảnh trong một lần suy luận. Bài này đọc luận đề early fusion và tự tay dựng một phiên bản đồ chơi từ đầu đến cuối.

**Type:** Build
**Languages:** Python (stdlib, VQ-VAE tokenizer + interleaved decoder)
**Prerequisites:** Giai đoạn 12, Bài 05, Giai đoạn 8 (AI sinh tạo)
**Time:** ~180 phút

## Mục tiêu học tập

- Giải thích vì sao một từ vựng chung + một loss duy nhất lại thay đổi những gì mô hình có thể làm.
- Mô tả cách VQ-VAE tokenize một ảnh thành chuỗi rời rạc tương thích với mục tiêu dự đoán token kế tiếp của transformer.
- Kể tên các mẹo ổn định huấn luyện của Chameleon: QK-Norm, vị trí dropout, thứ tự LayerNorm.
- So sánh Chameleon với cách tiếp cận Q-Former của BLIP-2 và nói khi nào mỗi bên là lựa chọn đúng.

## Vấn đề

Các VLM dựa trên adapter (LLaVA, BLIP-2, Qwen-VL) coi văn bản và ảnh là hai thứ khác nhau. Một token văn bản đi qua `embed(text_token)`; một ảnh đi qua `visual_encoder(image) → projector → ... pseudo_tokens`. Mô hình có hai đường đầu vào hoà vào nhau ở khoảng giữa.

Ba hệ quả:

1. LLM chỉ tiếp nhận được ảnh chứ không phát ra được ảnh. Đầu ra chỉ là văn bản.
2. Tài liệu đa phương thức (đoạn văn và ảnh xen kẽ, như trong một bài báo) rất khó xử lý — bạn hoặc phải phân tích đầu vào đa phương thức ở bên ngoài mô hình, hoặc phải nối chuỗi các lần sinh.
3. Lệch phân phối. Token thị giác và token văn bản nằm ở những vùng khác nhau của không gian ẩn, sinh ra những vấn đề căn chỉnh tinh vi.

Chameleon bác bỏ tiền đề đó: ảnh chỉ là chuỗi token rời rạc lấy từ một từ vựng chung. Huấn luyện mô hình trên các tài liệu xen kẽ, một loss, một decoder tự hồi quy, và bạn có được khả năng sinh đa phương thức miễn phí.

## Khái niệm

### VQ-VAE như một tokenizer hình ảnh (image tokenizer)

Tokenizer là một variational autoencoder được lượng tử hoá vector. Kiến trúc:

- Encoder: CNN + ViT ánh xạ ảnh thành một bản đồ đặc trưng không gian, ví dụ 32x32 đặc trưng có chiều 256.
- Codebook (sổ mã): một từ vựng học được gồm K vector (Chameleon dùng 8192), cũng có chiều 256.
- Lượng tử hoá: với mỗi đặc trưng không gian, tra mục codebook gần nhất theo khoảng cách L2. Thay đặc trưng liên tục bằng chỉ số nguyên.
- Decoder: CNN đưa các đặc trưng đã lượng tử hoá trở lại pixel.

Huấn luyện: loss tái tạo của VAE + commitment loss + codebook loss. Các chỉ số codebook tạo thành một bảng chữ cái rời rạc cho ảnh.

Với Chameleon: một ảnh trở thành 32*32 = 1024 token lấy từ từ vựng 8192. Ghép nối với token văn bản (từ từ vựng BPE của LLM, giả sử 32000). Từ vựng cuối cùng: 40192. Transformer thấy một chuỗi, một loss.

### Từ vựng chung

Từ vựng của Chameleon kết hợp token văn bản, token hình ảnh và các dấu phân cách phương thức. Mỗi token có một ID duy nhất. Lớp embedding đầu vào ánh xạ mọi ID thành một vector ẩn D chiều. Phép chiếu đầu ra ánh xạ vector ẩn trở lại logits của từ vựng. Softmax chọn token kế tiếp, bất kể phương thức nào.

Các dấu phân cách rất quan trọng: thẻ `<image>` và `</image>` bao quanh chuỗi token hình ảnh. Lúc sinh, nếu mô hình phát ra `<image>`, phần mềm phía sau biết rằng 1024 token tiếp theo là chỉ số VQ cần gửi tới decoder để dựng pixel.

### Sinh đa phương thức

Suy luận là dự đoán token kế tiếp trong từ vựng chung. Prompt ví dụ: "Vẽ một con mèo và mô tả nó." Chameleon phát ra:

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

Mô hình tự chọn thứ tự — nó có thể sinh ảnh rồi tới văn bản, văn bản rồi tới ảnh, hoặc xen kẽ. Cùng decoder, cùng loss.

So với các VLM adapter nơi việc sinh chỉ là văn bản. Chameleon mở lại câu hỏi về phương thức đầu ra của mô hình.

### Độ ổn định huấn luyện — QK-Norm, dropout, thứ tự LayerNorm

Huấn luyện early fusion không ổn định ở quy mô lớn. Bài báo của Chameleon ghi lại ba mẹo:

- QK-Norm. Áp LayerNorm lên các phép chiếu query và key bên trong attention, trước phép tích vô hướng. Ngăn độ lớn logit bùng nổ theo chiều sâu. Được nhiều mô hình lớn sau 2024 dùng.
- Vị trí dropout. Đặt dropout sau mỗi phép cộng residual, không chỉ sau attention và MLP. Cần điều tiết (regularization) nhiều hơn khi gradient từ token hình ảnh có thể lấn át.
- Thứ tự LayerNorm. Pre-LN trên nhánh residual (chuẩn), cộng thêm một LN nữa trên kết nối skip của block cuối. Ổn định dòng gradient ở lớp cuối.

Không có các mẹo này, huấn luyện Chameleon 34B tham số phân kỳ ở nhiều checkpoint. Có chúng, nó hội tụ. Công thức huấn luyện là một phần đóng góp quan trọng không kém kiến trúc.

### Giới hạn tái tạo của tokenizer

VQ-VAE là mất mát thông tin. Với 8192 mục codebook và 1024 token cho một ảnh 512x512, PSNR tái tạo chỉ đạt khoảng 26-28 dB. Đủ để sinh ảnh nhận ra được nhưng xấu hơn rõ rệt so với diffusion trong không gian liên tục (Stable Diffusion 3 đạt 32+ dB).

Tokenizer chính là nút cổ chai. Các tokenizer tốt hơn (MAGVIT-v2, IBQ, SBER-MoVQGAN) nâng giới hạn này lên. Emu3 (Bài 12.12) đạt chất lượng sinh ngang SDXL chỉ nhờ một tokenizer tốt hơn.

### Chameleon vs BLIP-2 / LLaVA

Chameleon (early fusion, từ vựng chung):
- Một loss, một decoder.
- Sinh đầu ra đa phương thức.
- Tokenizer là giới hạn chất lượng.
- Đắt: mỗi ảnh được sinh cần chạy decoder VQ-VAE trên đường suy luận.

BLIP-2 / LLaVA (late fusion – hợp nhất muộn, các tower tách riêng):
- Ảnh vào, chỉ văn bản ra.
- Tái dùng LLM đã huấn luyện trước.
- Không có nút cổ chai tokenizer cho phần hiểu.
- Rẻ: một lượt forward.

Chọn theo bài toán. Nếu bạn cần sinh ảnh, họ Chameleon. Nếu chỉ cần hiểu, VLM adapter đơn giản hơn và tái dùng được nhiều tính toán đã huấn luyện trước hơn.

### Fuyu và AnyGPT

Fuyu (Adept, 2023) là một cách tiếp cận liên quan: bỏ hẳn bộ mã hoá thị giác riêng, đưa các patch ảnh thô qua phép chiếu đầu vào của LLM như thể chúng là token, không cần tokenizer. Đơn giản hơn Chameleon nhưng mất khả năng sinh đầu ra từ từ vựng chung.

AnyGPT (Zhan et al., 2024) mở rộng Chameleon sang bốn phương thức: văn bản, hình ảnh, giọng nói, âm nhạc. Cùng mẹo VQ-VAE cho mỗi phương thức, dùng chung transformer. Sinh bất kỳ sang bất kỳ. Được nói kỹ hơn ở Bài 12.16.

```figure
vq-codebook
```

## Áp dụng

`code/main.py` dựng một mô hình early fusion đồ chơi từ đầu đến cuối:

- Một bộ lượng tử kiểu VQ-VAE bé xíu ánh xạ các patch 8x8 thành chỉ số codebook (K=16).
- Một từ vựng chung gồm (id văn bản 0..31) + (id hình ảnh 32..47) + (dấu phân cách 48, 49).
- Một decoder tự hồi quy đồ chơi (bảng bigram) huấn luyện trên caption tổng hợp + chuỗi token hình ảnh.
- Vòng lấy mẫu phát ra token văn bản và token hình ảnh xen kẽ khi có prompt.

Mã nguồn cố ý giữ transformer thật bé (bigram) để bạn lần theo dòng tín hiệu từ đầu đến cuối.

## Đóng gói

Bài này tạo ra `outputs/skill-tokenizer-vs-adapter-picker.md`. Cho một bản đặc tả sản phẩm (chỉ cần hiểu hay hiểu + sinh, chất lượng ảnh yêu cầu, ngân sách chi phí), nó chọn giữa họ Chameleon (early fusion) và họ LLaVA (late fusion), kèm lý do dựa trên các quy tắc ước lượng định lượng.

## Bài tập

1. Chameleon dùng K=8192 mục codebook và 1024 token cho một ảnh 512x512. Ước lượng tỷ lệ nén so với ảnh RGB 24-bit. Nó có mất mát thông tin không? Mất mát tới mức nào?

2. Một ảnh 4K (3840x2160) với cùng mật độ VQ-VAE đó sinh ra bao nhiêu token hình ảnh? Mô hình kiểu Chameleon có thể sinh một ảnh 4K trong một lần suy luận không? Thứ gì vỡ trước — context, chất lượng tokenizer, hay KV cache?

3. Cài QK-Norm bằng Python thuần. Cho một query và key 64 chiều, chỉ ra tích vô hướng trước và sau LayerNorm. Vì sao kiểm soát độ lớn lại quan trọng ở chiều sâu?

4. Đọc Mục 2.3 của Chameleon về độ ổn định huấn luyện. Mô tả đúng dạng hỏng mà bài báo quan sát ở 34B khi không có QK-Norm. Dấu hiệu "bùng nổ norm" trông như thế nào?

5. Mở rộng decoder đồ chơi để phát ra phản hồi đa phương thức khi chỉ có prompt văn bản. Đo xem mô hình chọn ảnh-trước hay văn bản-trước thường xuyên đến đâu với phân phối dữ liệu huấn luyện 60% văn bản-trước / 40% ảnh-trước.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| Early fusion | "Token hợp nhất" | Ảnh được chuyển thành token rời rạc dùng chung từ vựng của transformer ngay từ bước đầu |
| VQ-VAE | "Image tokenizer" | CNN + ViT + codebook ánh xạ ảnh thành chỉ số nguyên mà transformer có thể dự đoán |
| Shared vocabulary | "Một cuốn từ điển" | Một không gian ID token duy nhất phủ văn bản + hình ảnh + dấu phân cách phương thức |
| QK-Norm | "Bộ ổn định attention" | LayerNorm áp lên query và key trước tích vô hướng của chúng, ngăn norm bùng nổ |
| Mixed-modality generation | "Đầu ra văn bản + ảnh" | Suy luận tự động sinh token văn bản và token hình ảnh xen kẽ trong một lượt |
| Codebook size | "K mục" | Số vector rời rạc mà VQ-VAE có thể lượng tử hoá tới; đánh đổi nén lấy độ trung thực |
| Tokenizer ceiling | "Giới hạn tái tạo" | PSNR tốt nhất có thể đạt khi giải mã token VQ; chặn trên chất lượng ảnh của mô hình |

## Đọc thêm

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)

---

[← Bài trước: InternVL3: Tiền huấn luyện đa phương thức nguyên bản](../../10-internvl3-native-multimodal/docs/vi.md) · [Bài sau: Emu3: Dự đoán token kế tiếp cho sinh ảnh và sinh video →](../../12-emu3-next-token-for-generation/docs/vi.md)
