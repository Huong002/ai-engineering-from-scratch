# Janus-Pro: Bộ mã hoá tách rời cho mô hình đa phương thức hợp nhất

> Mô hình đa phương thức hợp nhất luôn có một căng thẳng không thể tránh. Understanding (hiểu) muốn đặc trưng semantic (ngữ nghĩa) — vector đầu ra của SigLIP hay DINOv2 giàu thông tin ở mức khái niệm. Generation (sinh) muốn các mã thân thiện với reconstruction (tái dựng) — các visual token (token thị giác) VQ ghép lại thành điểm ảnh sắc nét. Hai mục tiêu này không tương thích trong một encoder duy nhất. Janus (DeepSeek, tháng 10 năm 2024) và Janus-Pro (DeepSeek, tháng 1 năm 2025) cho rằng cách sửa là thôi cố gắng: tách rời hai encoder — decoupled encoder (bộ mã hoá tách rời). Dùng chung shared transformer (transformer chia sẻ) giữa các tác vụ, nhưng định tuyến hiểu qua SigLIP và sinh qua một VQ tokenizer. Ở mức 7B, Janus-Pro vượt DALL-E 3 trên GenEval trong khi ngang LLaVA trên MMMU. Bài này đọc lý do vì sao hai encoder hiệu quả ở chỗ một encoder thất bại.

**Type:** Build
**Languages:** Python (stdlib, dual-encoder routing + shared-body signal)
**Prerequisites:** Giai đoạn 12, Bài 13 (Transfusion), Giai đoạn 12, Bài 14 (Show-o)
**Time:** ~120 phút

## Mục tiêu học tập

- Giải thích vì sao một encoder chia sẻ duy nhất làm hy sinh chất lượng hiểu hoặc chất lượng sinh.
- Mô tả định tuyến của Janus-Pro: đặc trưng SigLIP ở phía đầu vào cho hiểu, token VQ ở cả đầu vào và đầu ra cho sinh.
- Truy vết việc mở rộng hỗn hợp dữ liệu khiến Janus-Pro thành công ở chỗ Janus thì không.
- So sánh kiến trúc decoupled (Janus-Pro), coupled-continuous (Transfusion) và coupled-discrete (Show-o).

## Vấn đề

Mô hình hợp nhất chia sẻ một thân transformer cho cả hiểu và sinh. Các nỗ lực trước (Chameleon, Show-o, Transfusion) đều dùng một tokenizer thị giác cho cả hai chiều. Tokenizer đó là một thoả hiệp:

- Tối ưu cho tái dựng (sinh): VQ-VAE bắt chi tiết điểm ảnh mịn nhưng tạo ra token có mạch lạc ngữ nghĩa yếu.
- Tối ưu cho ngữ nghĩa (hiểu): embedding SigLIP nhóm ảnh "mèo" gần các token "mèo" nhưng không cho phép tái dựng tốt.

Show-o và Transfusion trả giá cho điều này bằng một khoản thuế chất lượng thấy được ở một chiều. Janus-Pro hỏi: tại sao phải cần một tokenizer khi các tác vụ có nhu cầu khác nhau?

## Khái niệm

### Mã hoá thị giác tách rời

Kiến trúc Janus-Pro tách hai encoder ra:

- Đường hiểu. Ảnh đầu vào → SigLIP-SO400m → MLP 2 lớp → thân transformer.
- Đường sinh. Ảnh đầu vào (nếu điều kiện hoá trên một ảnh có sẵn) → VQ tokenizer → token ID → thân transformer.
- Sinh đầu ra. Token ảnh do transformer dự đoán → VQ decoder → điểm ảnh.

Thân transformer được chia sẻ. Mọi thứ phía trước và phía sau thân đều đặc thù theo tác vụ.

Đầu vào được phân biệt bằng định dạng prompt: thẻ `<understand>` định tuyến qua SigLIP; `<generate>` qua VQ. Hoặc việc định tuyến là ngầm định theo tác vụ.

### Vì sao cách này hiệu quả

Loss của hiểu nhận đặc trưng SigLIP, thứ mà tiền huấn luyện kiểu CLIP đã tinh chỉnh cho độ tương đồng ngữ nghĩa. Điểm benchmark nhận thức của mô hình tốt hơn Show-o / Transfusion vì đặc trưng đầu vào phù hợp với tác vụ hơn.

Loss của sinh nhận token VQ, thứ mà tokenizer đã tinh chỉnh cho tái dựng. Chất lượng ảnh tốt hơn Show-o vì mã VQ ghép lại thành điểm ảnh gọn gàng.

Thân transformer chia sẻ thấy hai phân phối đầu vào (SigLIP và VQ) và học cách làm việc với cả hai. Tuyên bố là: đủ dữ liệu + đủ tham số, thân sẽ hấp thu việc chuyển đổi.

### Mở rộng dữ liệu — Janus vs Janus-Pro

Janus (bản gốc, arXiv 2410.13848) giới thiệu việc tách rời nhưng ở quy mô nhỏ (1.3B tham số, dữ liệu hạn chế). Janus-Pro (arXiv 2501.17811) mở rộng:

- 7B tham số (so với 1.3B).
- 90M cặp ảnh-văn bản cho giai đoạn 1 (căn chỉnh), tăng từ 72M.
- 72M cho giai đoạn 2 (hợp nhất), tăng từ 26M.
- Thêm 200k mẫu chỉ dẫn sinh ảnh cho giai đoạn 3.

Kết quả: Janus-Pro-7B ngang LLaVA trên MMMU (60.3 so với ~58) và vượt DALL-E 3 trên GenEval (0.80 so với 0.67). Một mô hình mở, cạnh tranh ở cả hai phía của phổ hợp nhất.

### JanusFlow — biến thể rectified flow

JanusFlow (arXiv 2411.07975) thay đường sinh VQ bằng đường sinh rectified-flow (liên tục). Cách chia trở thành SigLIP-cho-hiểu + rectified-flow-cho-sinh. Trần chất lượng nâng cao hơn nữa. Kiến trúc vẫn là decoupled-encoders-shared-body.

### Nhiệm vụ của thân chia sẻ

Thân transformer xử lý một chuỗi hợp nhất nhưng với hai phân phối đầu vào. Nhiệm vụ của nó là:

- Với hiểu: tiêu thụ đặc trưng SigLIP + token văn bản → phát văn bản theo kiểu tự hồi quy.
- Với sinh: tiêu thụ token văn bản + (tuỳ chọn token VQ của ảnh) → phát token VQ của ảnh theo kiểu tự hồi quy.

Thân không có trọng số đặc thù modality nào theo từng block. Nó là transformer kiểu văn bản mà bạn vẫn thấy bên trong Qwen hay Llama, cộng thêm hai adapter đầu vào.

Điều thú vị là điều này có nghĩa thân của Janus-Pro có thể được khởi tạo từ một LLM đã tiền huấn luyện. Janus-Pro thật sự khởi tạo từ DeepSeek-MoE-7B. Lựa chọn đó quan trọng: LLM đóng góp năng lực suy luận mà các mô hình hợp nhất thuần from-scratch khó đạt tới.

### So sánh với InternVL-U

InternVL-U (Bài 12.10) là bản tiếp nối năm 2026. Nó kết hợp:

- Tiền huấn luyện đa phương thức bản địa (backbone InternVL3).
- Định tuyến decoupled-encoder (SigLIP vào, VQ + đầu diffusion ra).
- Hiểu + sinh + chỉnh sửa hợp nhất.

InternVL-U bao trùm lựa chọn kiến trúc của Janus-Pro vào một khung lớn hơn. Ý tưởng decoupled-encoder giờ là mặc định cho mô hình hợp nhất ở quy mô lớn.

### Hạn chế

Encoder tách rời thêm phức tạp kiến trúc. Hai tokenizer phải huấn luyện, hai đường đầu vào phải bảo trì, hai bộ kiểu lỗi. Với sản phẩm không cần sinh, Janus-Pro là quá tay — hãy chọn một mô hình hiểu thuộc họ LLaVA.

Với sản phẩm không cần hiểu, Janus-Pro là quá mức — hãy chọn mô hình Stable Diffusion 3 / Flux.

Với sản phẩm cần cả hai, Janus-Pro hiện là kiến trúc mở tham chiếu.

```figure
l5-janus-decouple
```

## Áp dụng

`code/main.py` mô phỏng định tuyến của Janus-Pro:

- Hai encoder giả lập: kiểu SigLIP (tạo vector ngữ nghĩa 256 chiều) và kiểu VQ (tạo mã số nguyên).
- Một router prompt chọn encoder dựa trên thẻ tác vụ.
- Một thân chia sẻ (đóng vai) xử lý chuỗi token bất kể encoder nào tạo ra chúng.
- Một lịch lấy mẫu có trọng số chuyển từ giai đoạn 1 (căn chỉnh) sang giai đoạn 3 (tinh chỉnh theo chỉ dẫn).

In ra các đường định tuyến cho 3 ví dụ: hỏi đáp ảnh, T2I, chỉnh sửa ảnh.

## Đóng gói

Bài này tạo ra `outputs/skill-decoupled-encoder-picker.md`. Khi đưa một sản phẩm muốn sinh + hiểu hợp nhất ở chất lượng gần mức tiên tiến, nó chọn Janus-Pro, JanusFlow hoặc InternVL-U kèm khuyến nghị quy mô dữ liệu cụ thể.

## Bài tập

1. Janus-Pro-7B vượt DALL-E 3 trên GenEval. Giải thích vì sao một mô hình mở 7B có thể ngang một mô hình độc quyền tiên tiến về sinh nhưng không ngang về hiểu.

2. Cài đặt một hàm router: cho văn bản prompt, phân loại thành `understand` hoặc `generate`. Bạn xử lý các prompt mơ hồ như "mô tả rồi phác hoạ" thế nào?

3. JanusFlow thay đường VQ bằng rectified flow. Thân transformer giờ phát ra gì, và loss thay đổi ra sao?

4. Đề xuất một tác vụ thứ tư mà kiến trúc Janus-Pro có thể xử lý với thêm một encoder tách rời. Ví dụ: phân đoạn ảnh (kiểu DINO), độ sâu (kiểu MiDaS).

5. Đọc Mục 4.2 của Janus-Pro về mở rộng dữ liệu. Giai đoạn dữ liệu nào đóng góp nhiều nhất cho mức tăng chất lượng T2I so với Janus?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| Decoupled encoding | "Hai encoder thị giác" | Tokenizer hoặc encoder riêng cho mỗi chiều: ngữ nghĩa cho hiểu, tái dựng cho sinh |
| Shared body | "Một transformer" | Một transformer duy nhất xử lý đầu ra của bất kỳ encoder nào; không có trọng số đặc thù modality |
| SigLIP for understanding | "Đặc trưng ngữ nghĩa" | Vision tower họ CLIP cung cấp đặc trưng khái niệm giàu nhưng tái dựng kém |
| VQ for generation | "Mã tái dựng" | Token lượng tử hoá vector giải mã sạch trở lại thành điểm ảnh |
| JanusFlow | "Biến thể rectified-flow" | Janus-Pro với đầu sinh flow-matching liên tục thay cho VQ |
| Routing tag | "Thẻ tác vụ" | Dấu đánh dấu trong prompt (`<understand>` / `<generate>`) chọn encoder đầu vào |

## Đọc thêm

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)

---

[← Bài trước: Show-o và các mô hình hợp nhất khuếch tán rời rạc](../../14-show-o-discrete-diffusion-unified/docs/vi.md) · [Bài sau: MIO và các mô hình đa phương thức any-to-any (bất kỳ sang bất kỳ) truyền trực tiếp →](../../16-mio-any-to-any-streaming/docs/vi.md)
