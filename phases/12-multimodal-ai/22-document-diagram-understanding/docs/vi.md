# Hiểu tài liệu và sơ đồ

> Tài liệu không phải ảnh chụp. Một PDF, bài báo khoa học, hoá đơn hay biểu mẫu viết tay có bố cục (layout), bảng, sơ đồ, chú thích chân trang, tiêu đề mục và cấu trúc ngữ nghĩa mà việc hiểu ảnh đơn thuần không nắm bắt được. Bộ công cụ thời trước VLM là một pipeline: Tesseract OCR + LayoutLMv3 + các heuristic trích bảng (table extraction). Làn sóng VLM thay thế bằng các mô hình không dùng OCR — Donut (2022), Nougat (2023), DocLLM (2023) — xuất trực tiếp markup có cấu trúc. Đến 2026, biên giới chỉ còn là "đưa ảnh trang vào Claude Opus 4.7 ở 2576px native", và đầu ra markup có cấu trúc có luôn. Bài này đọc vòng cung ba thời kỳ của document AI.

**Type:** Build
**Languages:** Python (stdlib, layout-aware document parser skeleton)
**Prerequisites:** Giai đoạn 12, Bài 05 (LLaVA), Giai đoạn 5 (NLP)
**Time:** ~180 phút

## Mục tiêu học tập

- Giải thích ba thời kỳ của document AI: pipeline OCR, không dùng OCR, VLM-native.
- Mô tả ba luồng đầu vào của LayoutLMv3: văn bản, bố cục (bbox), các mảnh ảnh, với cơ chế che (masking) thống nhất.
- So sánh Donut (không OCR, ảnh → markup), Nougat (bài báo khoa học → LaTeX), DocLLM (sinh có nhận biết bố cục), PaliGemma 2 (VLM-native).
- Chọn mô hình tài liệu cho một bài toán mới (hoá đơn, bài báo khoa học, biểu mẫu viết tay, hoá đơn tiếng Trung).

## Vấn đề

"Hiểu PDF này" khó hơn vẻ ngoài. Thông tin nằm ở:

- Nội dung văn bản (90% tín hiệu).
- Bố cục (tiêu đề mục, chú thích chân trang, cột bên, dạng hai cột).
- Bảng (hàng, cột, ô gộp).
- Hình và sơ đồ.
- Ghi chú viết tay.
- Phông chữ và kiểu chữ (tiêu đề so với thân bài).

OCR thô đổ ra văn bản và làm mất phần còn lại. Một hệ thống quan tâm đến hoá đơn cần biết "Total: $1,245" đến từ góc dưới bên phải, không phải từ chú thích chân trang.

## Khái niệm

### Thời kỳ 1 — pipeline OCR (trước 2021)

Bộ công cụ kinh điển:

1. PDF → ảnh từng trang.
2. Tesseract (hoặc OCR thương mại) trích văn bản kèm khung bao (bounding box) cho từng từ.
3. Bộ phân tích bố cục nhận diện các khối (tiêu đề, bảng, đoạn văn).
4. Bộ nhận dạng cấu trúc bảng phân tích các bảng.
5. Luật miền + regex trích các trường.

Hiệu quả với văn bản in sạch. Hỏng với chữ viết tay, bản scan bị nghiêng, bảng phức tạp, chữ không phải tiếng Anh. Mỗi kiểu lỗi lại cần một nhánh xử lý ngoại lệ riêng.

### TrOCR (2021)

TrOCR (Li et al., arXiv:2109.10282) thay CNN-CTC kinh điển của Tesseract bằng transformer encoder-decoder huấn luyện trên ảnh chữ tổng hợp + thật. Thắng rõ ở chữ viết tay và chữ đa ngôn ngữ. Vẫn là pipeline (phát hiện rồi TrOCR rồi bố cục), nhưng bước OCR cải thiện rất mạnh.

### Thời kỳ 2 — không dùng OCR (2022-2023)

Các mô hình không dùng OCR đầu tiên nói rằng: bỏ hẳn bước phát hiện, ánh xạ điểm ảnh trực tiếp sang đầu ra có cấu trúc.

Donut (Kim et al., arXiv:2111.15664):
- Transformer encoder-decoder, encoder là Swin-B.
- Đầu ra là JSON cho hiểu biểu mẫu, markdown cho tóm tắt, hoặc bất kỳ schema riêng cho tác vụ.
- Không OCR, không bố cục, không phát hiện.

Nougat (Blecher et al., arXiv:2308.13418):
- Huấn luyện riêng cho bài báo khoa học.
- Đầu ra là LaTeX / markdown.
- Xử lý công thức, bố cục nhiều cột, hình.
- Mô hình mà mọi trình phân tích arXiv đều gọi.

Đây là các chuyên gia, không phải mô hình tổng quát. Donut trên bài báo khoa học thì hỏng; Nougat trên hoá đơn thì hỏng.

### LayoutLMv3 (2022)

Một hướng khác. LayoutLMv3 (Huang et al., arXiv:2204.08387) giữ OCR nhưng thêm hiểu bố cục:

- Ba luồng đầu vào: token văn bản OCR, khung bao 2D cho từng token, các mảnh ảnh.
- Mục tiêu huấn luyện có che trên cả ba phương thức (che văn bản, che mảnh ảnh, che bố cục).
- Hạ nguồn: phân loại, trích thực thể, hỏi đáp bảng.

LayoutLMv3 là đỉnh của document understanding (hiểu tài liệu) dựa trên OCR. Mạnh ở biểu mẫu và hoá đơn. Cần OCR ở phía trước. Độ chính xác tốt nhất thời trước VLM trên các bộ chuẩn tài liệu.

### DocLLM (2023)

DocLLM (Wang et al., arXiv:2401.00908) là người anh em sinh của LayoutLM. Sinh câu trả lời tự do có điều kiện trên các token bố cục. Tốt hơn cho hỏi đáp trên tài liệu; vẫn phụ thuộc đầu vào OCR.

### Thời kỳ 3 — VLM-native (2024+)

VLM năm 2024 đã đủ tốt để thay thế hoàn toàn pipeline. Đưa toàn bộ ảnh trang ở độ phân giải cao cho VLM, đặt câu hỏi, nhận câu trả lời.

- LLaVA-NeXT 336-tile AnyRes dùng được cho tài liệu nhỏ.
- Qwen2.5-VL với độ phân giải động xử lý 2048+ điểm ảnh một cách native.
- Claude Opus 4.7 hỗ trợ tài liệu 2576px.
- PaliGemma 2 (tháng 4/2025) huấn luyện riêng cho tài liệu + chữ viết tay.

Khoảng cách giữa VLM-native và pipeline OCR khép lại nhanh chóng. Đến 2026, VLM-native thắng ở:

- Chữ trong cảnh (viết tay + in, nhiều loại chữ trộn lẫn).
- Bảng phức tạp có ô gộp.
- Công thức toán nằm trong văn bản.
- Hình có chú thích văn bản.

Pipeline OCR vẫn thắng ở:

- Khối lượng scan thuần ở quy mô cực lớn, nơi độ trễ mỗi trang quan trọng.
- Độ tin cậy của pipeline (lỗi tất định so với ảo giác của VLM).
- Môi trường bị quản lý, yêu cầu đầu ra OCR có thể kiểm toán.

### Biên giới Claude 4.7 / GPT-5

Với đầu vào native 2576 điểm ảnh, các VLM tiên phong hiểu tài liệu với độ chính xác gần mức con người. Các con số benchmark đầu 2026:

- DocVQA: Claude 4.7 ~95.1, PaliGemma 2 ~88.4, Nougat ~77.3, LayoutLMv3 dạng pipeline ~83.
- ChartQA: Claude 4.7 ~92.2, GPT-4V ~78.
- VisualMRC: Claude 4.7 ~94.

Khoảng cách với mô hình đóng chủ yếu nằm ở độ phân giải và quy mô LLM nền. Các mô hình mở 7B kém vài điểm nhưng đang đuổi kịp.

### Công thức toán và đầu ra LaTeX

Bài báo khoa học cần đầu ra LaTeX chính xác cho công thức. Nougat được huấn luyện đúng cho việc này. Các VLM huấn luyện với đích LaTeX (Qwen2.5-VL-Math, các biến thể của Nougat) tạo ra LaTeX dùng được. Không huấn luyện LaTeX rõ ràng thì VLM cho ra bản chép đọc được nhưng thiếu chính xác.

Với pipeline bài báo khoa học năm 2026: nối Nougat trên PDF, rồi dùng VLM cho các trang khó.

### Chữ viết tay

Vẫn là bài toán con khó nhất. Trộn in + viết tay (ghi chú bác sĩ, biểu mẫu đã điền) là chỗ pipeline OCR vẫn thắng VLM về chi phí. Các VLM chuyên chữ viết tay đang tiến bộ (Claude 4.7, PaliGemma 2).

### Công thức 2026

Với một dự án document-AI mới:

- Hoá đơn in thuần ở quy mô lớn: LayoutLMv3 + luật, tiết kiệm chi phí.
- Tài liệu hỗn hợp (khoa học + viết tay + biểu mẫu): VLM-native (PaliGemma 2 hoặc Qwen2.5-VL).
- Nạp toàn bộ arXiv: Nougat cho toán, VLM cho hình.
- Bị quản lý: pipeline OCR + VLM kiểm tra chéo.

```figure
mm-doc-layout
```

## Áp dụng

`code/main.py`:

- Một tokenizer nhận biết bố cục đồ chơi: nhận các cặp (text, bbox), tạo đầu vào kiểu LayoutLMv3.
- Một bộ sinh schema tác vụ kiểu Donut: mẫu JSON cho biểu mẫu.
- So sánh ngân sách token mỗi trang giữa pipeline OCR, Donut, Nougat và VLM-native.

## Đóng gói

Bài này tạo ra `outputs/skill-document-ai-stack-picker.md`. Cho một dự án document-AI (miền, quy mô, chất lượng, quản lý), nó chọn giữa pipeline OCR, chuyên gia không OCR và VLM-native.

## Bài tập

1. Dự án của bạn là 10 triệu hoá đơn mỗi ngày. Bộ công cụ nào giảm chi phí mỗi trang mà không mất độ chính xác?

2. Vì sao LayoutLMv3 vượt các VLM thuần CLIP ở hỏi đáp biểu mẫu nhưng lại kém ở chữ trong cảnh? Luồng bbox phải đánh đổi điều gì?

3. Nougat sinh LaTeX. Đề xuất một ca kiểm thử mà đầu ra VLM-native thắng Nougat về độ trung thực LaTeX, và một ca Nougat thắng.

4. Đọc bài PaliGemma 2 (Google, 2024). Bổ sung dữ liệu huấn luyện then chốt nào đã nâng độ chính xác tài liệu so với PaliGemma 1?

5. Thiết kế một hệ hybrid an toàn về mặt quản lý: pipeline OCR là chính, VLM là kiểm tra chéo phụ. Bạn xử lý bất đồng thế nào?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| OCR pipeline | "kiểu Tesseract" | Bộ công cụ theo từng bước: phát hiện -> OCR -> bố cục -> luật; tất định, dễ vỡ |
| OCR-free | "kiểu Donut" | Transformer ảnh-sang-đầu-ra bỏ qua OCR rõ ràng; một mô hình duy nhất |
| Layout-aware | "LayoutLM" | Đầu vào gồm toạ độ bbox cho từng token; che thống nhất trên mọi phương thức |
| VLM-native | "VLM tiên phong" | Đưa thẳng ảnh trang cho VLM Claude/GPT/Qwen ở độ phân giải cao; không pipeline |
| DocVQA | "bộ chuẩn tài liệu" | Chuẩn VQA tài liệu; điểm được trích dẫn nhiều nhất |
| Markup output | "LaTeX / MD" | Định dạng đầu ra có cấu trúc thay cho văn bản tự do; mở đường cho tự động hoá hạ nguồn |

## Đọc thêm

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)

---

[← Bài trước: VLA có thân thể: RT-2, OpenVLA, π0, GR00T](../../21-embodied-vlas-openvla-pi0-groot/docs/vi.md) · [Bài sau: ColPali và RAG tài liệu thuần thị giác →](../../23-colpali-vision-native-rag/docs/vi.md)
