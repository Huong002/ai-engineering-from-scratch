# ColPali và RAG tài liệu thuần thị giác

> RAG truyền thống phân tích PDF thành văn bản, chia thành các đoạn, nhúng đoạn, lưu vector. Mỗi bước đều làm mất tín hiệu: OCR bỏ dữ liệu biểu đồ, việc chia đoạn phá vỡ các hàng của bảng, embedding văn bản phớt lờ hình vẽ. ColPali (Faysse và cộng sự, tháng 7/2024) đặt câu hỏi đơn giản hơn: tại sao phải trích xuất văn bản làm gì? Cứ nhúng thẳng ảnh trang (page image) bằng PaliGemma, dùng late interaction (tương tác muộn) kiểu ColBERT để truy hồi (retrieval), và giữ lại toàn bộ tín hiệu về bố cục, hình vẽ, phông chữ và định dạng mà tài liệu mang theo. Kết quả công bố: độ chính xác đầu-cuối tốt hơn 20-40% so với RAG văn bản trên các tài liệu giàu hình ảnh. ColQwen2, ColSmol và VisRAG đã mở rộng hướng tiếp cận này. Bài học này đọc luận đề RAG thuần thị giác và tự xây một bộ chỉ mục nhỏ kiểu ColPali.

**Type:** Build
**Languages:** Python (stdlib, multi-vector indexer + MaxSim scorer)
**Prerequisites:** Giai đoạn 11 (Kỹ nghệ LLM — kiến thức cơ bản về RAG), Giai đoạn 12, Bài 05 (LLaVA)
**Time:** ~180 phút

## Mục tiêu học tập

- Phân biệt truy hồi bi-encoder (một vector cho mỗi tài liệu) với truy hồi late interaction (nhiều vector cho mỗi tài liệu).
- Mô tả phép toán MaxSim của ColBERT và cách ColPali mở rộng nó từ token văn bản sang các mảnh ảnh.
- Xây một bộ chỉ mục nhỏ kiểu ColPali: trang → patch embedding (embedding mảnh ảnh) → MaxSim trên embedding của token truy vấn → top-k trang.
- So sánh ColPali + generator Qwen2.5-VL với text-RAG + GPT-4 trên một ca dùng hóa đơn / báo cáo tài chính.

## Vấn đề

RAG văn bản trên PDF ném đi phần lớn tài liệu. Mức tăng doanh thu quý 3 trong một báo cáo tài chính thường nằm trong biểu đồ; kết luận của một báo cáo y khoa nằm trong hình ảnh có chú thích; khối chữ ký của một hợp đồng pháp lý là sự kiện về bố cục, không phải sự kiện về văn bản.

Pipeline text-RAG:

1. PDF → văn bản qua OCR (pipeline OCR) / pdftotext.
2. Văn bản → các đoạn 300-500 token.
3. Đoạn → embedding bi-encoder (một vector).
4. Truy vấn của người dùng → embedding → độ tương đồng cosine → top-k đoạn.
5. Đoạn + truy vấn → LLM.

Năm bước đều gây mất mát. Biểu đồ không được nắm bắt. Bảng bị cắt vụn qua các đoạn. Bố cục nhiều cột bị làm phẳng. Chú thích trên hình biến mất.

Cách sửa của ColPali: bỏ qua OCR, nhúng thẳng ảnh trang. Dùng late interaction kiểu ColBERT để truy hồi, nhờ đó mô hình có thể chú ý tới các mảnh ảnh chi tiết ngay tại thời điểm truy vấn.

## Khái niệm

### ColBERT (2020)

ColBERT (Khattab & Zaharia, arXiv:2004.12832) là một phương pháp truy hồi văn bản. Thay vì một vector cho mỗi tài liệu, nó tạo một vector cho mỗi token. Khi truy vấn:

- Các token truy vấn có embedding riêng (N_q vector).
- Các token tài liệu có embedding (N_d vector, thường được cache).
- Điểm = tổng trên các token truy vấn của giá trị lớn nhất trên các token tài liệu của độ tương đồng cosine: Σ_i max_j cos(q_i, d_j).

Đây chính là phép toán MaxSim. Mỗi token truy vấn "chọn" token tài liệu khớp nhất với nó. Điểm cuối cùng là tổng.

Ưu: recall mạnh, xử lý được ngữ nghĩa ở mức từ. Nhược: N_d vector cho mỗi tài liệu, lưu trữ tốn kém.

### ColPali

ColPali (Faysse và cộng sự, arXiv:2407.01449) áp dụng khuôn mẫu ColBERT cho hình ảnh.

- Mỗi trang được PaliGemma (ViT + ngôn ngữ) mã hoá thành patch embedding: N_p vector cho mỗi trang.
- Mỗi truy vấn của người dùng (văn bản) được mã hoá thành embedding của token truy vấn: N_q vector.
- Điểm = Σ_i max_j cos(q_i, p_j), tức là MaxSim trên các token văn bản truy vấn và các mảnh ảnh của trang.
- Truy hồi top-k trang theo tổng điểm.

Ở thời điểm nạp tài liệu: nhúng mọi trang bằng PaliGemma, lưu toàn bộ patch embedding. Khi truy vấn: nhúng các token truy vấn, tính MaxSim với mọi embedding trang đã lưu, trả về top-k trang.

Ưu: đầu-cuối đánh bại text-RAG 20-40% trên tài liệu giàu hình ảnh. Mỗi vector mảnh ảnh nắm được bố cục và nội dung cục bộ.

Nhược: N_p mảnh × float 4 byte × vector D chiều cho mỗi trang = lưu trữ tăng rất nhanh. Giảm nhẹ bằng lượng tử hoá PQ / OPQ.

### ColQwen2 và ColSmol

ColQwen2 (illuin-tech, 2024-2025) đổi PaliGemma thành Qwen2-VL. Encoder nền tốt hơn, truy hồi tốt hơn.

ColSmol là biến thể quy mô nhỏ hơn cho dùng cục bộ / ở biên. Một bộ truy hồi ColSmol khoảng 1B tham số chạy được trên GPU tiêu dùng.

### VisRAG

VisRAG (Yu và cộng sự, arXiv:2410.10594) là một biến thể khác: thay vì MaxSim trên các mảnh ảnh, gộp mỗi trang thành một vector duy nhất bằng VLM rồi truy hồi bi-encoder. Lập chỉ mục nhanh hơn + lưu trữ nhỏ hơn, recall yếu hơn.

Đánh đổi chất lượng so với chi phí: ColPali cho chất lượng, VisRAG cho quy mô.

### M3DocRAG

M3DocRAG (Cho và cộng sự, arXiv:2411.04952) mở rộng truy hồi đa phương thức sang suy luận nhiều trang, nhiều tài liệu. Nó truy hồi các trang trong nhiều tài liệu, rồi ghép thành ngữ cảnh nhiều trang cho VLM.

### ViDoRe — bộ đánh giá

Bộ đánh giá đi kèm của ColPali. Visual Document Retrieval Evaluation. Các bài toán gồm báo cáo tài chính, bài báo khoa học, tài liệu hành chính, hồ sơ y khoa, sách hướng dẫn. Chỉ số: nDCG@5.

ColPali-v1 đạt khoảng 80% nDCG@5 trên ViDoRe; text-RAG trên cùng bộ tài liệu đạt khoảng 50-60%.

### Pipeline RAG đầu-cuối

Với một RAG thuần thị giác:

1. Nạp: PDF → ảnh trang → PaliGemma mã hoá → lưu toàn bộ patch embedding.
2. Truy vấn: văn bản người dùng → embedding token truy vấn → MaxSim với mọi trang đã lập chỉ mục → top-k trang.
3. Sinh: ảnh của top-k trang + truy vấn → VLM (Qwen2.5-VL hoặc Claude) → câu trả lời.

Không có OCR ở bất kỳ đâu. Hình vẽ, biểu đồ, phông chữ, bố cục đều chảy vào câu trả lời.

### Tính toán lưu trữ

Một báo cáo tài chính 50 trang với 729 mảnh mỗi trang và embedding 128 chiều:

- ColPali: 50 * 729 * 128 * 4 bytes = ~18 MB thô, ~4 MB sau PQ.
- Text-RAG: 50 đoạn * 768 chiều * 4 bytes = ~150 kB.

ColPali tốn lưu trữ gấp khoảng 30 lần cho mỗi tài liệu. Ở quy mô lớn, OPQ / PQ kéo xuống còn khoảng 5-10 lần, thường chấp nhận được.

### Khi text-RAG vẫn thắng

- Tài liệu thuần văn bản, không có tín hiệu bố cục (bài wiki, log trò chuyện). Text-RAG đơn giản hơn và rẻ lưu trữ hơn.
- Kho lưu trữ hàng triệu trang, nơi lưu trữ chiếm phần lớn chi phí.
- Yêu cầu pháp lý nghiêm ngặt buộc phải có văn bản OCR trích xuất được song song với truy hồi.

Với mọi thứ còn lại trong năm 2026 — báo cáo tài chính, bài báo khoa học, hợp đồng pháp lý, hồ sơ y khoa, tài liệu UX — RAG thuần thị giác thắng.

```figure
mm-maxsim
```

## Áp dụng

`code/main.py`:

- Bộ mã hoá mảnh ảnh đồ chơi: ánh xạ một "trang" (lưới nhỏ các vector đặc trưng) thành một mảng patch embedding.
- Bộ chấm điểm MaxSim: tính điểm kiểu ColBERT giữa tập embedding token truy vấn và tập mảnh ảnh của trang.
- Lập chỉ mục cho 5 trang đồ chơi, chạy 3 truy vấn, trả về top-k kèm điểm.

## Đóng gói

Bài học này tạo ra `outputs/skill-vision-rag-designer.md`. Cho một dự án RAG tài liệu, nó chọn ColPali / ColQwen2 / VisRAG / text-RAG và tính toán kích thước lưu trữ.

## Bài tập

1. Một báo cáo thường niên 200 trang với 729 mảnh mỗi trang, emb 128 chiều, float 4 byte. Tính lưu trữ thô và lưu trữ sau nén PQ (8x).

2. MaxSim là Σ_i max_j cos(q_i, p_j). Tổng này nắm bắt được điều gì mà một độ tương đồng trung bình đơn giản không nắm được?

3. ColPali lập chỉ mục trang dưới dạng tập mảnh ảnh. Điều gì thay đổi nếu ta lập chỉ mục ở mức từ (như ColBERT làm)? Các đánh đổi?

4. Thiết kế pipeline đầu-cuối cho một kho 1 triệu trang với ngân sách độ trễ 500ms mỗi truy vấn. Chọn ColQwen2 / VisRAG và lý giải.

5. Đọc M3DocRAG (arXiv:2411.04952). Mô tả kiểu chú ý nhiều trang và nó khác truy hồi ColPali một trang ở chỗ nào.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| Late interaction | "Kiểu ColBERT" | Truy hồi bằng embedding theo từng token hoặc từng mảnh ảnh + MaxSim, không phải một vector tài liệu duy nhất |
| MaxSim | "Max trên các mảnh ảnh" | Với mỗi token truy vấn, chọn token tài liệu có độ tương đồng cao nhất; rồi cộng dồn trên toàn bộ truy vấn |
| Bi-encoder | "Một vector" | Một vector cho mỗi tài liệu; nhanh hơn nhưng mất độ chi tiết |
| Multi-vector | "Nhiều vector mỗi tài liệu" | Lưu N_p vector cho mỗi tài liệu / trang; chi phí lưu trữ tăng nhưng recall tốt hơn |
| Patch embedding | "Đặc trưng trang" | Một vector cho mỗi mảnh ảnh do encoder VLM tạo ra, được cache theo trang |
| ViDoRe | "Bộ đánh giá tài liệu thị giác" | Bộ benchmark của ColPali cho truy hồi tài liệu thị giác |
| PQ quantization | "Lượng tử hoá tích" | Phép nén giữ được độ tương đồng vector trong khi thu nhỏ lưu trữ khoảng 8 lần |

## Đọc thêm

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)

---

[← Bài trước: Hiểu tài liệu và sơ đồ](../../22-document-diagram-understanding/docs/vi.md) · [Bài sau: RAG đa phương thức và Truy hồi liên phương thức →](../../24-multimodal-rag-cross-modal/docs/vi.md)
