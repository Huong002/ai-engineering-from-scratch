# RAG đa phương thức và Truy hồi liên phương thức

> RAG tài liệu thuần thị giác chỉ là một lát cắt. RAG đa phương thức trong sản xuất rộng hơn nhiều — truy hồi đa phương thức (multimodal retrieval) xuyên text, ảnh, âm thanh và video cho các quy trình như lập kế hoạch du lịch ("tìm cho tôi một quán brunch thuần chay yên tĩnh, nhiều ánh sáng tự nhiên"), phân loại y tế ("vết thương nào khớp với ảnh + ghi chú này"), thương mại điện tử ("bộ đồ tương tự selfie này, đúng cỡ của tôi"), và dịch vụ hiện trường ("chẩn đoán tiếng động cơ này cùng ảnh bộ phận"). Ba khảo sát năm 2025 — Abootorabi et al., Mei et al., Zhao et al. — đã hệ thống hoá các bài toán con: truy hồi liên phương thức, hợp nhất kết quả truy hồi, neo sinh văn bản vào dữ liệu (grounding), đánh giá đa phương thức. Bài này đọc các khảo sát đó và thiết kế một pipeline sản xuất.

**Type:** Build
**Languages:** Python (stdlib, cross-modal retriever with fusion + grounded generator)
**Prerequisites:** Giai đoạn 12, Bài 23 (ColPali), Giai đoạn 11 (kiến thức cơ bản về RAG)
**Time:** ~180 phút

## Mục tiêu học tập

- Thiết kế truy hồi liên phương thức: text → ảnh, ảnh → text, âm thanh → video, v.v.
- So sánh ba chiến lược hợp nhất: hợp nhất điểm, hợp nhất dựa trên attention, hợp nhất MoE.
- Giải thích việc neo sinh văn bản vào dữ liệu: "trích dẫn nguồn" trông thế nào khi nguồn là hỗn hợp các phương thức.
- Kể tên ba khảo sát RAG đa phương thức chuẩn của năm 2025 và cách phân loại bài toán con của chúng.

## Vấn đề

RAG đơn phương thức là một mẫu đã được giải quyết: embed truy vấn, embed các khối (chunk), truy hồi, nhồi vào LLM. RAG đa phương thức cần:

1. Nhiều đầu truy hồi (mỗi phương thức cần embedding trong một không gian tương thích).
2. Hợp nhất kết quả truy hồi giữa các phương thức (chỉ mục hợp nhất — fused index).
3. Neo sinh văn bản có trích dẫn nguồn xuyên các phương thức.
4. Thang đo đánh giá bao phủ tín hiệu liên phương thức.

Cả ba khảo sát năm 2025 đều đi tới cùng một cách phân loại.

## Khái niệm

### Truy hồi liên phương thức

Truy hồi tài liệu ở phương thức B khi có truy vấn ở phương thức A. Có ba mẫu:

1. Không gian embedding chia sẻ. CLIP và CLAP tạo embedding text + ảnh / text + âm thanh trong một không gian chung. Độ tương đồng cosine giữa các phương thức dùng được trực tiếp. Bị giới hạn ở các cặp mà CLIP đã huấn luyện.

2. Encoder riêng từng phương thức + mô-đun dịch. Encoder text + encoder ảnh + một mô-đun dịch nhỏ ánh xạ giữa các không gian. Sen2Sen của Gupta et al. và các thiết kế khác năm 2024. Linh hoạt nhưng tăng độ phức tạp.

3. VLM làm encoder. Dùng hidden state của VLM làm biểu diễn truy hồi. Phương thức nào VLM hỗ trợ đều dùng được. Chất lượng cao hơn, tốn kém hơn.

Lựa chọn: CLIP / SigLIP 2 cho text+ảnh; CLAP cho text+âm thanh; hidden state của VLM cho liên phương thức ở chất lượng biên.

### Chiến lược hợp nhất

Bạn truy hồi được 10 kết quả: 5 ảnh, 3 đoạn text, 2 clip âm thanh. Gộp chúng lại thế nào?

Hợp nhất điểm (rẻ nhất). Mỗi phương thức có retriever riêng, mỗi cái trả về điểm. Chuẩn hoá điểm trong từng phương thức rồi cộng lại. Đơn giản, thường hiệu quả.

Hợp nhất dựa trên attention. Nối tất cả mục đã truy hồi, để một mạng attention nhỏ gán trọng số cho chúng. Cần huấn luyện.

Hợp nhất MoE. Mạng gating định tuyến tới các expert chuyên theo phương thức. Các loại truy vấn khác nhau định tuyến khác nhau — câu hỏi thị giác đặt trọng số cao hơn cho ảnh.

Mặc định khi triển khai: hợp nhất điểm với một thiên lệch nhẹ về phương thức trội của truy vấn. Nâng lên MoE nếu A/B cho thấy thắng rõ ràng trên miền của bạn.

### Neo sinh văn bản vào dữ liệu

LLM nên trích dẫn mục nào đã truy hồi dẫn tới mỗi khẳng định. Với đa phương thức:

- Nguồn là text: trích dẫn (citation) chuẩn `[1]`.
- Nguồn là ảnh: `[img 3]` kèm một chú thích ngắn.
- Âm thanh: `[audio 2 at 0:34]`.

Huấn luyện generator với dữ liệu có ý thức về neo: mỗi khẳng định trong đích huấn luyện được gắn nhãn chỉ số nguồn. Khi suy luận, mô hình tự nhiên phát ra trích dẫn.

### Ba khảo sát năm 2025

Abootorabi et al. (arXiv:2502.08826, "Ask in Any Modality"): cách phân loại cho RAG đa phương thức. Bao gồm truy hồi, hợp nhất, sinh văn bản. Phạm vi bao phủ rộng nhất.

Mei et al. (arXiv:2504.08748, "A Survey of Multimodal RAG"): tập trung vào benchmark bài toán con và các dạng lỗi. Hữu ích cho việc thiết kế đánh giá.

Zhao et al. (arXiv:2503.18016): khảo sát thiên về thị giác. Mạnh về nhóm công trình ColPali.

Đọc cả ba cho bạn trạng thái tiên tiến nhất tính đến mùa xuân 2025. Phần lớn bài toán con vẫn còn mở.

### MuRAG — bài báo nền tảng

MuRAG (Chen et al., 2022) là RAG đa phương thức đầu tiên. Truy hồi ảnh + text từ một KB đa phương thức, sinh ra câu trả lời. Cho thấy tính khả thi trước làn sóng VLM. Các hệ thống hiện đại (REACT, VisRAG, M3DocRAG) xây tiếp trên nó.

### Ví dụ thực tế: trợ lý lập kế hoạch du lịch

Truy vấn: "tìm cho tôi một quán brunch thuần chay yên tĩnh, nhiều ánh sáng tự nhiên."

Pipeline:

1. Phân rã truy vấn. "quiet" → từ khoá âm thanh/đánh giá; "vegan brunch" → món trong menu; "natural light" → đặc trưng ảnh.
2. Truy hồi theo từng phương thức:
   - Truy hồi text trên các đánh giá: "vegan brunch, quiet ambiance."
   - Truy hồi ảnh trên ảnh nhà hàng: "natural light, airy."
   - Truy hồi âm thanh trên các clip âm thanh môi trường: "low decibel, no music."
3. Hợp nhất điểm. Mỗi nhà hàng có một điểm tổng hợp.
4. Top-k nhà hàng → VLM generator với toàn bộ bằng chứng → câu trả lời kèm trích dẫn.

Đây vượt xa text-RAG. Mỗi phương thức thêm tín hiệu mà chỉ text thì bỏ sót.

### RAG đa phương thức dạng tác tử

Đa bước: nếu lần truy hồi đầu không trả về câu trả lời độ tin cậy cao, LLM viết lại truy vấn và truy hồi lại. Các mẫu Agentic RAG ở Giai đoạn 14 áp dụng được ở đây. Ví dụ:

- Truy hồi top-10 ban đầu → LLM hỏi "quá ồn, lọc <40 dB" → truy hồi lại.
- Truy hồi ảnh → LLM thấy một ảnh có menu → truy hồi text của menu → trả lời.

Tăng độ phức tạp nhưng xử lý được các truy vấn mà truy hồi một phát không làm được.

### Đánh giá

Đánh giá liên phương thức vẫn còn non. Các đại diện phổ biến:

- Recall@k theo từng phương thức.
- Độ chính xác top-k của chỉ mục hợp nhất (fused).
- Mức hài lòng đầu-cuối do con người chấm.
- Theo nhiệm vụ cụ thể (đơn đặt chỗ hoàn tất, giao dịch mua đã thực hiện).

Không có benchmark chuẩn nào bao phủ mọi phương thức. Hầu hết bài báo đánh giá trên các nhiệm vụ đặc thù của miền.

```figure
contrastive-matrix
```

## Áp dụng

`code/main.py`:

- Ba retriever giả lập (text, ảnh, âm thanh) hoạt động trên một kho dữ liệu nhà hàng chia sẻ.
- Hợp nhất điểm, kết hợp điểm các phương thức với trọng số cấu hình được.
- Một generator stub phát ra câu trả lời cuối kèm trích dẫn.
- Một vòng lặp tác tử đơn giản viết lại truy vấn nếu độ tin cậy thấp.

## Đóng gói

Bài này tạo ra `outputs/skill-multimodal-rag-designer.md`. Cho một spec sản phẩm với luồng truy vấn đa phương thức, nó thiết kế retriever, hợp nhất, generator và đánh giá.

## Bài tập

1. Đề xuất một RAG đa phương thức cho phân loại y tế: truy vấn = ảnh vết thương + triệu chứng dạng text. Phương thức nào truy hồi từ KB nào?

2. Hợp nhất điểm là một tổng có trọng số đơn giản. Nó có dạng lỗi nào mà hợp nhất MoE tránh được?

3. Đọc cách phân loại của Abootorabi et al. (Mục 3). Ba bài toán con chuẩn là gì và chúng ánh xạ thế nào vào sản phẩm bạn chọn?

4. Thiết kế một spec đánh giá cho RAG đa phương thức lập kế hoạch du lịch. Thang đo nào bao phủ recall ảnh, recall âm thanh và độ đúng tổng hợp?

5. RAG đa bước dạng tác tử phải trả giá độ trễ cho mỗi vòng. Ở độ khó truy vấn nào thì mức tăng độ chính xác biện minh cho độ trễ?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| Cross-modal retrieval | "Truy vấn một phương thức, truy hồi phương thức khác" | Truy vấn text truy hồi ảnh; truy vấn ảnh truy hồi text; cần một không gian chia sẻ hoặc mô-đun dịch |
| Score fusion | "Gộp điểm" | Tổng có trọng số của điểm truy hồi từng phương thức; cách hợp nhất đơn giản nhất |
| MoE fusion | "Expert định tuyến theo phương thức" | Mạng gating chọn điểm của phương thức nào đáng tin cho từng truy vấn |
| Grounded generation | "Trích dẫn nguồn" | Mỗi khẳng định trong câu trả lời được gắn chỉ số nguồn |
| MuRAG | "RAG đa phương thức đầu tiên" | Bài báo 2022 đặt ra mẫu RAG đa phương thức |
| Agentic multi-hop | "Viết lại và thử lại" | LLM truy vấn lại retriever khi độ tin cậy lần đầu thấp |

## Đọc thêm

- [Abootorabi et al. — Ask in Any Modality (arXiv:2502.08826)](https://arxiv.org/abs/2502.08826)
- [Mei et al. — A Survey of Multimodal RAG (arXiv:2504.08748)](https://arxiv.org/abs/2504.08748)
- [Zhao et al. — Vision RAG Survey (arXiv:2503.18016)](https://arxiv.org/abs/2503.18016)
- [Chen et al. — MuRAG (arXiv:2210.02928)](https://arxiv.org/abs/2210.02928)
- [Liu et al. — REACT (arXiv:2301.10382)](https://arxiv.org/abs/2301.10382)

---

[← Bài trước: ColPali và RAG tài liệu thuần thị giác](../../23-colpali-vision-native-rag/docs/vi.md) · [Bài sau: Agent đa phương thức và computer use (dùng máy tính) →](../../25-multimodal-agents-computer-use/docs/vi.md)
