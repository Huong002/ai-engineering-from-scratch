# Bộ nhớ lai: Vector + Graph + KV

> Bộ nhớ lai (hybrid memory) chạy ba kho song song — vector cho độ tương đồng ngữ nghĩa, KV cho tra cứu fact nhanh, graph cho suy luận quan hệ giữa các thực thể (entity) — cùng một lớp chấm điểm hợp nhất chúng khi truy hồi. Đây là một mẫu production được dùng rộng rãi cho bộ nhớ ngoài; Mem0 (Chhikara và cộng sự, 2025) là một bản tham chiếu cụ thể.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 07 (MemGPT), Giai đoạn 14, Bài 08 (khối Letta)
**Time:** ~75 phút

## Mục tiêu học tập

- Giải thích vì sao một kho đơn lẻ (chỉ vector, chỉ graph, chỉ KV) là không đủ cho bộ nhớ của agent.
- Kể tên ba kho song song của Mem0 và mỗi kho tối ưu cho việc gì.
- Mô tả cách chấm điểm hợp nhất của Mem0 — relevance, importance, recency — và vì sao đó là tổng có trọng số chứ không phải phân cấp.
- Tự viết một bộ nhớ ba kho tối giản bằng stdlib, với `add()` ghi vào cả ba kho và `search()` hợp nhất kết quả.

## Vấn đề

Một kho là sai cho một trong ba loại truy vấn:

- **Tương đồng ngữ nghĩa** — "tuần trước ta đã nói gì về agent drift?" Vector thắng; KV và graph trượt.
- **Tra cứu fact** — "số điện thoại của người dùng là gì?" KV thắng; vector thì lãng phí, graph thì quá tay.
- **Suy luận quan hệ** — "những khách hàng nào dùng chung một thực thể thanh toán?" Graph thắng; vector và KV không trả lời được.

Agent production phát ra cả ba loại truy vấn trong một phiên. Bộ nhớ một kho luôn sai với hai trong ba. Đóng góp của Mem0 là nối cả ba kho phía sau một bề mặt `add`/`search` duy nhất với một hàm chấm điểm hợp nhất chúng.

## Khái niệm

### Ba kho song song

Mem0 (arXiv:2504.19413, tháng 4 năm 2025) với `add(text, user_id, metadata)`:

1. Trích xuất (extraction) các fact ứng viên từ văn bản (một bước do LLM đảm nhiệm).
2. Ghi mỗi fact vào vector store (kho vector) để tìm kiếm ngữ nghĩa bằng embedding.
3. Ghi mỗi fact vào KV store với khoá `(user_id, fact_type, entity)` để tra cứu O(1).
4. Ghi mỗi fact vào graph store (bộ nhớ đồ thị, Mem0g) dưới dạng các cạnh có kiểu cho truy vấn quan hệ.

Khi `search(query, user_id)`:

1. Vector store trả về top-k theo cosine của embedding.
2. KV store trả về các kết quả khớp trực tiếp theo (user_id, type, entity) suy ra từ truy vấn.
3. Graph store trả về đồ thị con có thể tới được từ các thực thể trong truy vấn.
4. Một lớp chấm điểm hợp nhất cả ba.

### Chấm điểm hợp nhất

```
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Relevance** — cosine của vector, khớp chính xác của KV, trọng số đường đi trên graph.
- **Importance** — được gắn thẻ lúc ghi hoặc học được (một số fact quan trọng hơn: tên, ID, chính sách).
- **Recency** — phân rã theo hàm mũ tính từ lần ghi hoặc đọc cuối.

Trọng số được tinh chỉnh theo từng sản phẩm. `w_recency` cao hơn cho agent trò chuyện; `w_importance` cao hơn cho agent tuân thủ; `w_relevance` cao hơn cho agent truy hồi.

### Mem0g và suy luận theo thời gian

Mem0g thêm một bộ phát hiện xung đột. Khi một fact mới mâu thuẫn với một cạnh đã có, cạnh cũ bị đánh dấu không hợp lệ nhưng không bị xoá. Các truy vấn theo thời gian ("tháng Ba người dùng ở thành phố nào?") sẽ duyệt đồ thị con hợp lệ tại thời điểm đó.

Đây là hành vi chuẩn mức tuân thủ mà mẫu invalidation của Letta khái quát hoá.

### Con số benchmark

Bài báo Mem0 báo cáo (2025):

- **LoCoMo** (bộ nhớ hội thoại dài): 91.6
- **LongMemEval** (bộ nhớ tình huống tầm xa): 93.4
- **BEAM 1M** (benchmark bộ nhớ 1M token): 64.1

Các baseline so sánh (LLM full-context 128k, vector store phẳng, KV phẳng) đều thua 10+ điểm. Benchmark một mình không đủ để biện minh cho lựa chọn — hình dạng vận hành mới là lý do — nhưng những con số cho thấy thiết kế hợp nhất không phải là sai số làm tròn.

### Phân loại theo phạm vi

Mem0 chia bộ nhớ theo phạm vi:

- **Bộ nhớ người dùng** — tồn tại qua các phiên, khoá theo `user_id`.
- **Bộ nhớ phiên** — tồn tại trong một luồng hội thoại.
- **Bộ nhớ agent** — trạng thái của từng phiên bản agent.

Mỗi lần ghi chọn một phạm vi. Truy hồi có thể hỏi xuyên phạm vi với trọng số riêng cho từng phạm vi. Trộn phạm vi mà không suy nghĩ là cách tạo ra sự cố "trợ lý kể cho Alice về dự án của Bob".

### Mẫu này hỏng ở đâu

- **Trôi embedding.** Kết quả vector trông đúng ở trăm truy vấn đầu rồi tệ dần khi corpus lớn lên. Hãy định kỳ embedding lại các bản ghi được dùng nhiều nhất (top-N).
- **Phình schema KV.** `(user_id, type, entity)` trông đơn giản cho tới khi mỗi team tự thêm `type` của mình. Soát lại tập type mỗi quý.
- **Bùng nổ graph.** Một bộ trích xuất nhiễu có thể thêm 50 cạnh mỗi tin nhắn. Đặt trần số lần ghi graph cho mỗi lần gọi `add`; bỏ những cạnh có độ tin cậy thấp.

```figure
ae-memory-fusion
```

## Tự tay làm

`code/main.py` hiện thực mẫu ba kho bằng stdlib:

- `VectorStore` — độ tương đồng trùng token đơn giản thay cho embedding.
- `KVStore` — dict khoá theo `(user_id, fact_type, entity)`.
- `GraphStore` — các cạnh có kiểu (subject, relation, object, valid).
- `Mem0` — facade cấp cao với `add()`, `search()`, chấm điểm hợp nhất và truy hồi theo phạm vi.
- Một vết chạy mẫu trên cuộc hội thoại nhiều người dùng, nhiều phiên.

Chạy nó:

```
python3 code/main.py
```

Output cho thấy ba đường truy hồi riêng biệt cộng với top-k hợp nhất. Đổi trọng số chấm điểm ở đầu `main()` và xem thứ hạng thay đổi.

## Áp dụng

- **Mem0 (Apache 2.0)** — sẵn sàng cho production. Tự host với Postgres + Qdrant + Neo4j, hoặc dùng bản cloud được quản lý.
- **Letta** — lõi ba tầng core/recall/archival; tự mang backend vector và graph của bạn.
- **Zep** — lựa chọn thương mại với KG theo thời gian và trích xuất fact.
- **Tự xây** — khi bạn cần kiểm soát chính xác bộ trích xuất (tuân thủ) hoặc trọng số hợp nhất (agent giọng nói, nơi recency chiếm ưu thế).

## Đóng gói

`outputs/skill-hybrid-memory.md` sinh ra một bộ khung bộ nhớ ba kho với bộ chấm điểm hợp nhất, phân loại theo phạm vi và invalidation theo thời gian được nối sẵn.

## Bài tập

1. Thay độ tương đồng vector tối giản bằng một embedding model thật (sentence-transformers, Ollama, OpenAI embeddings). Đo recall@10 trên một hội thoại dài tổng hợp. Thứ hạng có trôi qua 1000 lần ghi không?
2. Thêm truy vấn theo thời gian: `search(query, as_of=timestamp)`. Chỉ trả về các bản ghi hợp lệ tại hoặc trước thời điểm đó. Kho nào cần sửa nhiều nhất?
3. Viết bộ phát hiện xung đột: nếu một fact đến mâu thuẫn với một cạnh trong graph, đánh dấu cạnh cũ không hợp lệ và ghi log cả hai. Test trên "user lives in Berlin" -> "user lives in Lisbon."
4. Mở rộng bộ chấm điểm hợp nhất để thêm chiều `user_feedback` (thumbs-up trên các bản ghi được truy hồi). Làm sao tránh bị game (agent chỉ trả về những bản ghi nó đã thích)?
5. Đọc tài liệu Mem0 (`docs.mem0.ai`). Chuyển bản tối giản sang các lệnh gọi client `mem0`. So sánh chất lượng truy hồi trên cùng 20 truy vấn test.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Hybrid memory | "Vector cộng graph cộng KV" | Ba kho được ghi song song, hợp nhất khi truy hồi |
| Fact extraction | "Nạp bộ nhớ" | Bước LLM chia văn bản thành các bộ (entity, relation, fact) |
| Fusion scoring | "Xếp hạng theo độ liên quan" | Tổng có trọng số của relevance, importance, recency |
| Scope | "Namespace bộ nhớ" | user / session / agent — quyết định ai thấy gì |
| Mem0g | "Đồ thị bộ nhớ" | Các cạnh có kiểu kèm hiệu lực theo thời gian cho truy vấn quan hệ |
| Temporal invalidation | "Xoá mềm" | Đánh dấu cạnh bị mâu thuẫn là không hợp lệ; không bao giờ xoá |
| Embedding drift | "Truy hồi mục ruỗng" | Chất lượng vector giảm khi corpus lớn lên; định kỳ embedding lại |

## Đọc thêm

- [Chhikara và cộng sự, Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — bài báo gốc
- [Tài liệu Mem0](https://docs.mem0.ai/platform/overview) — API production, SDK, cloud được quản lý
- [Packer và cộng sự, MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — tiền thân của ngữ cảnh ảo
- [Letta, blog Memory Blocks](https://www.letta.com/blog/memory-blocks) — thiết kế anh em ba tầng

---

[← Bài trước: Memory Blocks và Sleep-Time Compute](../../08-memory-blocks-sleep-time-compute/docs/vi.md) · [Bài sau: Thư viện kỹ năng và học tập suốt đời (Voyager) →](../../10-skill-libraries-voyager/docs/vi.md)
