# Tranh luận và cộng tác đa tác nhân

> Du et al. (ICML 2024, "Society of Minds") chạy N bản thể mô hình tự đề xuất câu trả lời độc lập, rồi lặp đi lặp lại phê bình lẫn nhau qua R vòng để hội tụ. Cải thiện tính đúng sự thật, tuân thủ quy tắc, suy luận. Topology thưa đánh bại full mesh về chi phí token.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 12 (mẫu workflow), Giai đoạn 14, Bài 05 (Self-Refine và CRITIC)
**Time:** ~60 phút

## Mục tiêu học tập

- Giải thích giao thức debate (tranh luận): N bên đề xuất, R vòng, hội tụ về một câu trả lời chung.
- Mô tả vì sao debate cải thiện tính đúng sự thật, tuân thủ quy tắc và suy luận.
- Giải thích topology thưa: không phải mọi bên tranh luận đều cần thấy mọi bên khác.
- Cài đặt một debate thuần stdlib trên LLM kịch bản với biến thể full-mesh và thưa; đo chi phí token so với độ chính xác.

## Vấn đề

Self-Refine (Bài 05) là một mô hình tự phê bình chính nó — dễ rơi vào tư duy tập thể. CRITIC (Bài 05) neo phê bình vào công cụ bên ngoài — không phải lúc nào cũng có. Debate mở ra chế độ thứ ba: nhiều bản thể, phê bình chéo, hội tụ nhờ bất đồng.

## Khái niệm

### Society of Minds (Du et al., ICML 2024)

- N bản thể mô hình độc lập đề xuất câu trả lời cho cùng một câu hỏi.
- Qua R vòng (vòng), mỗi mô hình đọc đề xuất của những mô hình khác và phê bình chúng.
- Các mô hình cập nhật câu trả lời dựa trên những phê bình đó.
- Sau R vòng, trả về câu trả lời đã hội tụ.

Thí nghiệm ban đầu dùng N=3, R=2 vì chi phí. Độ chính xác tăng khi có thêm tác nhân và thêm vòng trên các bài toán khó (MMLU, GSM8K, Chess Move Validity, sinh tiểu sử).

Kết hợp nhiều mô hình đánh bại debate một mô hình: ChatGPT + Bard cùng nhau > từng cái riêng lẻ.

### Topology thưa

"Improving Multi-Agent Debate with Sparse Communication Topology" (arXiv:2406.11776, 2024-2025) cho thấy debate full-mesh không phải lúc nào cũng tối ưu. Các topology thưa (star, ring, hub-and-spoke) có thể đạt độ chính xác tương đương với chi phí token thấp hơn. Mỗi bên tranh luận chỉ thấy một tập con các bên khác.

Hệ quả:

- Full mesh N=5, R=3 = 5 × 3 = 15 đề xuất, mỗi bên đọc 4 bên khác = 60 thao tác phê bình.
- Star N=5, R=3 (một hub + 4 spoke) = 15 đề xuất, spoke chỉ đọc hub = 12 thao tác phê bình.

### Khi nào debate có ích

- **Tính đúng sự thật.** N đề xuất độc lập, kiểm tra chéo giảm ảo giác.
- **Tuân thủ quy tắc.** Chess move validity — một mô hình bỏ sót quy tắc, các mô hình khác bắt được.
- **Suy luận mở.** Nhiều cách diễn giải khác nhau cùng thu hẹp về câu trả lời đúng.

### Khi nào debate gây hại

- **UX nhạy cảm với độ trễ.** N × R vòng tuần tự là độ trễ mà bạn có thể không có.
- **Quy mô nhạy cảm với chi phí.** N × R token cho mỗi câu hỏi.
- **Tra cứu sự thật đơn giản.** Một lần tra cứu rẻ hơn năm cuộc tranh luận.

### Hiện thực thực tế năm 2026

- **Anthropic orchestrator-workers** (Bài 12) — một biến thể của debate có thêm bước tổng hợp.
- **LangGraph supervisor** (Bài 13) — router trung tâm + các tác nhân chuyên môn có thể cài debate như một node.
- **OpenAI Agents SDK** (Bài 16) — các tác nhân chuyển giao qua lại để phê bình lặp.
- **Multi-agent evals** — ghép debate + evaluator-optimizer để lấy tín hiệu eval.

### Mẫu này hỏng ở đâu

- **Sụp hội tụ.** Mọi tác nhân hội tụ về câu trả lời sai đầu tiên. Giảm thiểu bằng cách bắt buộc có các vòng bất đồng.
- **Hub hỏng.** Trong topology star, một hub tồi làm hỏng tất cả. Luân phiên hoặc dùng nhiều hub.
- **Đồng nhất prompt.** Mọi tác nhân dùng cùng prompt; chúng tạo ra cùng câu trả lời. Dùng prompt và/hoặc mô hình đa dạng.

```figure
debate-converge
```

## Tự tay làm

`code/main.py` cài đặt debate thuần stdlib:

- Class `Debater` (LLM kịch bản với độ trôi quan điểm riêng cho từng bên tranh luận).
- Runner `FullMeshDebate` và `SparseDebate`.
- Ba câu hỏi: một về sự thật, một dựa trên quy tắc, một về suy luận.
- Số liệu: câu trả lời hội tụ, số vòng tới hội tụ, tổng thao tác phê bình.

Chạy:

```
python3 code/main.py
```

Kết quả: độ chính xác và chi phí theo từng giao thức; bản thưa khớp full mesh ở 2/3 câu hỏi với chi phí thấp hơn.

## Áp dụng

- **Anthropic orchestrator-workers** cho debate 2-3 worker đơn giản.
- **LangGraph** cho debate nhiều vòng có trạng thái với checkpointing.
- **Tự viết** cho nghiên cứu hoặc các đảm bảo đúng đắn chuyên biệt.

## Đóng gói

`outputs/skill-debate.md` dựng khung một debate đa tác nhân với topology, N, R và quy tắc hội tụ cấu hình được.

## Bài tập

1. Cài đặt quy tắc "bất đồng bắt buộc": ở vòng 1, mỗi bên tranh luận phải đưa ra một đề xuất khác biệt. Đo ảnh hưởng lên tốc độ hội tụ.
2. Thêm tổng hợp có trọng số độ tin cậy: các bên trả về (answer, confidence); bộ tổng hợp nhân trọng số theo độ tin cậy. Có ích không?
3. Đổi một "tác nhân" sang một LLM kịch bản khác có quan điểm khác. Tính không đồng nhất có cải thiện độ chính xác không?
4. Đo chi phí token cho full mesh so với bản thưa trên 3 câu hỏi của bạn. Vẽ đồ thị chi phí so với độ chính xác.
5. Đọc bài báo Society of Minds. Chuyển bản đồ chơi của bạn sang N=5, R=3. Cái gì vỡ? Cái gì tốt hơn?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Debate | "Phê bình đa tác nhân" | N bên đề xuất, R vòng phê bình chéo, hội tụ |
| Full mesh | "Ai cũng đọc mọi người" | Mỗi bên tranh luận đọc mọi bên khác trong mỗi vòng |
| Sparse topology | "Góc nhìn hạn chế về bên khác" | Các bên tranh luận chỉ đọc một tập con các bên khác |
| Hub-and-spoke | "Topology hình sao" | Một bên tranh luận trung tâm, N-1 spoke chỉ đọc hub |
| Convergence | "Đồng thuận" | Các bên tranh luận hội tụ về một câu trả lời chung |
| Society of Minds | "Bài báo debate của Du et al." | Phương pháp tranh luận đa tác nhân của ICML 2024 |

## Đọc thêm

- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — bài gốc về tranh luận đa tác nhân
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) — kết quả của topology thưa
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — orchestrator-workers như một biến thể của debate
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — bản đối chiếu tự phê bình một mô hình

---

[← Bài trước: Khả năng quan sát agent: Langfuse, Phoenix, Opik](../../24-agent-observability-platforms/docs/vi.md) · [Bài sau: Kiểu hỏng: Vì sao agent vỡ →](../../26-failure-modes-agentic/docs/vi.md)
