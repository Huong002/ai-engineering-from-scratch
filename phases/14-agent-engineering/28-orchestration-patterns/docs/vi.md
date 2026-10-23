# Các mẫu điều phối (orchestration): Supervisor, Swarm, Hierarchical

> Bốn mẫu điều phối lặp đi lặp lại trong các framework năm 2026: supervisor (người giám sát) - worker, swarm / peer-to-peer, hierarchical, debate. Hướng dẫn của Anthropic: "Điều quan trọng là xây đúng hệ thống cho nhu cầu của bạn." Hãy bắt đầu đơn giản; chỉ thêm topology khi một agent duy nhất cộng năm mẫu workflow vẫn không đủ.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 12 (mẫu workflow), Giai đoạn 14, Bài 25 (tranh luận đa agent)
**Time:** ~60 phút

## Mục tiêu học tập

- Kể tên bốn mẫu điều phối lặp lại và trường hợp phù hợp của từng mẫu.
- Mô tả khuyến nghị của LangChain năm 2026: giám sát dựa trên tool call so với các thư viện supervisor.
- Giải thích quy tắc "xây đúng hệ thống" của Anthropic và cách nó quyết định việc chọn topology.
- Hiện thực cả bốn mẫu bằng stdlib trên một LLM được kịch bản hoá dùng chung.

## Vấn đề

Các đội thường vươn tới "multi-agent" trước khi thực sự cần. Bốn mẫu lặp lại trong các framework; một khi gọi được tên chúng, bạn có thể chọn đúng mẫu — hoặc bỏ qua topology hoàn toàn.

## Khái niệm

### Supervisor-worker

- Một LLM định tuyến trung tâm phân việc cho các agent chuyên trách.
- Quyết định: quay lại chính nó, handoff (bàn giao) cho specialist, hay kết thúc.
- Các specialist không nói chuyện với nhau; mọi việc định tuyến đều đi qua supervisor.

Frameworks: LangGraph `create_supervisor`, Anthropic orchestrator-workers, CrewAI Hierarchical Process.

**Khuyến nghị của LangChain năm 2026:** thực hiện việc giám sát thông qua các tool call trực tiếp thay vì `create_supervisor`. Cách này cho quyền kiểm soát context engineering chi tiết hơn — bạn quyết định chính xác mỗi specialist nhìn thấy gì.

### Swarm / peer-to-peer

- Các agent bàn giao trực tiếp qua một bề mặt tool dùng chung.
- Không có router trung tâm.
- Độ trễ thấp hơn supervisor (ít bước nhảy hơn).
- Khó suy luận hơn (không có một điểm kiểm soát duy nhất).

Frameworks: topology swarm của LangGraph, handoff của OpenAI Agents SDK (khi mọi agent đều có thể bàn giao cho mọi agent khác).

### Hierarchical

- Các supervisor quản lý sub-supervisor, sub-supervisor quản lý worker.
- Được hiện thực dưới dạng subgraph lồng nhau trong LangGraph; crew lồng nhau trong CrewAI.
- Mở rộng tới quần thể agent lớn với cái giá là độ phức tạp vận hành.

Khi nào bạn cần: khi ngân sách context của một supervisor không thể chứa mô tả của tất cả specialist.

### Debate

- Các proposer chạy song song + phản biện chéo lặp lại (Bài 25).
- Thực ra không hẳn là điều phối — thiên về kiểm chứng — nhưng vẫn xuất hiện như một lựa chọn topology trong các framework.

### Crew tự trị so với luồng tất định

CrewAI chính thức hoá hai chế độ triển khai:

- **Flow** cho tự động hoá hướng sự kiện một cách tất định (điểm khởi đầu được khuyến nghị cho production).
- **Crew** cho cộng tác tự trị dựa trên vai trò.

Điều này trực giao với bốn mẫu ở trên nhưng ánh xạ được sang topology: Flow thường là supervisor hoặc hierarchical; Crew thường là supervisor có thêm router LLM.

### Hướng dẫn của Anthropic

"Thành công trong không gian LLM không nằm ở việc xây hệ thống tinh vi nhất. Mà ở việc xây hệ thống phù hợp với nhu cầu của bạn."

Thứ tự quyết định:

1. Một agent duy nhất + các mẫu workflow (Bài 12) — bắt đầu từ đây.
2. Supervisor-worker — khi bạn có 2-4 specialist.
3. Swarm — khi độ trễ quan trọng hơn sự rõ ràng trong suy luận.
4. Hierarchical — chỉ khi ngân sách context của supervisor không đủ.
5. Debate — khi độ chính xác quan trọng hơn chi phí.

### Mẫu này đi sai ở đâu

- **Tư duy topology trước.** "Chúng ta cần multi-agent" trước khi xác định multi-agent giải quyết bài toán gì.
- **Handoff nảy qua nảy lại trong swarm.** A -> B -> A -> B. Hãy dùng bộ đếm bước nhảy.
- **Hệ phân cấp giả tạo.** Ba tầng vì "enterprise"; thực tế chỉ có hai đội. Hãy thu gọn.

```figure
orchestration-pattern
```

## Tự tay làm

`code/main.py` hiện thực cả bốn mẫu bằng stdlib trên một LLM được kịch bản hoá:

- `Supervisor` — router trung tâm.
- `Swarm` — peer-to-peer với handoff trực tiếp.
- `Hierarchical` — supervisor của các supervisor.
- `Debate` — các proposer song song + phản biện.

Mỗi mẫu xử lý cùng một tác vụ ba ý định (refund / bug / sales). Hình dạng trace khác nhau.

Chạy nó:

```
python3 code/main.py
```

Output: trace theo từng mẫu + số op. Supervisor gọn nhất; swarm ngắn nhất; hierarchical sâu nhất; debate tốn kém nhất.

## Áp dụng

- **LangGraph** cho supervisor và hierarchical (subgraph lồng nhau).
- **OpenAI Agents SDK** cho handoff dưới dạng tool (hình dạng supervisor).
- **CrewAI Flow** cho production tất định.
- **Tự viết** cho debate hoặc khi bạn muốn kiểm soát chính xác.

## Đóng gói

`outputs/skill-orchestration-picker.md` chọn một topology và hiện thực nó.

## Bài tập

1. Chuyển một supervisor-worker thành swarm bằng cách bỏ router. Cái gì hỏng? Cái gì tốt lên?
2. Thêm bộ đếm bước nhảy vào swarm: từ chối sau 3 lần handoff. Nó có bắt được việc nảy A->B->A không?
3. Xây một hệ phân cấp hai tầng cho miền có 12 specialist. Ngân sách context hỏng ở đâu nếu không lồng nhau?
4. Đo hiệu năng bốn mẫu trên một khối lượng công việc giống production. Mẫu nào thắng ở chỉ số nào (độ trễ, chi phí, độ chính xác, khả năng debug)?
5. Đọc bài "Building Effective Agents" của Anthropic. Ánh xạ từng luồng production của bạn vào một trong bốn mẫu. Có luồng nào không ánh xạ gọn được không?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Supervisor-worker | "Router + các specialist" | LLM trung tâm phân việc cho specialist; chúng không nói chuyện với nhau |
| Swarm | "Peer-to-peer" | Handoff trực tiếp qua tool dùng chung; không có router trung tâm |
| Hierarchical | "Supervisor của các supervisor" | Subgraph lồng nhau cho quần thể lớn |
| Debate | "Proposer + phản biện" | Các proposer song song, phản biện chéo (Bài 25) |
| Tool-call-based supervision | "Supervisor không cần thư viện" | Hiện thực supervisor bằng các tool call trực tiếp để kiểm soát context |
| Crew | "Đội tự trị" | Chế độ cộng tác dựa trên vai trò của CrewAI |
| Flow | "Luồng công việc tất định" | Chế độ production hướng sự kiện của CrewAI |

## Đọc thêm

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — năm mẫu + agent so với workflow
- [Tổng quan LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor, swarm, hierarchical
- [Tài liệu CrewAI](https://docs.crewai.com/en/introduction) — Crew so với Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — mẫu debate

---

[← Bài trước: Prompt injection (tiêm lệnh vào prompt) và phòng thủ PVE](../../27-prompt-injection-defense/docs/vi.md) · [Bài sau: Runtime production: Hàng đợi, Sự kiện, Cron →](../../29-production-runtimes/docs/vi.md)
