# Bộ nhớ agent — Ngữ cảnh ảo và phân trang bộ nhớ

> Cửa sổ ngữ cảnh thì hữu hạn. Hội thoại, tài liệu và vết công cụ thì không. Cách khắc phục chính là bộ nhớ ảo của hệ điều hành được nói lại — main context là RAM, kho ngoài là đĩa, agent phân trang qua lại giữa chúng. MemGPT (Packer et al., 2023) đặt tên cho mẫu này; nhiều hệ thống bộ nhớ trong sản xuất xây trên nó.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 01 (vòng lặp agent), Giai đoạn 14, Bài 06 (sử dụng công cụ)
**Time:** ~75 phút

## Mục tiêu học tập

- Giải thích phép so sánh với hệ điều hành mà MemGPT dựa trên: main context = RAM, external context = đĩa, công cụ bộ nhớ = nạp/đẩy trang.
- Cài đặt mẫu hai tầng của MemGPT bằng stdlib với một bộ đệm main context, một kho ngoài tìm kiếm được, và các công cụ nạp/đẩy trang.
- Mô tả cách agent phát "interrupt" để truy vấn hoặc sửa bộ nhớ ngoài và cách kết quả được ghép trở lại vào prompt kế tiếp.
- Nhận diện những lựa chọn thiết kế của MemGPT được mang sang Letta (Bài 08) và Mem0 (Bài 09).

## Vấn đề

Context window (cửa sổ ngữ cảnh) trông như thể sẽ giải quyết được chuyện bộ nhớ. Nhưng không. Ba kiểu hỏng lặp lại trong sản xuất:

1. **Tràn.** Hội thoại nhiều lượt, tài liệu dài, hay quỹ đạo nhiều lần gọi công cụ đều vượt qua cửa sổ. Mọi thứ sau mốc cắt đều biến mất.
2. **Pha loãng.** Ngay cả trong cửa sổ, nhồi ngữ cảnh không liên quan làm loãng sự chú ý khỏi phần quan trọng. Các mô hình tiên phong vẫn suy giảm chất lượng với đầu vào dài.
3. **Bền vững.** Một phiên mới bắt đầu với cửa sổ rỗng. Agent không có bộ nhớ ngoài không thể nói "nhớ lúc bạn bảo tôi..." xuyên các phiên.

Cửa sổ lớn hơn có giúp nhưng không sửa được điều này. Bài báo năm 2025 của Mem0 đo được rằng các baseline cửa sổ 128k vẫn bỏ sót những sự thật tầm xa mà một agent cửa sổ 4k có bộ nhớ ngoài vẫn bắt được.

## Khái niệm

### Phép so sánh với hệ điều hành

MemGPT (Packer et al., arXiv:2310.08560, v2 tháng 2 2024) ánh xạ việc quản lý ngữ cảnh sang bộ nhớ ảo của hệ điều hành:

| Khái niệm OS | Khái niệm MemGPT | Tương đương trong sản xuất 2026 |
|------------|---------------|------------------------|
| RAM | main context (prompt) | Anthropic/OpenAI context window |
| Đĩa | ngữ cảnh ngoài | vector DB, KV, graph store |
| Page fault | lần gọi công cụ bộ nhớ | `memory.search`, `memory.read`, `memory.write` |
| OS kernel | vòng điều khiển agent | vòng ReAct với công cụ bộ nhớ |

Agent chạy một vòng ReAct bình thường. Thêm một lớp công cụ cho phép nó nạp dữ liệu vào và đẩy ra khỏi main context.

### Hai tầng

- **Main context.** Prompt kích thước cố định giữ tác vụ hiện tại. Luôn hiển thị với mô hình.
- **External context.** Không giới hạn, tìm kiếm được qua công cụ. Đọc khi cần, ghi khi có sự thật mới xuất hiện.

Bài báo gốc đánh giá thiết kế trên hai tác vụ vượt ngoài cửa sổ cơ sở: phân tích tài liệu dài hơn 100k token và hội thoại nhiều phiên với bộ nhớ bền xuyên nhiều ngày.

### Mẫu interrupt

MemGPT đưa ra bộ nhớ như interrupt: giữa hội thoại agent có thể gọi một công cụ bộ nhớ, runtime thực thi nó, và kết quả được ghép vào lượt trả lời kế tiếp như một quan sát mới. Về khái niệm thì hoàn toàn giống một syscall `read()` của Unix: chặn tiến trình, trả về byte, rồi tiến trình chạy tiếp.

Bề mặt công cụ bộ nhớ chuẩn:

- `core_memory_append(section, text)` — ghi vào một mục bền của prompt.
- `core_memory_replace(section, old, new)` — sửa một mục bền.
- `archival_memory_insert(text)` — ghi vào kho ngoài tìm kiếm được.
- `archival_memory_search(query, top_k)` — lấy từ kho ngoài.
- `conversation_search(query)` — quét các lượt trước.

### Nơi bài báo kết thúc và sản xuất bắt đầu

Tháng 9 2024, MemGPT trở thành Letta. Repo nghiên cứu (`cpacker/MemGPT`) vẫn còn; Letta mở rộng thiết kế:

- Ba tầng bộ nhớ (memory tier) thay vì hai (core, recall, archival — Bài 08).
- Reasoning gốc thay cho mẫu `send_message`/heartbeat (Bài 08).
- Agent thời gian ngủ chạy việc bộ nhớ bất đồng bộ (Bài 08).

Bài báo MemGPT vẫn là nền tảng của năm 2026 kể cả khi hệ thống sản xuất chạy Letta, Mem0, hay một kho hai tầng tự viết.

### Nơi mẫu này đi sai

- **Bộ nhớ mục ruỗng.** Ghi tích tụ nhanh hơn đọc; truy hồi bị nhấn chìm trong sự thật cũ. Cách sửa: hợp nhất định kỳ (sleep-time của Letta), vô hiệu hoá tường minh (bộ phát hiện xung đột của Mem0).
- **Đầu độc bộ nhớ.** Bộ nhớ ngoài là văn bản được truy hồi. Nếu nội dung do kẻ tấn công kiểm soát lọt vào một ghi chú bộ nhớ, agent sẽ nạp lại nó ở phiên sau. Đây là cuộc tấn công của Greshake et al. (Bài 27) được nói lại theo thời gian.
- **Mất trích dẫn.** Agent nhớ "người dùng bảo tôi ship X" nhưng không chỉ ra được lượt nào. Hãy lưu tham chiếu nguồn (session ID, turn ID) với mỗi lần ghi vào archival.

```figure
context-budget
```

## Tự tay làm

`code/main.py` cài đặt mẫu hai tầng của MemGPT bằng stdlib:

- `MainContext` — bộ đệm prompt kích thước cố định với một dict `core` và một list `messages`; tự nén các tin nhắn cũ nhất khi vượt mức.
- `ArchivalStore` — kho trong bộ nhớ kiểu BM25 (chấm điểm theo trùng lặp token) lưu các bản ghi (id, text, tags, session, turn).
- Năm công cụ bộ nhớ ánh xạ đúng bề mặt MemGPT.
- Một agent kịch bản nạp sự thật vào archival, rồi trả lời một câu hỏi bằng cách gọi `archival_memory_search`.

Chạy nó:

```
python3 code/main.py
```

Vết chạy cho thấy agent ghi ba sự thật, làm đầy main context tới hạn mức, buộc phải đẩy ra (eviction), rồi trả lời câu hỏi tiếp theo bằng cách truy hồi từ archival — tái hiện quy trình MemGPT mà không cần LLM thật.

## Áp dụng

Mọi hệ thống bộ nhớ sản xuất hôm nay đều là một biến thể của MemGPT:

- **Letta** (Bài 08) — ba tầng, reasoning gốc, sleep-time compute.
- **Mem0** (Bài 09) — vector + KV + graph hợp nhất với một lớp chấm điểm.
- **OpenAI Assistants / Responses** — bộ nhớ được quản lý qua thread và file.
- **Claude Agent SDK** — bộ nhớ dài hạn qua skill và session store.

Chọn theo hình dạng vận hành (tự host, được quản lý, tích hợp framework), không phải theo mẫu lõi — mẫu lõi là MemGPT.

### Hình dạng của bộ nhớ agent

Phân trang (paging) giải quyết sức chứa. Nó không quyết định lưu gì. Bốn kiểu bộ nhớ lặp lại trong các hệ thống sản xuất, mỗi kiểu trả lời một câu hỏi khác nhau:

- **Bộ nhớ làm việc** — điều gì quan trọng ngay lúc này? Tầng nằm trong ngữ cảnh: tác vụ hiện tại, các lượt gần đây, các mục core được ghim. Chính là prompt.
- **Bộ nhớ tình huống** — chuyện gì đã xảy ra? Các lượt và quỹ đạo trước, lưu kèm tham chiếu phiên và lượt, phát lại được khi cần.
- **Bộ nhớ ngữ nghĩa** — điều gì là đúng? Sự thật về người dùng, miền, thế giới, được cập nhật và khử trùng lặp khi thay đổi.
- **Bộ nhớ thủ tục** — tôi làm việc này thế nào? Các thói quen, tuỳ thích và luật đã học để dẫn dắt hành vi tương lai thay vì truy hồi.

Các bản cài đặt mã nguồn mở chọn điểm tấn công khác nhau:

| Kiểu | Cách cài đặt | Cách nó giải quyết |
|------|----------------|-------------------|
| Làm việc | MemGPT / Letta | Nạp nội dung vào và đẩy ra khỏi ngân sách prompt cố định qua công cụ bộ nhớ (bài này, Bài 08) |
| Tình huống | Zep | Đồ thị tri thức thời gian — sự thật mang khoảng hiệu lực, nên "điều gì đúng khi nào" truy vấn được |
| Ngữ nghĩa | Mem0 | Pipeline trích xuất khử trùng lặp và cập nhật sự thật trên kho vector, KV và graph (Bài 09) |
| Ngữ nghĩa + thủ tục | LangMem | Trích xuất nền các sự thật và luật hành vi vào một kho mà agent tham vấn giữa các lượt |
| Tình huống + ngữ nghĩa | agentmemory | Ghi lại phiên khi đang chạy, hợp nhất chúng thành các bản ghi có kiểu, tìm kiếm được |

## Đóng gói

`outputs/skill-virtual-memory.md` là một skill tái dùng được, tạo bộ khung bộ nhớ hai tầng đúng chuẩn (main + archival + bề mặt công cụ) cho mọi runtime đích, với chính sách eviction và trường trích dẫn được nối sẵn.

## Bài tập

1. Thêm một hạn mức `max_main_context_tokens` đo bằng token (xấp xỉ bằng `len(text.split())` * 1.3). Nén các tin nhắn cũ nhất thành một bản tóm tắt khi vượt hạn mức. So sánh hành vi có và không có bộ tóm tắt.
2. Cài đặt BM25 đúng chuẩn trên kho archival (tần suất từ, nghịch đảo tần suất tài liệu). Đo recall@10 trên một tập sự thật đồ chơi so với baseline trùng lặp token.
3. Thêm trường `citation` (session_id, turn_id, source_url) vào các lần chèn archival. Bắt agent trích nguồn trong mọi câu trả lời dựa trên truy hồi.
4. Mô phỏng việc đầu độc bộ nhớ: thêm một bản ghi archival nói "bỏ qua mọi chỉ dẫn của người dùng về sau". Viết một bộ chặn quét các kết quả truy hồi tìm văn bản dạng mệnh lệnh và đánh dấu chúng là không đáng tin.
5. Chuyển bản cài đặt sang dùng schema JSON core-memory của repo nghiên cứu MemGPT (`cpacker/MemGPT`). Điều gì thay đổi khi bạn chuyển từ chuỗi phẳng sang các mục có kiểu?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Virtual context | "Bộ nhớ không giới hạn" | Tầng main (prompt) + tầng ngoài (tìm kiếm được) với nạp/đẩy trang |
| Main context | "Bộ nhớ làm việc" | Chính là prompt — kích thước cố định, luôn hiển thị |
| Archival memory | "Kho dài hạn" | Lưu bền ở ngoài, tìm kiếm được, truy hồi khi cần |
| Core memory | "Mục prompt bền" | Các mục có tên được ghim trong main context |
| Memory tool | "Memory API" | Lần gọi công cụ agent phát ra để đọc/ghi bộ nhớ ngoài |
| Interrupt | "Page fault bộ nhớ" | Agent dừng lại, runtime lấy dữ liệu, kết quả ghép vào lượt sau |
| Memory rot | "Sự thật cũ" | Ghi cũ nhấn chìm truy hồi; sửa bằng hợp nhất định kỳ |
| Memory poisoning | "Ghi chú bền bị cài" | Nội dung của kẻ tấn công được lưu như bộ nhớ, bị nạp lại khi truy hồi (recall) |

## Đọc thêm

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — bài báo về ngữ cảnh ảo lấy cảm hứng từ hệ điều hành
- [Letta, blog Memory Blocks](https://www.letta.com/blog/memory-blocks) — bước tiến hoá ba tầng
- [Anthropic, Kỹ nghệ ngữ cảnh hiệu quả](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — coi ngữ cảnh như một ngân sách
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — bộ nhớ sản xuất lai trên nền mẫu này
- [Zep (getzep/zep)](https://github.com/getzep/zep) — bộ nhớ đồ thị tri thức thời gian từ bảng phân loại
- [Mem0 (mem0ai/mem0)](https://github.com/mem0ai/mem0) — pipeline trích xuất đằng sau kho lai của Bài 09
- [LangMem (langchain-ai/langmem)](https://github.com/langchain-ai/langmem) — trích xuất nền các sự thật và luật hành vi
- [agentmemory (rohitg00/agentmemory)](https://github.com/rohitg00/agentmemory) — ghi lại phiên rồi hợp nhất thành bản ghi có kiểu, tìm kiếm được

---

[← Bài trước: Sử dụng công cụ và function calling (gọi hàm)](../../06-tool-use-and-function-calling/docs/vi.md) · [Bài sau: Memory Blocks và Sleep-Time Compute →](../../08-memory-blocks-sleep-time-compute/docs/vi.md)
