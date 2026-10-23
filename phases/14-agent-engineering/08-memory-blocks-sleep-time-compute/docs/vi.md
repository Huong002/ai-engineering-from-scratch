# Memory Blocks và Sleep-Time Compute

> Những memory block (khối bộ nhớ) chức năng rời rạc mà model sửa được trực tiếp, cùng một agent lúc nghỉ làm nhiệm vụ consolidation (cố kết) bộ nhớ bất đồng bộ trong lúc agent chính không bận. Hai ý tưởng này chính là cách bạn mở rộng bộ nhớ vượt khỏi một cuộc hội thoại.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 07 (MemGPT)
**Time:** ~75 phút

## Mục tiêu học tập

- Kể tên ba tầng bộ nhớ Letta dùng — core, recall và archival (lưu trữ) — cùng vai trò của từng tầng.
- Giải thích mẫu memory block: block Human, block Persona và các block do người dùng tự định nghĩa như những đối tượng có kiểu hạng nhất.
- Mô tả sleep-time compute (tính toán lúc nghỉ) là gì, vì sao nó nằm ngoài critical path, và vì sao nó có thể chạy một model mạnh hơn agent chính.
- Triển khai một vòng lặp hai agent theo kịch bản, trong đó agent chính phục vụ phản hồi và agent lúc nghỉ cố kết các block giữa các lượt.

## Vấn đề

MemGPT (Bài 07) đã giải quyết phần luồng điều khiển của bộ nhớ ảo. Ba vấn đề nảy sinh trong môi trường production:

1. **Độ trễ.** Mọi thao tác bộ nhớ đều nằm trên critical path. Nếu agent phải cắt tỉa, tóm tắt hay hoà giải trong khi người dùng đang chờ, độ trễ đuôi sẽ bùng nổ.
2. **Bộ nhớ mục ruỗng.** Các lượt ghi cứ tích tụ. Sự thật bị mâu thuẫn vẫn ở lại. Truy xuất bị nhấn chìm trong nội dung cũ.
3. **Mất cấu trúc.** Một kho archival phẳng không thể diễn đạt "block Human luôn nằm trong prompt; block Persona luôn nằm trong prompt; block Task thay đổi theo từng phiên."

Letta (letta.com) là tên nền tảng mà dự án MemGPT gốc đã nhận từ năm 2024 — mẫu trong bài báo vẫn giữ tên MemGPT — còn bản viết lại Letta V1 năm 2026 là một bước sau đó, tách biệt. Memory block làm cấu trúc trở nên tường minh; sleep-time compute đưa consolidation ra khỏi critical path.

## Khái niệm

### Ba tầng

| Tầng | Phạm vi | Nơi nó sống | Ai ghi |
|------|-------|----------------|------------|
| Core | Luôn hiển thị | Bên trong prompt chính | Agent gọi tool + ghi lại lúc nghỉ |
| Recall | Lịch sử hội thoại | Truy xuất được | Tự động ghi log từng lượt |
| Archival | Sự thật bất kỳ | Vector + KV + graph | Agent gọi tool + nạp vào lúc nghỉ |

Core chính là core memory (bộ nhớ lõi) của MemGPT. Recall là buffer hội thoại cùng phần đuôi đã bị đẩy ra. Archival là kho bên ngoài. Cách chia này dọn dẹp sự quá tải hai tầng của MemGPT.

### Memory blocks

Một block là một phần có kiểu, bền vững, sửa được của tầng core. Bài báo MemGPT gốc định nghĩa hai block:

- **Human block** — sự thật về người dùng (tên, vai trò, sở thích, mục tiêu).
- **Persona block** — tự nhận thức của agent (danh tính, giọng điệu, ràng buộc).

Letta tổng quát hoá thành các block tuỳ ý do người dùng định nghĩa: một block `Task` cho mục tiêu hiện tại, một block `Project` cho sự thật về codebase, một block `Safety` cho các ràng buộc cứng. Mỗi block có `id`, `label`, `value`, `limit` (giới hạn ký tự), `description` (để model biết khi nào nên sửa).

Các block sửa được qua bề mặt tool:

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)` — cô đọng một block đang gần chạm giới hạn.

### Sleep-time compute

Bổ sung năm 2025 của Letta: chạy một agent thứ hai ở chế độ nền, ngoài critical path. Agent lúc nghỉ xử lý bản ghi hội thoại và ngữ cảnh codebase, ghi `learned_context` vào các block chia sẻ, rồi cố kết hoặc vô hiệu hoá các bản ghi archival.

Các tính chất kéo theo:

- **Không tốn độ trễ.** Phản hồi chính không phải chờ thao tác bộ nhớ.
- **Được phép dùng model mạnh hơn.** Agent lúc nghỉ có thể là một model đắt hơn, chậm hơn, vì nó không bị ràng buộc độ trễ.
- **Cửa sổ cố kết tự nhiên.** Khử trùng lặp, tóm tắt, vô hiệu hoá sự thật mâu thuẫn khi người dùng không chờ.

Hình dạng này khớp với cách con người làm việc: bạn làm nhiệm vụ, bạn ngủ một giấc, bộ nhớ dài hạn lắng lại qua đêm.

### Native reasoning

Letta V1 (`letta_v1_agent`, 2026) loại bỏ `send_message`/heartbeat và các token `Thought:` nội tuyến để chuyển sang native reasoning. Responses API (OpenAI) và Messages API với extended thinking (Anthropic) phát ra reasoning trên một kênh riêng, được truyền qua các lượt (mã hoá giữa các provider trong production). Vòng điều khiển vẫn là ReAct. Vết suy nghĩ mang tính cấu trúc, không phải dạng prompt.

### Mẫu này hỏng ở đâu

- **Block phình to.** `block_append` vô hạn chạm giới hạn rất nhanh. Nối một bộ tóm tắt block trước lượt ghi đẩy vượt mức trần.
- **Trôi lặng lẽ.** Agent lúc nghỉ viết lại một block mà agent chính không bao giờ nhận ra. Hãy tạo version cho block và hiển thị diff trong trace.
- **Cố kết bị đầu độc.** Agent lúc nghỉ xử lý nội dung mà kẻ tấn công chạm tới rồi đưa vào core. Bài 27 cũng áp dụng cho bề mặt lúc nghỉ.

```figure
memory-blocks
```

## Tự tay làm

`code/main.py` triển khai:

- `Block` — id, label, value, limit, description.
- `BlockStore` — CRUD + helper `near_limit(label)`.
- Hai agent theo kịch bản — `PrimaryAgent` phục vụ một lượt, `SleepTimeAgent` cố kết giữa các lượt.
- Một trace cho thấy hội thoại ba lượt với các lượt ghi block, cộng một lượt lúc nghỉ tóm tắt một block và vô hiệu hoá một sự thật cũ.

Chạy:

```
python3 code/main.py
```

Bản ghi cho thấy sự phân chia: các lượt chính nhanh và tạo ra ghi thô; lượt lúc nghỉ nén lại và dọn dẹp.

## Áp dụng

- **Letta** (letta.com) cho bản triển khai tham chiếu. Tự lưu trữ hoặc dùng cloud có quản lý.
- **Claude Agent SDK skills** như kiến thức dạng block — một skill là một block hướng dẫn có tên, có version, truy xuất được mà agent nạp khi cần.
- **Bản tự xây** cho những đội muốn kiểm soát backend lưu trữ. Dùng hợp đồng API của Letta để sau này còn chuyển đi.

## Đóng gói

`outputs/skill-memory-blocks.md` sinh ra một hệ block mang dáng Letta với các hook lúc nghỉ cho bất kỳ runtime nào, gồm cả luật an toàn và cách nối citation.

## Bài tập

1. Thêm một tool `block_summarize` thay giá trị block bằng bản tóm tắt do model sinh khi `near_limit` trả về true. Ngưỡng kích hoạt nào tối thiểu hoá cả số lần gọi tóm tắt lẫn việc block tràn?
2. Triển khai khử trùng lặp lúc nghỉ trên archival: hai bản ghi có phần text trùng token >90% gộp thành một. Chỉ làm trong lượt lúc nghỉ, không bao giờ trên critical path.
3. Tạo version cho block. Ở mỗi lượt ghi, lưu giá trị cũ và một diff. Mở `block_history(label)` để người vận hành debug được "vì sao agent quên X."
4. Coi agent lúc nghỉ là người ghi không đáng tin. Khi chúng chạm vào block Persona hoặc Safety, yêu cầu một agent thứ hai duyệt trước khi commit.
5. Chuyển ví dụ sang dùng Letta API (`letta_v1_agent`). Schema block thay đổi gì, và native reasoning làm hình dạng trace đổi thế nào?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Memory block | "Phần prompt sửa được" | Đoạn bộ nhớ lõi có kiểu, bền vững, LLM sửa được |
| Human block | "Bộ nhớ người dùng" | Sự thật về người dùng, được ghim trong core |
| Persona block | "Danh tính agent" | Tự nhận thức, giọng điệu, ràng buộc, được ghim trong core |
| Sleep-time compute | "Việc bộ nhớ chạy nền" | Agent thứ hai làm consolidation ngoài critical path |
| Core / Recall / Archival | "Các tầng" | Chia bộ nhớ ba lớp: luôn hiển thị / hội thoại / bên ngoài |
| Block limit | "Trần" | Giới hạn ký tự mỗi block; buộc phải tóm tắt |
| Native reasoning | "Kênh suy nghĩ" | Reasoning do provider phát ra, không phải `Thought:` ở mức prompt |
| Learned context | "Đầu ra lúc nghỉ" | Sự thật agent lúc nghỉ ghi vào các block chia sẻ |

## Đọc thêm

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — mẫu block
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute) — cố kết bất đồng bộ
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent) — bản viết lại với native reasoning
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — nguồn gốc
