# Gọi công cụ song song và truyền trực tiếp với công cụ

> Ba lượt tra thời tiết độc lập chạy tuần tự là ba vòng khứ hồi. Chạy chúng song song thì tổng thời gian tụt xuống bằng lần gọi chậm nhất. Mọi nhà cung cấp mô hình hàng đầu hiện đều phát ra nhiều lần gọi công cụ trong một lượt. Lợi ích là thật; phần đường ống thì tinh tế. Bài này đi qua cả hai nửa: gọi công cụ song song (parallel tool call) toả ra, và việc lắp lại tham số được truyền trực tiếp (streaming), nhấn mạnh cái bẫy tương quan id.

**Type:** Build
**Languages:** Python (stdlib, thread pool + streaming harness)
**Prerequisites:** Giai đoạn 13, Bài 02 (đi sâu vào function calling)
**Time:** ~75 phút

## Mục tiêu học tập

- Giải thích vì sao `parallel_tool_calls: true` tồn tại và khi nào nên tắt nó.
- Ghép các mảnh tham số truyền trực tiếp về đúng id của lần gọi công cụ trong lúc toả song song.
- Lắp các chuỗi `arguments` dở dang thành JSON hoàn chỉnh mà không phân tích cú pháp sớm.
- Chạy benchmark thời tiết ba thành phố minh hoạ độ trễ tuần tự so với song song.

## Vấn đề

Không có gọi song song, một agent trả lời "thời tiết ở Bengaluru, Tokyo và Zurich thế nào" sẽ làm thế này:

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

Ba vòng khứ hồi tới LLM, mỗi vòng còn phải trả giá độ trễ của executor. Tổng thời gian thực tế rơi vào khoảng 4 lần so với mức lý tưởng.

Với gọi song song:

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

Một vòng khứ hồi tới LLM. Thời gian executor là mức lớn nhất trong ba, không phải tổng. Các benchmark thực tế trên OpenAI, Anthropic và Gemini cho thấy giảm 60 đến 70 phần trăm thời gian thực tế trên các workload toả ra.

Cái giá là độ phức tạp tương quan. Khi ba lần gọi xong không theo thứ tự (ordering), kết quả của bạn phải mang đúng `tool_call_id` để mô hình ghép chúng lại. Khi kết quả được truyền trực tiếp, bạn phải lắp các mảnh tham số dở dang thành JSON hoàn chỉnh trước khi thực thi. Gemini 3 thêm id duy nhất một phần để giải quyết vấn đề thực tế là hai lần gọi song song tới cùng một công cụ không phân biệt được.

## Khái niệm

### Bật chế độ song song

- **OpenAI.** `parallel_tool_calls: true` bật mặc định. Đặt `false` để buộc chạy tuần tự.
- **Anthropic.** Song song qua `disable_parallel_tool_use: false` (mặc định bật trên Claude 3.5 trở lên). Đặt `true` để chạy tuần tự.
- **Gemini.** Luôn có khả năng song song; `tool_config.function_calling_config.mode = "AUTO"` để mô hình tự quyết định.

Tắt song song khi các công cụ có phụ thuộc thứ tự (`create_file` rồi `write_file`), khi đầu ra của lần gọi này làm đầu vào cho lần gọi kia, hoặc khi bộ giới hạn tốc độ không chịu nổi việc toả ra.

### Tương quan id

Mỗi lần gọi mô hình phát ra đều có một `id`. Mỗi kết quả host trả về phải kèm đúng id đó. Không có nó, kết quả sẽ mơ hồ.

- **OpenAI.** `tool_call_id` (mã định danh lần gọi công cụ) trên mỗi message role tool.
- **Anthropic.** `tool_use_id` trên mỗi block `tool_result`.
- **Gemini.** `id` trên mỗi `functionResponse` (Gemini 3 trở lên; Gemini 2 ghép theo tên nên hỏng với các lần gọi song song trùng tên).

### Chạy các lần gọi đồng thời

Host chạy executor của mỗi lần gọi trên thread, coroutine hoặc worker từ xa riêng. Bộ khung đơn giản nhất dùng thread pool; môi trường production dùng asyncio với `asyncio.gather` hoặc structured concurrency. Thứ tự hoàn thành không đoán trước được — id chính là định danh.

Một lỗi phổ biến: trả kết quả theo thứ tự danh sách gọi thay vì thứ tự hoàn thành. Cách này thường vẫn chạy vì mô hình chỉ quan tâm `tool_call_id`, nhưng nếu một kết quả bị mất hoặc bị nhân đôi, việc gửi sai thứ tự khiến gỡ lỗi khó hơn. Nên trả theo thứ tự hoàn thành kèm id tường minh.

### Truyền trực tiếp các lần gọi công cụ

Khi mô hình truyền trực tiếp, `arguments` đến thành từng mảnh. Ba dòng mảnh riêng biệt cho ba lần gọi song song đan xen trên đường truyền. Bạn cần một accumulator cho mỗi id.

Hình dạng theo từng nhà cung cấp:

- **OpenAI.** Mỗi chunk là `choices[0].delta.tool_calls[i].function.arguments` (chuỗi dở dang; `delta` là phần gia tăng gửi thêm cho mỗi lần gọi). Chunk mang `index` (vị trí trong danh sách gọi). Bạn tích luỹ theo index, đọc `id` khi nó xuất hiện lần đầu, và phân tích JSON khi `finish_reason = "tool_calls"`.
- **Anthropic.** Các sự kiện stream là `message_start`, rồi một `content_block_start` cho mỗi block với type `tool_use` (chứa id, name, input rỗng). Các sự kiện `content_block_delta` mang các mảnh `input_json_delta`. `content_block_stop` đóng mỗi block.
- **Gemini.** `streamFunctionCallArguments` (Gemini 3 trở lên) phát các chunk kèm `functionCallId` để các lần gọi đan xen gọn gàng. Trước Gemini 3, truyền trực tiếp trả về mỗi lần một lần gọi hoàn chỉnh.

### JSON dở dang và cái bẫy phân tích sớm

Bạn không thể phân tích `arguments` cho tới khi nó hoàn chỉnh. JSON dở dang (partial JSON) như `{"city": "Beng` là không hợp lệ và sẽ ném lỗi. Cửa chặn đúng là tín hiệu kết thúc lần gọi của nhà cung cấp: `finish_reason = "tool_calls"` của OpenAI, `content_block_stop` của Anthropic, hay sự kiện kết thúc stream của Gemini. Chỉ khi đó mới thử `json.loads`. Một cách vững hơn là dùng bộ phân tích JSON tăng dần, phát sự kiện khi cấu trúc hoàn thành; hướng dẫn streaming của OpenAI khuyến nghị cách này cho UX hiển thị chỉ báo "đang suy nghĩ" trực tiếp. Đếm dấu ngoặc là phép thử tính đầy đủ không đáng tin (dấu ngoặc trong chuỗi được trích hoặc nội dung escaped gây dương tính giả) và chỉ nên dùng như một mẹo gỡ lỗi không chính thức.

### Hoàn thành không theo thứ tự

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

Kết quả trả về của host vẫn phải nêu rõ các id:

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

Thứ tự trong kết quả trả về không ảnh hưởng tính đúng đắn trên OpenAI hay Anthropic. Gemini nhận mọi thứ tự miễn là các id khớp.

### Benchmark: tuần tự so với song song

Bộ khung trong `code/main.py` mô phỏng ba executor với độ trễ 400, 600 và 800 ms. Chạy tuần tự mất tổng cộng 1800 ms. Chạy song song mất max(400, 600, 800) = 800 ms. Chênh lệch là hằng số, không tỉ lệ, nên tiết kiệm càng tăng khi số công cụ tăng.

Lưu ý thực tế: gọi song song gây áp lực lên các API phía sau. Một lần toả 10 nhánh vào một dịch vụ giới hạn tốc độ sẽ thất bại. Giai đoạn 13 · 17 đề cập backpressure ở mức gateway; ngữ nghĩa retry dự kiến nằm ở một giai đoạn sau.

### Thời gian thực tế khi toả song song có truyền trực tiếp

Nếu bản thân mô hình truyền trực tiếp, bạn có thể bắt đầu thực thi ngay khi tham số của một lần gọi hoàn chỉnh, thay vì chờ mọi lần gọi kết thúc. Đây là tối ưu mà OpenAI có ghi tài liệu nhưng không phải SDK nào cũng mở ra. Bộ khung trong bài này làm vậy: ngay khi luồng mô phỏng phát ra một object tham số hoàn chỉnh, host khởi chạy lần gọi đó.

```figure
tp-parallel-fanout
```

## Áp dụng

`code/main.py` có hai nửa. Nửa đầu chạy ba lần gọi thời tiết mô phỏng tuần tự và song song bằng `concurrent.futures.ThreadPoolExecutor` rồi in thời gian thực tế. Nửa sau phát lại một phản hồi stream giả — các mảnh `arguments` của ba lần gọi song song đan xen trên một stream — và lắp chúng lại theo id bằng `StreamAccumulator`. Không LLM, không mạng, chỉ logic lắp lại.

Những điểm cần xem:

- Bộ đếm thời gian tuần tự chạm 1.8 giây. Bộ đếm song song chạm 0.8 giây với cùng độ trễ giả lập.
- Accumulator xử lý các mảnh đến không theo thứ tự bằng cách đệm theo id và chỉ phân tích khi JSON của mỗi lần gọi đã hoàn chỉnh.
- Executor khởi chạy ngay khi tham số của một id hoàn tất, không phải sau khi mọi stream kết thúc.

## Đóng gói

Bài này tạo ra `outputs/skill-parallel-call-safety-check.md`. Cho một registry công cụ, skill kiểm tra những công cụ nào an toàn để chạy song song, công cụ nào có phụ thuộc thứ tự, và công cụ nào sẽ làm quá tải giới hạn tốc độ phía sau — trả về một registry đã chỉnh với cờ `parallel_safe` cho từng công cụ.

## Bài tập

1. Chạy `code/main.py` và thay đổi độ trễ giả lập. Xác nhận rằng tỉ lệ song song trên tuần tự xấp xỉ `max/sum` (lần chạy thật lệch một chút so với lý tưởng do lập lịch thread, tuần tự hoá và chi phí của bộ khung). Với phân bố độ trễ nào thì song song không còn đáng kể?

2. Mở rộng accumulator để xử lý trường hợp "lần gọi bị huỷ giữa stream" bằng cách bỏ đệm của nó và phát sự kiện `cancelled`. Nhà cung cấp nào ghi tài liệu rõ trường hợp này? Xem ngữ nghĩa `content_block_stop` của Anthropic và hành vi `finish_reason: "length"` của OpenAI.

3. Thay thread pool bằng `asyncio.gather`. Benchmark cả hai. Bạn sẽ thấy lợi ích nhỏ với async nhờ chi phí chuyển ngữ cảnh thấp hơn, nhưng chỉ khi executor thực sự làm I/O.

4. Chọn hai công cụ KHÔNG nên chạy song song (ví dụ `create_file` rồi `write_file`). Thêm đồ thị `ordering_dependency` vào registry và chặn việc toả song song dựa trên đồ thị đó. Đây là bộ máy tối thiểu cho lập lịch có nhận biết phụ thuộc, thứ mà một giai đoạn về agent engineering sau này sẽ hình thức hoá.

5. Đọc phần parallel function calling của OpenAI và tài liệu `disable_parallel_tool_use` của Anthropic. Xác định một loại công cụ thực tế mà Anthropic khuyến nghị tắt song song. (Gợi ý: các thay đổi có hệ quả trên cùng một tài nguyên.)

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Parallel tool calls | "Fan-out trong một lượt" | Mô hình phát ra nhiều lần gọi công cụ trong một message assistant duy nhất |
| `parallel_tool_calls` | "Cờ của OpenAI" | Bật hoặc tắt việc phát nhiều lần gọi |
| `disable_parallel_tool_use` | "Cờ nghịch của Anthropic" | Cờ chọn không dùng; mặc định là bật song song |
| Tool call id | "Tay nắm tương quan" | Mã định danh mỗi lần gọi mà message kết quả phải lặp lại |
| Accumulator | "Bộ đệm stream" | Bộ đệm chuỗi theo id cho các mảnh `arguments` dở dang |
| Out-of-order completion | "Nhanh nhất về trước" | Các lần gọi song song kết thúc theo thứ tự không đoán trước; id là keo dính |
| Dependency graph | "Ràng buộc thứ tự" | Công cụ có đầu ra chảy vào đầu vào của công cụ khác; không thể chạy song song |
| Parse-early trap | "JSON.parse nổ tung" | Cố phân tích một chuỗi `arguments` chưa hoàn chỉnh |
| `streamFunctionCallArguments` | "Tính năng Gemini 3" | Các mảnh tham số truyền trực tiếp với id duy nhất cho mỗi lần gọi |
| Completion-order reply | "Đừng chờ tất cả" | Trả kết quả ngay khi chúng đến, khoá theo id |

## Đọc thêm

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — hành vi mặc định và cờ chọn không dùng
- [Anthropic — Parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use) — `disable_parallel_tool_use` và gom kết quả theo lô
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — các lần gọi song song tương quan theo id từ Gemini 3
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — lắp lại tham số theo chunk cho stream của OpenAI
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — `content_block_delta` với `input_json_delta`

---

[← Bài trước: Đi sâu vào Function Calling (gọi hàm)](../../02-function-calling-deep-dive/docs/vi.md) · [Bài sau: Đầu ra có cấu trúc →](../../04-structured-output/docs/vi.md)
