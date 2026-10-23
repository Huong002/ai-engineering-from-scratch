# Giao diện công cụ — Vì sao agent cần I/O có cấu trúc

> Một mô hình ngôn ngữ sinh ra token. Một chương trình thì thực hiện hành động. Khoảng cách giữa hai thứ đó chính là giao diện tool (công cụ): một khế ước cho phép mô hình yêu cầu một hành động và để host thực thi nó. Mọi stack năm 2026 — function calling trên OpenAI, Anthropic và Gemini; `tools/call` của MCP; các task part của A2A — đều là những cách mã hoá khác nhau của cùng một vòng lặp bốn bước. Bài này gọi tên vòng lặp đó và trình bày bộ máy tối thiểu để chạy nó.

**Type:** Learn
**Languages:** Python (stdlib, no LLM)
**Prerequisites:** Giai đoạn 11 (API hoàn thành của LLM)
**Time:** ~45 phút

## Mục tiêu học tập

- Giải thích vì sao một LLM chỉ biết sinh văn bản thì tự nó không thể hành động lên thế giới thực.
- Vẽ vòng lặp tool call (lượt gọi công cụ) bốn bước (describe → decide → execute → observe) và chỉ ra ai sở hữu từng bước.
- Viết mô tả tool thành ba phần: tên, đầu vào JSON Schema, và một hàm thực thi tất định.
- Phân biệt tool thuần và tool có tác dụng phụ, và nêu vì sao cách chia này quan trọng cho an toàn.

## Vấn đề

Một LLM phát ra phân bố xác suất cho token tiếp theo. Đó là toàn bộ bề mặt đầu ra của nó. Nếu bạn hỏi một mô hình chat "thời tiết ở Bengaluru ngay bây giờ thế nào", nó có thể viết một câu nghe hợp lý, nhưng nó không thể gọi vào một API thời tiết. Câu đó có thể đúng nhờ trùng hợp, hoặc đã cũ ba ngày.

Lấp khoảng cách đó chính là mục đích của giao diện tool. Chương trình host — runtime agent của bạn, Claude Desktop, ChatGPT, Cursor, hay một script tự viết — công bố một danh sách tool có thể gọi cho mô hình. Khi quyết định cần hành động, mô hình phát ra một payload có cấu trúc nêu tên tool và các đối số của nó. Host phân tích payload đó, chạy tool thật, rồi đưa kết quả trở lại. Vòng lặp tiếp tục cho tới khi mô hình quyết định không cần gọi thêm.

Phiên bản đầu của khế ước này ra mắt tháng 6 năm 2023 dưới dạng tham số "functions" của OpenAI. Anthropic tiếp bước với các block `tool_use` trong Claude 2.1. Gemini thêm `functionDeclarations` vài tháng sau. Mọi nhà cung cấp giờ đều phơi ra cùng một hình dạng: một danh sách tool kiểu JSON Schema đưa vào, một tool call kiểu JSON payload trả ra. Model Context Protocol (tháng 11 năm 2024) tổng quát hoá khế ước để một registry tool phục vụ được mọi mô hình. A2A (tháng 4 năm 2026, v1.0) xếp nguyên thuỷ đó lên một tầng nữa cho việc uỷ thác agent-với-agent.

Vòng lặp bốn bước là bất biến nằm dưới tất cả những thứ đó. Mọi thứ còn lại trong Giai đoạn 13 chỉ là phần triển khai chi tiết.

## Khái niệm

### Bước một: mô tả

Host khai báo mỗi tool bằng ba trường.

- **Name.** Một định danh ổn định, máy đọc được. `get_weather`, chứ không phải "weather thing".
- **Description.** Một bản tóm tắt bằng ngôn ngữ tự nhiên, dài một đoạn văn. "Dùng khi người dùng hỏi về điều kiện thời tiết hiện tại của một thành phố cụ thể. Không dùng cho dữ liệu lịch sử."
- **Input schema.** Một đối tượng JSON Schema (draft 2020-12) mô tả các đối số của tool.

Mô hình nhận danh sách đó. Các nhà cung cấp hiện đại tuần tự hoá những khai báo này vào system prompt bằng một mẫu riêng của họ, nên bạn — với vai trò bên gọi — chỉ phải làm việc với dạng có cấu trúc.

### Bước hai: quyết định

Với tin nhắn của người dùng và các tool sẵn có, mô hình chọn một trong ba hành vi.

1. **Trả lời trực tiếp** bằng văn bản. Không có tool call.
2. **Gọi một hoặc nhiều tool.** Phát ra các đối tượng call có cấu trúc. Dưới `parallel_tool_calls: true` (mặc định bật trên OpenAI và Gemini, phải bật tuỳ chọn trên Anthropic), mô hình có thể phát nhiều call trong một lượt.
3. **Từ chối.** Đầu ra có cấu trúc ở strict mode có thể tạo một block `refusal` có kiểu thay vì một call.

Payload của tool call có ba trường ổn định: `id` của call, `name` của tool, và một đối tượng JSON `arguments`. Id tồn tại để host ghép kết quả về sau với đúng call cụ thể — điều này quan trọng khi các call song song trả về không theo thứ tự.

### Bước ba: thực thi

Host nhận call, kiểm tra đối số với schema (lược đồ) đã khai báo, rồi chạy executor. Đối số không hợp lệ nghĩa là mô hình bịa ra một trường hoặc dùng sai kiểu — một kiểu lỗi rất phổ biến ở các mô hình yếu. Host production xử lý đối số không hợp lệ theo một trong ba cách: thất bại nhanh và đưa lỗi về cho mô hình, sửa JSON bằng một parser bị ràng buộc, hoặc gọi lại mô hình với lỗi kiểm tra nằm trong prompt.

Bản thân executor chỉ là code bình thường. Python, TypeScript, một lệnh shell, một truy vấn database. Nó tạo ra một kết quả, thường là string nhưng có thể là bất kỳ giá trị JSON nào hoặc một content block có cấu trúc (text, image, hay tham chiếu resource trong MCP). Kết quả phải tuần tự hoá được.

### Bước bốn: quan sát

Host nối tool result (kết quả công cụ) vào cuộc hội thoại (dưới dạng một tin nhắn role `tool` với `id` khớp) rồi gọi lại mô hình. Mô hình giờ đã có đầu ra của tool trong ngữ cảnh và có thể đưa câu trả lời cuối cùng hoặc yêu cầu thêm call. Việc này tiếp diễn cho tới khi mô hình ngừng phát call hoặc host chạm giới hạn an toàn về số lần lặp.

### Ranh giới tin cậy

Tool có hai loại cần phân biệt vì lý do an toàn.

- **Pure.** Chỉ đọc, tất định, không tác dụng phụ. `get_weather`, `search_docs`, `get_current_time`. An toàn để gọi thử trước.
- **Consequential.** Làm thay đổi trạng thái, tiêu tiền, chạm vào dữ liệu người dùng. `send_email`, `delete_file`, `execute_trade`. Phải có cổng chặn.

"Rule of Two" của Meta (2026) về an ninh agent nói rằng một lượt chỉ được kết hợp tối đa hai trong ba thứ: đầu vào không tin cậy, dữ liệu nhạy cảm, hành động hệ trọng. Giao diện tool là nơi bạn thực thi quy tắc đó — bằng cách từ chối call, yêu cầu người dùng xác nhận, hoặc nâng phạm vi quyền. Xem Giai đoạn 13 · 15 cho chương bảo mật đầy đủ và Giai đoạn 14 · 09 cho chính sách quyền ở mức agent.

### Vòng lặp sống ở đâu

| Ngữ cảnh | Ai mô tả | Ai quyết định | Ai thực thi |
|---------|---------------|-------------|--------------|
| Function calling một lượt (OpenAI/Anthropic/Gemini) | Lập trình viên ứng dụng | LLM | Lập trình viên ứng dụng |
| MCP | MCP server | LLM qua MCP client | MCP server |
| A2A | Nhà phát hành Agent Card | Agent gọi | Agent được gọi |
| Trình duyệt web (agent dùng function calling) | Tiện ích trình duyệt / WebMCP | LLM | Runtime trình duyệt |

Ở đâu cũng vậy, vẫn bốn bước đó. Tên cột có thể đổi; cấu trúc thì không.

### Vì sao không chỉ nhắc mô hình phát ra JSON?

"Bảo mô hình trả lời bằng JSON" là cách làm thời trước function calling. Nó hỏng khoảng 5 đến 15 phần trăm số lần trên các mô hình frontier và nhiều hơn nữa trên mô hình nhỏ hơn. Các kiểu hỏng gồm thiếu ngoặc nhọn, dấu phẩy thừa, trường bịa ra, và sai kiểu. Khi đó bạn cần một lượt sửa JSON, một lần thử lại, hoặc một decoder bị ràng buộc.

Function calling gốc tốt hơn vì ba lý do. Thứ nhất, nhà cung cấp huấn luyện mô hình end-to-end trên đúng hình dạng call, nên tỉ lệ JSON hợp lệ leo lên 98 đến 99 phần trăm ở strict mode. Thứ hai, payload của call nằm trong khe giao thức riêng, không nằm trong văn bản tự do — nên một tool call không bao giờ rò rỉ vào câu trả lời mà người dùng thấy. Thứ ba, nhà cung cấp áp đặt việc tuân thủ schema bằng constrained decoding (strict mode của OpenAI, `tool_use` của Anthropic, `responseSchema` của Gemini). Đầu ra được bảo đảm hợp lệ.

Giai đoạn 13 · 02 đi qua ba API nhà cung cấp song song. Giai đoạn 13 · 04 đi sâu vào đầu ra có cấu trúc.

### Cầu dao ngắt mạch

Vòng lặp dừng khi mô hình ngừng phát call hoặc host chạm số lượt tối đa. Host production đặt con số này trong khoảng 5 đến 20 lượt. Vượt quá đó, gần như chắc chắn bạn đang ở trong một vòng lặp mà mô hình không thể thoát ra. Claude Code mặc định 20; OpenAI Assistants là 10; chế độ agent của Cursor là 25.

Phương án còn lại — vòng lặp không giới hạn — cứ sáu tháng lại xuất hiện dưới dạng bài mổ xẻ "agent tiêu $400 tiền gọi API qua một đêm". Đừng đưa lên production mà thiếu giới hạn.

Giai đoạn 14 · 12 nói sâu về phục hồi lỗi và tự chữa lành; Giai đoạn 17 nói về rate limit production.

### Giai đoạn 13 đi tiếp từ đây

- Bài 02 đến 05 trau chuốt bề mặt tool call ở mức nhà cung cấp.
- Bài 06 đến 14 tổng quát hoá vòng lặp thành MCP.
- Bài 15 đến 18 phòng thủ vòng lặp trước server thù địch, người dùng đối kháng, và các bề mặt xác thực từ xa không được xác thực.
- Bài 19 đến 22 mở rộng mẫu này sang cộng tác agent-với-agent, khả năng quan sát, định tuyến, và đóng gói.
- Bài 23 đóng gói một hệ sinh thái hoàn chỉnh dùng mọi nguyên thuỷ.

Mọi bài còn lại đều là phần triển khai chi tiết của vòng lặp bốn bước này. Hãy giữ nó trong đầu như một bất biến.

```figure
tp-tool-loop
```

## Áp dụng

`code/main.py` chạy vòng lặp bốn bước mà không cần LLM. Một hàm "decider" giả lập mô hình bằng cách so khớp mẫu trên tin nhắn người dùng; executor, bộ kiểm tra schema, và harness cho bước observe là thật. Chạy nó để xem toàn bộ vũ đạo request/response với trạng thái trung gian in ra được, rồi thay decider giả bằng một nhà cung cấp thật trong bài sau.

Cần chú ý:

- Registry tool giữ ba trường cho mỗi tool: name, description, schema, và một tham chiếu tới executor.
- Bộ kiểm tra là một tập con JSON Schema tối thiểu (types, required, enum, min/max) viết chỉ bằng stdlib. Giai đoạn 13 · 04 cung cấp bản đầy đủ hơn.
- Vòng lặp giới hạn số lần lặp ở năm. Agent production cần đúng kiểu cầu dao ngắt mạch này.

## Đóng gói

Bài này tạo ra `outputs/skill-tool-interface-reviewer.md`. Nhận một bản nháp định nghĩa tool (name + description + schema + phác thảo executor), skill này rà nó xem có hợp với vòng lặp không: tên có ổn định với máy không, mô tả có phải bản hướng dẫn sử dụng đầy đủ không, schema có dùng JSON Schema 2020-12 đúng không, và phân loại pure-vs-consequential có tường minh không.

## Bài tập

1. Thêm một tool thứ tư vào `code/main.py` tên là `get_stock_price(ticker)`. Viết mô tả của nó là "Dùng khi người dùng hỏi giá cổ phiếu hiện tại theo mã. Không dùng cho giá lịch sử hay tóm tắt thị trường." Chạy harness và xác nhận decider giả định tuyến các truy vấn có nhắc mã cổ phiếu sang tool mới.

2. Phá bộ kiểm tra schema. Truyền một call có đối tượng `arguments` thiếu một trường bắt buộc, và xác nhận host từ chối nó trước khi thực thi. Rồi truyền một call có thêm một trường lạ. Quyết định: host nên từ chối hay bỏ qua? Biện luận cho lựa chọn của bạn bằng một lập luận về an toàn.

3. Phân loại từng tool trong harness là pure hay consequential. Thêm cờ `consequential: true` vào những mục registry cần nó, và đổi vòng lặp để in dòng "would confirm with user" mỗi khi một tool consequential được chọn. Đây chính là hình dạng của cổng xác nhận mà mọi host production đều cần.

4. Vẽ vòng lặp bốn bước ra giấy, điền bảng theo cột nhà cung cấp ở trên cho client bạn thích nhất (Claude Desktop, Cursor, ChatGPT, hay một stack tự viết). Đối chiếu với biến thể riêng cho MCP ở Giai đoạn 13 · 06.

5. Đọc hướng dẫn function calling của OpenAI từ đầu đến cuối. Tìm trường duy nhất nằm trong request nhưng không có trong vòng lặp bốn bước như trình bày ở đây. Giải thích nó thêm gì và vì sao nó tiện chứ không thiết yếu.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Tool | "Một thứ mô hình có thể gọi" | Bộ ba: name + đầu vào kiểu JSON Schema + hàm executor |
| Function calling | "Cách dùng tool gốc của nhà cung cấp" | Hỗ trợ API ở mức nhà cung cấp để phát ra tool call có cấu trúc thay vì văn xuôi |
| Tool call | "Yêu cầu hành động của mô hình" | Một JSON payload với `id`, `name`, `arguments` do mô hình phát ra |
| Tool result | "Thứ tool trả về" | Đầu ra của executor, được bọc trong tin nhắn role `tool` với id khớp |
| Parallel tool calls | "Nhiều call cùng lúc" | Nhiều đối tượng call trong một lượt của mô hình, độc lập và sắp được theo id |
| Strict mode | "JSON được bảo đảm" | Constrained decoding buộc đầu ra của mô hình phải hợp lệ với schema đã khai báo |
| Pure tool | "Tool chỉ đọc" | Không tác dụng phụ; an toàn để chạy lại |
| Consequential tool | "Tool hành động" | Làm thay đổi trạng thái bên ngoài; cần cổng chặn, kiểm toán, hoặc xác nhận của người dùng |
| Four-step loop | "Chu trình tool call" | describe → decide → execute → observe |
| Host | "Runtime của agent" | Chương trình giữ registry tool, gọi mô hình, và chạy executor |

## Đọc thêm

- [OpenAI — Hướng dẫn function calling](https://platform.openai.com/docs/guides/function-calling) — tài liệu tham chiếu chuẩn cho khai báo tool và hình dạng call kiểu OpenAI
- [Anthropic — Tổng quan tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — định dạng block `tool_use` / `tool_result` của Claude
- [Google — Function calling trong Gemini](https://ai.google.dev/gemini-api/docs/function-calling) — `functionDeclarations` và ngữ nghĩa gọi song song trong Gemini
- [Model Context Protocol — Đặc tả 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) — bản tổng quát hoá không trạng thái, độc lập nhà cung cấp hiện tại của giao diện tool
- [JSON Schema — Ghi chú phát hành 2020-12](https://json-schema.org/draft/2020-12/release-notes) — phương ngữ schema mà mọi API tool hiện đại đều dùng

---

[Bài sau: Đi sâu vào Function Calling (gọi hàm) →](../../02-function-calling-deep-dive/docs/vi.md)
