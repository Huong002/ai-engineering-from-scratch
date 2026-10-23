# Đi sâu vào Function Calling (gọi hàm) — OpenAI, Anthropic, Gemini

> Ba nhà cung cấp tiên phong đều hội tụ về cùng một vòng lặp gọi công cụ vào năm 2024 rồi rẽ hướng ở mọi thứ còn lại. OpenAI dùng `tools` và `tool_calls`. Anthropic dùng các khối `tool_use` và `tool_result`. Gemini dùng `functionDeclarations` và tương quan bằng id duy nhất. Bài này so sánh ba bên cạnh nhau để code chạy trên một nhà cung cấp không vỡ khi bạn chuyển sang bên khác.

**Type:** Build
**Languages:** Python (stdlib, schema translators)
**Prerequisites:** Giai đoạn 13, Bài 01 (giao diện công cụ)
**Time:** ~75 phút

## Mục tiêu học tập

- Nêu ba khác biệt về hình dạng giữa payload gọi hàm của OpenAI, Anthropic và Gemini (khai báo, lệnh gọi, kết quả).
- Chuyển một khai báo công cụ sang cả ba định dạng nhà cung cấp và dự đoán nơi ràng buộc strict mode sẽ khác nhau.
- Dùng `tool_choice` ở từng nhà cung cấp để ép, cấm hoặc để tự chọn lệnh gọi công cụ.
- Biết giới hạn cứng của từng nhà cung cấp (số công cụ, độ sâu schema, độ dài đối số) và chữ ký lỗi mà mỗi bên phát ra khi vi phạm giới hạn.

## Vấn đề

Hình dạng của một yêu cầu gọi hàm khác nhau theo nhà cung cấp. Ba ví dụ cụ thể từ các stack production năm 2026:

**OpenAI Chat Completions / Responses API.** Bạn truyền `tools: [{type: "function", function: {name, description, parameters, strict}}]`. Phản hồi của model chứa `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]` trong đó `arguments` là một chuỗi JSON bạn phải parse. Strict mode (chế độ nghiêm ngặt) — bật bằng `strict: true` — áp đặt việc tuân thủ schema thông qua giải mã có ràng buộc.

**Anthropic Messages API.** Bạn truyền `tools: [{name, description, input_schema}]`. Phản hồi trả về dưới dạng `content: [{type: "text"}, {type: "tool_use", id, name, input}]`. `input` đã được parse sẵn (một object, không phải chuỗi). Bạn trả lời bằng một message `user` mới chứa khối `{type: "tool_result", tool_use_id, content}`.

**Google Gemini API.** Bạn truyền `tools: [{functionDeclarations: [{name, description, parameters}]}]` (lồng trong `functionDeclarations`). Phản hồi đến dưới dạng `candidates[0].content.parts: [{functionCall: {name, args, id}}]` trong đó `id` là duy nhất ở Gemini 3 trở lên để tương quan lệnh gọi song song. Bạn trả lời bằng `{functionResponse: {name, id, response}}`.

Cùng một vòng lặp. Khác tên trường, khác cách lồng, khác quy ước chuỗi-so-với-object, khác cơ chế tương quan. Một nhóm viết agent thời tiết trên OpenAI phải trả hai ngày để chuyển sang Anthropic và thêm một ngày cho Gemini chỉ để lo phần nối dây.

Bài này xây một bộ chuyển đổi hợp nhất ba định dạng thành một khai báo công cụ chuẩn tắc và định tuyến ở phần rìa. Giai đoạn 13 · 17 tổng quát hoá cùng mẫu này thành một cổng LLM.

## Khái niệm

### Cấu trúc chung

Mọi nhà cung cấp đều cần năm thứ:

1. **Danh sách công cụ.** Tên, mô tả và input schema (lược đồ công cụ) cho từng công cụ.
2. **Tool choice (lựa chọn công cụ).** Ép một công cụ cụ thể, cấm dùng công cụ, hoặc để model tự quyết.
3. **Phát lệnh gọi.** Đầu ra có cấu trúc nêu tên công cụ và arguments (đối số).
4. **Id lệnh gọi.** Tương quan phản hồi với đúng lệnh gọi (quan trọng khi chạy song song).
5. **Chèn kết quả.** Một message hoặc khối gắn kết quả trở lại lệnh gọi.

### Khác biệt hình dạng, từng trường

| Khía cạnh | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| Định dạng khai báo | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Trường schema | `parameters` | `input_schema` | `parameters` |
| Nơi chứa phản hồi | `tool_calls[]` trên message assistant | `content[]` kiểu `tool_use` | `parts[]` kiểu `functionCall` |
| Kiểu đối số | JSON dạng chuỗi | object đã parse | object đã parse |
| Định dạng id | `call_...` (OpenAI sinh) | `toolu_...` (Anthropic) | UUID (Gemini 3+) |
| Khối kết quả | role `tool`, `tool_call_id` | `user` với `tool_result`, `tool_use_id` | `functionResponse` với `id` khớp |
| Ép một công cụ | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| Cấm dùng công cụ | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| Schema nghiêm ngặt | `strict: true` | schema là schema (luôn bị áp đặt) | `responseSchema` ở mức yêu cầu |

### Những giới hạn bạn sẽ thực sự gặp

- **OpenAI.** 128 công cụ mỗi yêu cầu. Độ sâu schema 5. Chuỗi đối số <= 8192 bytes. Strict mode yêu cầu không có `$ref`, không `oneOf`/`anyOf`/`allOf` chồng lấn, mọi thuộc tính phải có trong `required`.
- **Anthropic.** 64 công cụ mỗi yêu cầu. Độ sâu schema thực tế không giới hạn nhưng giới hạn thực dụng là 10. Không có cờ strict mode; schema là một hợp đồng và model thường tuân theo.
- **Gemini.** 64 hàm mỗi yêu cầu. Các kiểu schema là tập con của OpenAPI 3.0 (hơi lệch so với JSON Schema 2020-12). Lệnh gọi song song có id duy nhất từ Gemini 3.

### Hành vi của `tool_choice`

Ba chế độ mà mọi bên đều hỗ trợ, tên gọi khác nhau.

- **Auto.** Model chọn công cụ hoặc văn bản. Mặc định.
- **Required / Any.** Model buộc phải gọi ít nhất một công cụ.
- **None.** Model không được gọi công cụ.

Thêm một chế độ riêng cho từng nhà cung cấp:

- **OpenAI.** Ép một công cụ cụ thể theo tên.
- **Anthropic.** Ép một công cụ cụ thể theo tên; cờ `disable_parallel_tool_use` tách chế độ một lệnh và nhiều lệnh.
- **Gemini.** `mode: "VALIDATED"` đưa mọi phản hồi qua một bộ kiểm tra schema bất kể ý định của model.

### Lệnh gọi song song

OpenAI với `parallel_tool_calls: true` (mặc định) phát nhiều lệnh gọi trong một message assistant. Bạn chạy tất cả rồi trả lời bằng một message role tool gộp, chứa một mục cho mỗi `tool_call_id`. Anthropic trước đây chỉ làm một lệnh; `disable_parallel_tool_use: false` (mặc định từ Claude 3.5) bật nhiều lệnh. Gemini 2 cho phép gọi song song nhưng không cho id ổn định; Gemini 3 thêm UUID để phản hồi về không đúng thứ tự vẫn tương quan sạch.

### Streaming

Cả ba đều hỗ trợ lệnh gọi công cụ dạng streaming. Định dạng trên đường truyền khác nhau:

- **OpenAI.** Các chunk delta của `tool_calls[i].function.arguments` đến dần. Bạn tích luỹ cho đến `finish_reason: "tool_calls"`.
- **Anthropic.** Sự kiện block-start / block-delta / block-stop. Các chunk `input_json_delta` mang đối số từng phần.
- **Gemini.** `streamFunctionCallArguments` (mới ở Gemini 3) phát chunk kèm `functionCallId` để nhiều lệnh gọi song song có thể xen kẽ.

Giai đoạn 13 · 03 đi sâu vào phần ghép lại song song + streaming. Bài này tập trung vào hình dạng khai báo và lệnh gọi đơn.

### Lỗi và sửa lỗi

Lỗi đối số không hợp lệ cũng trông khác nhau.

- **OpenAI (không strict).** Model trả `arguments: "{bad json}"`, parse JSON của bạn thất bại, bạn chèn một message lỗi rồi gọi lại.
- **OpenAI (strict).** Việc kiểm tra diễn ra trong lúc giải mã; JSON không hợp lệ là không thể nhưng `refusal` có thể xuất hiện.
- **Anthropic.** `input` có thể chứa trường không mong đợi; schema chỉ mang tính khuyến nghị. Hãy kiểm tra ở phía server.
- **Gemini.** Điểm lạ của OpenAPI 3.0: `enum` trên trường object bị bỏ qua một cách âm thầm; tự kiểm tra lấy.

### Mẫu bộ chuyển đổi

Một khai báo công cụ chuẩn tắc trong code của bạn trông như thế này (bạn chọn hình dạng):

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

Ba hàm nhỏ chuyển nó sang ba hình dạng nhà cung cấp. Bộ khung trong `code/main.py` làm đúng việc đó, rồi thực hiện round trip (lượt khứ hồi) một lệnh gọi công cụ giả qua hình dạng phản hồi của từng nhà cung cấp. Không cần mạng — bài này dạy hình dạng, không dạy HTTP.

Các nhóm production bọc bộ chuyển đổi này trong `AbstractToolset` (Pydantic AI), `UniversalToolNode` (LangGraph), hoặc `BaseTool` (LlamaIndex). Giai đoạn 13 · 17 phát hành một cổng expose API dạng OpenAI trước mặt bất kỳ ai trong ba bên.

```figure
function-call-args
```

## Áp dụng

`code/main.py` định nghĩa một dataclass `Tool` chuẩn tắc và ba bộ chuyển đổi sinh JSON khai báo cho OpenAI, Anthropic và Gemini. Sau đó nó parse một phản hồi nhà cung cấp được soạn tay của từng hình dạng vào cùng một object lệnh gọi chuẩn tắc, chứng minh rằng ngữ nghĩa là giống nhau bên dưới. Chạy thử và diff ba khai báo cạnh nhau.

Những điểm cần xem:

- Ba khối khai báo chỉ khác nhau ở phần định dạng bao và tên trường.
- Ba khối phản hồi khác nhau ở chỗ lệnh gọi nằm (top-level `tool_calls`, khối `content[]`, mục `parts[]`).
- Một hàm `canonical_call()` trích `{id, name, args}` từ cả ba hình dạng phản hồi.

## Đóng gói

Bài này tạo ra `outputs/skill-provider-portability-audit.md`. Cho một tích hợp gọi hàm với một nhà cung cấp, skill tạo ra một bản audit khả năng chuyển đổi: tích hợp đó dựa vào giới hạn nào của nhà cung cấp, trường nào cần đổi tên, và cái gì vỡ khi chuyển sang từng nhà cung cấp khác.

## Bài tập

1. Chạy `code/main.py` và xác nhận rằng ba JSON khai báo nhà cung cấp đều serialize cùng một object `Tool` gốc. Sửa công cụ chuẩn tắc để thêm một tham số enum và xác nhận chỉ bộ chuyển đổi Gemini cần xử lý điểm lạ OpenAPI.

2. Thêm một parser `ListToolsResponse` cho từng nhà cung cấp, trích danh sách công cụ mà model trả về sau một lệnh `list_tools` hoặc lệnh khám phá. OpenAI không có sẵn cái này; hãy ghi chú sự bất đối xứng đó.

3. Cài đặt chuyển đổi `tool_choice`: ánh xạ một `ToolChoice(mode="force", tool_name="x")` chuẩn tắc sang cả ba hình dạng nhà cung cấp. Sau đó ánh xạ `mode="any"` và `mode="none"`. Đối chiếu với bảng khác biệt của bài.

4. Chọn một trong ba nhà cung cấp và đọc hướng dẫn gọi hàm của họ từ đầu đến cuối. Tìm một trường trong spec schema của họ mà hai bên kia không hỗ trợ. Gợi ý: `strict` của OpenAI, `disable_parallel_tool_use` của Anthropic, `function_calling_config.allowed_function_names` của Gemini.

5. Viết một test vector: một lệnh gọi công cụ có đối số vi phạm schema đã khai báo. Chạy nó qua bộ kiểm tra của từng nhà cung cấp (bộ kiểm tra stdlib ở Bài 01 dùng làm proxy cũng được) và ghi lại lỗi nào bật. Ghi chú bạn sẽ dùng nhà cung cấp nào trong production nếu cần độ nghiêm ngặt.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Function calling | "Sử dụng công cụ" | API ở cấp nhà cung cấp để phát lệnh gọi công cụ có cấu trúc |
| Tool declaration | "Spec công cụ" | Tên + mô tả + payload đầu vào JSON Schema |
| `tool_choice` | "Ép / cấm" | Chế độ auto / required / none / theo tên cụ thể |
| Strict mode | "Áp đặt schema" | Cờ của OpenAI ràng buộc việc giải mã khớp schema |
| `tool_use` block | "Hình dạng lệnh gọi của Anthropic" | Khối nội dung nội tuyến chứa id, name, input |
| `functionCall` part | "Hình dạng lệnh gọi của Gemini" | Một mục `parts[]` chứa name, args và id |
| Arguments-as-string | "JSON dạng chuỗi" | OpenAI trả args dưới dạng chuỗi JSON, không phải object |
| Parallel tool calls | "Toả ra trong một lượt" | Nhiều lệnh gọi công cụ trong một message assistant |
| Refusal | "Model từ chối" | Khối từ chối chỉ có ở strict mode thay cho một lệnh gọi |
| OpenAPI 3.0 subset | "Điểm lạ schema của Gemini" | Gemini dùng một phương ngữ giống JSON Schema với vài khác biệt nhỏ |

## Đọc thêm

- [OpenAI — Hướng dẫn gọi hàm](https://platform.openai.com/docs/guides/function-calling) — tài liệu tham chiếu chuẩn, gồm strict mode và lệnh gọi song song
- [Anthropic — Tổng quan sử dụng công cụ](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — ngữ nghĩa của khối `tool_use` và `tool_result`
- [Google — Gọi hàm với Gemini](https://ai.google.dev/gemini-api/docs/function-calling) — lệnh gọi song song, id duy nhất và tập con OpenAPI
- [Vertex AI — Tham chiếu gọi hàm](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — bề mặt doanh nghiệp của Gemini
- [OpenAI — Đầu ra có cấu trúc](https://platform.openai.com/docs/guides/structured-outputs) — chi tiết áp đặt schema ở strict mode

---

[← Bài trước: Giao diện công cụ](../../01-the-tool-interface/docs/vi.md) · [Bài sau: Gọi công cụ song song và truyền trực tiếp với công cụ →](../../03-parallel-and-streaming-tool-calls/docs/vi.md)
