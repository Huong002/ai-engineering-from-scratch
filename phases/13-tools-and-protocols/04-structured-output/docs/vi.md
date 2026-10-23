# Đầu ra có cấu trúc — JSON Schema, Pydantic, Zod, giải mã ràng buộc

> "Bảo model trả về JSON một cách tử tế" thất bại 5 đến 15 phần trăm số lần, ngay cả trên các model tiên tiến nhất. Đầu ra có cấu trúc (structured output) thu hẹp khoảng cách đó bằng constrained decoding (giải mã ràng buộc): model bị chặn theo nghĩa đen, không được phát ra token nào vi phạm schema (lược đồ). Chế độ strict của OpenAI, tool use có kiểu schema của Anthropic, `responseSchema` của Gemini, `output_type` của Pydantic AI và `.parse` của Zod là năm dạng bề mặt của cùng một ý tưởng. Bài này xây dựng bộ kiểm tra schema và khế ước strict-mode mà học viên sẽ dùng cho mọi pipeline trích xuất trong production.

**Type:** Build
**Languages:** Python (stdlib, JSON Schema 2020-12 subset)
**Prerequisites:** Giai đoạn 13, Bài 02 (đi sâu vào function calling)
**Time:** ~75 phút

## Mục tiêu học tập

- Viết một JSON Schema 2020-12 cho một mục tiêu trích xuất với đúng các ràng buộc (enum, min/max, required, pattern).
- Giải thích vì sao chế độ strict và giải mã ràng buộc cho những bảo đảm khác với "kiểm tra sau khi sinh".
- Phân biệt ba kiểu thất bại: lỗi parse, vi phạm schema, model khước từ.
- Đóng gói một pipeline trích xuất có xử lý sửa lỗi và xử lý refusal đều có kiểu.

## Vấn đề

Một agent đọc email đơn đặt hàng cần biến văn bản tự do thành `{customer, line_items, total_usd}`. Có ba cách.

**Cách một: prompt để model trả JSON.** "Hãy trả lời bằng JSON với các trường customer, line_items, total_usd." Cách này hiệu quả 85 đến 95 phần trăm số lần trên các model tiên tiến. Nó thất bại theo sáu kiểu: thiếu dấu ngoặc nhọn, dấu phẩy thừa ở cuối, sai kiểu dữ liệu, bịa thêm trường, bị cắt do giới hạn token, rò rỉ văn xuôi như "Here is your JSON:".

**Cách hai: kiểm tra hợp lệ sau khi sinh.** Sinh tự do, parse, so khớp với schema, rồi retry (thử lại) khi thất bại. Đáng tin nhưng tốn kém — bạn trả tiền cho mỗi lần thử lại, và lỗi cắt cụt tốn thêm một lượt cho mỗi lần xảy ra.

**Cách ba: giải mã ràng buộc.** Nhà cung cấp áp đặt schema ngay tại thời điểm giải mã. Các token không hợp lệ bị che khỏi phân phối lấy mẫu. Đầu ra được bảo đảm parse được và bảo đảm hợp lệ. Thất bại thu về một kiểu duy nhất: refusal (từ chối), tức model quyết định đầu vào không khớp schema.

Mọi nhà cung cấp tiên tiến năm 2026 đều cung cấp một dạng nào đó của cách ba.

- **OpenAI.** `response_format: {type: "json_schema", strict: true}` kèm `refusal` trong phản hồi nếu model khước từ.
- **Anthropic.** Áp đặt schema trên đầu vào `tool_use`; `stop_reason: "refusal"` không tồn tại, nhưng `end_turn` mà không có lệnh gọi tool chính là tín hiệu.
- **Gemini.** `responseSchema` ở mức request; năm 2026 Gemini cung cấp ràng buộc ngữ pháp ở mức token cho một số kiểu dữ liệu được chọn.
- **Pydantic AI.** `output_type=InvoiceModel` phát ra một `RunResult` có cấu trúc, mang kiểu `InvoiceModel`.
- **Zod (TypeScript).** Bộ parse lúc chạy, kiểm tra đầu ra của nhà cung cấp dựa trên schema Zod; đi kèm `beta.chat.completions.parse` của OpenAI.

Sợi chỉ chung: khai báo schema một lần, áp đặt nó từ đầu đến cuối.

## Khái niệm

### JSON Schema 2020-12 — ngôn ngữ chung

Mọi nhà cung cấp đều nhận JSON Schema 2020-12. Những cấu trúc bạn dùng nhiều nhất:

- `type`: một trong `object`, `array`, `string`, `number`, `integer`, `boolean`, `null`.
- `properties`: ánh xạ tên trường sang subschema.
- `required`: danh sách tên trường bắt buộc phải xuất hiện.
- `enum`: tập giá trị đóng được phép.
- `minimum` / `maximum` (số), `minLength` / `maxLength` / `pattern` (chuỗi).
- `items`: subschema áp cho từng phần tử mảng.
- `additionalProperties`: `false` cấm các trường thêm (giá trị mặc định tuỳ chế độ).

Chế độ strict của OpenAI thêm ba yêu cầu: mọi thuộc tính phải nằm trong `required`, `additionalProperties: false` ở mọi nơi, và không có `$ref` chưa phân giải. Nếu bạn vi phạm những điều này, API trả về 400 ngay lúc request.

### Pydantic, bản gắn kết cho Python

Pydantic v2 sinh JSON Schema từ các model dạng dataclass qua `model_json_schema()`. Pydantic AI bọc lại phần này để bạn viết:

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

và framework agent sẽ dịch schema sang chế độ strict của OpenAI, `input_schema` của Anthropic, hoặc `responseSchema` của Gemini ngay ở biên. Đầu ra của model trả về dưới dạng một thực thể `Invoice` có kiểu. Lỗi kiểm tra hợp lệ (validation) ném `ValidationError` kèm đường dẫn lỗi có kiểu.

### Zod, bản gắn kết cho TypeScript

Zod (`z.object({customer: z.string(), ...})`) là bản tương đương cho TS. SDK Node của OpenAI cung cấp `zodResponseFormat(Invoice)`, hàm này dịch sang payload JSON Schema của API.

### Refusal

Chế độ strict không thể ép model trả lời. Nếu đầu vào không khớp được schema ("email đó là một bài thơ, không phải hoá đơn"), model phát ra trường `refusal` chứa lý do. Code của bạn phải xử lý đây như một kết quả hạng nhất, không phải một thất bại. Refusal cũng hữu ích như một tín hiệu an toàn: một model được yêu cầu trích xuất số thẻ tín dụng từ email thuộc nội dung được bảo vệ sẽ trả về refusal kèm lý do an toàn.

### Giải mã ràng buộc trong thế giới mở

Các bản triển khai open-weights dùng ba kỹ thuật.

1. **Giải mã dựa trên ngữ pháp (grammar-constrained decoding)** (`outlines`, `guidance`, `lm-format-enforcer`): dựng một automaton hữu hạn tất định từ schema; ở mỗi bước, che logits của các token vi phạm FSM.
2. **Che logits bằng bộ parse JSON**: chạy một bộ parse JSON dạng streaming song hành với model; ở mỗi bước, tính tập token hợp lệ tiếp theo.
3. **Giải mã suy đoán kèm bộ kiểm chứng**: model nháp rẻ tiền đề xuất token, bộ kiểm chứng áp đặt schema.

Các nhà cung cấp thương mại chọn một trong ba kỹ thuật này ở hậu trường. Trạng thái nghệ thuật năm 2026 nhanh hơn sinh thường đối với đầu ra có cấu trúc ngắn và xấp xỉ cùng tốc độ với đầu ra dài.

### Ba kiểu thất bại

1. **Lỗi parse.** Đầu ra không phải JSON hợp lệ. Không thể xảy ra dưới chế độ strict. Vẫn có thể xảy ra với nhà cung cấp không strict.
2. **Vi phạm schema.** Đầu ra parse được nhưng vi phạm schema. Không thể xảy ra dưới chế độ strict. Phổ biến khi không dùng strict.
3. **Refusal.** Model khước từ. Phải được xử lý như một kết quả có kiểu.

### Chiến lược thử lại

Khi bạn ở ngoài chế độ strict (tool use của Anthropic, OpenAI không strict, Gemini đời cũ), mẫu khôi phục là:

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

Một lần thử lại thường là đủ. Ba lần bắt được các trục trặc của model yếu. Vượt quá ba lần là dấu hiệu của một schema tồi: model không thể thoả mãn nó với một số đầu vào, và prompt hoặc schema cần được sửa.

### Hỗ trợ model nhỏ

Giải mã ràng buộc vẫn hoạt động trên các model nhỏ. Một model mở 3B tham số có ràng buộc ngữ pháp vượt trội một model 70B tham số chỉ prompt thô trên các tác vụ có cấu trúc. Đây là lý do chính khiến đầu ra có cấu trúc quan trọng trong production: nó tách độ tin cậy khỏi kích thước model.

```figure
constrained-decoding
```

## Áp dụng

`code/main.py` cung cấp một bộ kiểm tra JSON Schema 2020-12 tối giản bằng stdlib (types, required, enum, min/max, pattern, items, additionalProperties). Nó bọc một schema `Invoice` và chạy đầu ra LLM giả qua bộ kiểm tra, minh hoạ các nhánh lỗi parse, vi phạm schema và refusal. Khi lên production, hãy thay đầu ra giả bằng phản hồi thật của bất kỳ nhà cung cấp nào.

Những điểm cần xem:

- Bộ kiểm tra trả về danh sách `[ValidationError]` có kiểu, kèm đường dẫn và thông điệp. Đó chính là dạng bạn muốn đưa vào prompt thử lại.
- Nhánh refusal KHÔNG thử lại. Nó ghi log và trả về refusal có kiểu. Giai đoạn 14 · 09 dùng refusal như một tín hiệu an toàn.
- Kiểm tra `additionalProperties: false` kích hoạt trên đầu vào kiểm thử đối kháng, cho thấy vì sao chế độ strict đóng cửa với các trường bịa ra.

## Đóng gói

Bài này tạo ra `outputs/skill-structured-output-designer.md`. Cho một mục tiêu trích xuất văn bản tự do (hoá đơn, ticket hỗ trợ, CV, v.v.), skill tạo một JSON Schema 2020-12 tương thích strict-mode và một model Pydantic phản chiếu nó, kèm xử lý refusal và thử lại có kiểu được dựng sẵn.

## Bài tập

1. Chạy `code/main.py`. Thêm một ca kiểm thử thứ tư với `total_usd` là số âm. Xác nhận bộ kiểm tra từ chối nó qua đường dẫn ràng buộc `minimum`.

2. Mở rộng bộ kiểm tra để hỗ trợ `oneOf` kèm discriminator. Ca phổ biến: `line_item` hoặc là một sản phẩm hoặc là một dịch vụ, được gắn nhãn bằng `kind`. Chế độ strict có các quy tắc tinh tế ở đây; hãy đọc hướng dẫn structured outputs của OpenAI.

3. Viết cùng schema Invoice đó dưới dạng một Pydantic BaseModel và so sánh đầu ra `model_json_schema()` với schema bạn tự viết. Xác định trường duy nhất mà Pydantic đặt mặc định còn bản tự viết bỏ qua.

4. Đo tỉ lệ refusal. Tạo mười đầu vào không thể trích xuất (lời một bài hát, một chứng minh toán học, một email trống) và chạy chúng qua một nhà cung cấp thật ở chế độ strict. Đếm số refusal so với số đầu ra bịa. Đây là chuẩn tham chiếu của bạn cho việc thử lại có nhận biết refusal.

5. Đọc hướng dẫn structured outputs của OpenAI từ đầu đến cuối. Xác định cấu trúc duy nhất mà nó cấm rõ ràng ở chế độ strict trong khi JSON Schema thường vẫn cho phép. Sau đó thiết kế một schema dùng cấu trúc bị cấm đó một cách không thiết yếu rồi tái cấu trúc nó cho tương thích strict.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| JSON Schema 2020-12 | "Bản đặc tả schema" | Phương ngữ schema dạng IETF-draft mà mọi nhà cung cấp hiện đại đều nói |
| Strict mode | "Schema được bảo đảm" | Cờ của OpenAI áp đặt schema qua giải mã ràng buộc |
| Constrained decoding | "Che logits" | Áp đặt lúc giải mã, che các token tiếp theo không hợp lệ |
| Refusal | "Model khước từ" | Kết quả có kiểu khi đầu vào không khớp được schema |
| Parse error | "JSON không hợp lệ" | Đầu ra không parse được thành JSON; bất khả thi dưới strict |
| Schema violation | "Sai hình dạng" | Parse được nhưng vi phạm types / required / enum / range |
| `additionalProperties: false` | "Không cho phép thêm" | Cấm các trường không xác định; bắt buộc trong strict của OpenAI |
| Pydantic BaseModel | "Đầu ra có kiểu" | Lớp Python phát ra và kiểm tra JSON Schema |
| Zod schema | "Kiểu đầu ra TypeScript" | Schema lúc chạy của TS để kiểm tra đầu ra nhà cung cấp |
| Grammar enforcement | "Giải mã ràng buộc open-weights" | Che logits dựa trên FSM, như trong outlines / guidance |

## Đọc thêm

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — chế độ strict, refusal và các yêu cầu schema
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — bài đăng ra mắt tháng 8 năm 2024 giải thích bảo đảm giải mã
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — các ràng buộc output_type có kiểu, tuần tự hoá theo từng nhà cung cấp
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — bản đặc tả chuẩn
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — ghi chú triển khai doanh nghiệp và các lưu ý strict-mode

---

[← Bài trước: Gọi công cụ song song và truyền trực tiếp với công cụ](../../03-parallel-and-streaming-tool-calls/docs/vi.md) · [Bài sau: Thiết kế lược đồ công cụ →](../../05-tool-schema-design/docs/vi.md)
