# Xây dựng một MCP server: Python và TypeScript không trạng thái

> Một MCP server hiện đại không ghi nhớ bắt tay. Nó kiểm tra metadata trên từng yêu cầu, chạy một handler (hàm xử lý), rồi trả về một kết quả có kiểu.

**Type:** Build
**Languages:** Python, TypeScript
**Prerequisites:** Giai đoạn 13, Bài 06
**Time:** ~85 phút

## Mục tiêu học tập

- Triển khai `server/discover` bắt buộc cho MCP `2026-07-28`.
- Kiểm tra phiên bản giao thức và năng lực của client trên từng yêu cầu.
- Phơi ra tools, resources và prompts với thứ tự danh sách tất định.
- Trả về `resultType`, danh tính máy chủ và gợi ý cache trên đúng các kết quả.
- Phục vụ cùng một hợp đồng không trạng thái qua stdio (luồng vào/ra chuẩn) phân tách bằng dòng mới trong Python và TypeScript.

## Vấn đề

Một máy chủ lưu năng lực của client sau tin nhắn đầu tiên thì dễ xây mà khó vận hành. Cùng một tiến trình có thể phục vụ nhiều client nối tiếp nhau. Một yêu cầu từ xa có thể rơi vào một worker khác. Một khai báo năng lực đã cũ có thể làm rò rỉ hành vi qua ranh giới phân quyền.

MCP `2026-07-28` giải quyết phần giao thức của vấn đề đó bằng cách khiến mỗi yêu cầu tự mô tả chính nó. Ứng dụng của bạn vẫn có thể giữ ghi chú bền vững, job, hoặc tay nắm trạng thái tường minh. Thứ nó không được giữ là trạng thái giao thức ẩn làm thay đổi cách một yêu cầu về sau được giải mã.

Bài này xây một notes server hai lần. Bản Python và bản TypeScript chỉ dùng thư viện chuẩn cho phần lõi giao thức. Cả hai phơi ra cùng các phương thức và thực thi cùng một hợp đồng truyền tải.

## Khái niệm

### Vòng lặp điều phối hiện đại

```text
read one JSON-RPC line
parse the envelope
if it is a notification, do not respond
validate params._meta for this request
route by method
wrap success with resultType and serverInfo
write one JSON-RPC response line
forget request-scoped metadata
```

Ba quy tắc stdio vẫn quan trọng:

- Chỉ ghi thông điệp JSON-RPC vào stdout. Gửi chẩn đoán ra stderr.
- Phân tách thông điệp bằng một dòng mới và flush từng phản hồi.
- Thoát ngay khi stdin chạm EOF.

Vòng đời máy chủ (server lifecycle) chính là vòng đời của transport (kênh truyền). Nó không phải một phiên MCP hiện đại.

### Kiểm tra yêu cầu

Mỗi yêu cầu phải có:

```json
{
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "notes-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Hai trường đầu là bắt buộc. `clientInfo` là khuyến nghị. Hãy kiểm tra schema (lược đồ) của phần nhận dạng khi nó hiện diện, nhưng đừng coi đó là xác thực.

Nếu phiên bản không được hỗ trợ, trả về mã lỗi (error code) `-32022` kèm `requested` và `supported`. Thiếu metadata yêu cầu là invalid params, mã `-32602`. Không bao giờ điền các trường còn thiếu từ một lời gọi trước.

### Khám phá bắt buộc

Máy chủ hiện đại phải triển khai `server/discover`. Một kết quả khám phá đầy đủ gồm các phiên bản hiện đại được hỗ trợ, năng lực, instructions tuỳ chọn, gợi ý cache và danh tính máy chủ trong `_meta` của kết quả:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": false},
    "resources": {"listChanged": false, "subscribe": false},
    "prompts": {"listChanged": false}
  },
  "ttlMs": 3600000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

Khám phá không mở khoá máy chủ. Client có thể gọi `tools/list` mà không cần gọi khám phá vì `tools/list` đã mang cùng metadata yêu cầu.

### Công cụ

`tools/list` trả về một danh sách mô tả công cụ tất định. Thứ tự ổn định giúp cache phản hồi tốt hơn và giữ ngữ cảnh của model ổn định. Kết quả cũng yêu cầu `ttlMs` và `cacheScope`.

`tools/call` trả về các khối nội dung và `isError`. Dùng lỗi JSON-RPC khi envelope giao thức hoặc tham số phương thức không hợp lệ. Dùng `isError: true` khi một lời gọi công cụ hợp lệ chạy nhưng bản thân công cụ thất bại.

Chú thích công cụ vẫn chỉ là gợi ý, không phải cưỡng chế:

- `readOnlyHint`
- `destructiveHint`
- `idempotentHint`
- `openWorldHint`

Host nên dùng chúng để xác nhận và trình bày. Máy chủ vẫn phải thực thi phân quyền thật.

### Tài nguyên

`resources/list` trả về các mô tả URI ổn định. `resources/read` trả về nội dung có kiểu. Cả hai đều cache được trong `2026-07-28`, nên cả hai đều gồm `ttlMs` và `cacheScope`.

Dùng `cacheScope: "private"` cho dữ liệu ghi chú riêng của người dùng. Cache chia sẻ không được dùng lại một phản hồi riêng tư qua các ngữ cảnh phân quyền khác nhau.

Cách gửi thay đổi hiện đại không dùng `resources/subscribe`. Client mở `subscriptions/listen` và yêu cầu `resourceSubscriptions` hoặc các nhóm thay đổi danh sách. Bài 10 xây luồng đó.

### Prompt

`prompts/list` cache được và tất định. `prompts/get` render một prompt có tên cùng tham số. Kết quả prompt đã render là complete, nhưng nó không thuộc các kết quả list hoặc read cache được cần gợi ý cache.

### Mọi kết quả thành công đều có kiểu

Các ví dụ dùng một wrapper cho mọi thành công:

```python
def complete(payload):
    return {
        "resultType": "complete",
        **payload,
        "_meta": {SERVER_INFO_KEY: SERVER_INFO},
    }
```

Handler của list, read và discovery thêm `ttlMs` cùng `cacheScope`. Tập trung wrapper này giúp một handler không thể âm thầm bỏ sót các trường kết quả hiện đại.

### Không có yêu cầu do máy chủ khởi phát

Máy chủ hiện đại có thể gửi thông báo liên quan tới một yêu cầu của client, hoặc thông báo trên luồng `subscriptions/listen` do client mở. Nó không được gửi yêu cầu JSON-RPC của riêng nó.

Khi một handler cần sampling, elicitation hoặc roots input, nó trả về kết quả `input_required`. Client đáp ứng các yêu cầu đầu vào nhúng rồi thử lại phương thức gốc với id yêu cầu mới. Bài 11 bao quát mẫu Multi Round-Trip Request đó.

### Tương thích legacy tường minh

Một máy chủ lưỡng thời kỳ cũng có thể triển khai bắt tay `2025-11-25` trên một nhánh legacy tách bạch rõ ràng. Nó chọn hành vi hiện đại khi có đủ các trường `_meta` hiện đại bắt buộc và hành vi legacy khi nhận `initialize`.

Đừng đưa một yêu cầu `2026-07-28` qua đường bắt tay legacy. Đừng đóng dấu các trường `resultType` hiện đại lên kết quả khởi tạo legacy. Mã trong bài này cố ý chỉ hiện đại để các bất biến của nó luôn lộ ra.

```figure
t3-dispatch-loop
```

## Áp dụng

Chạy demo hữu hạn và test của máy chủ Python:

```bash
cd code
python3 main.py --demo
python3 -m unittest discover tests -v
```

Chạy bản TypeScript với một TypeScript runner:

```bash
npx tsx main.ts --demo
```

Demo gửi `server/discover`, liệt kê từng primitive, gọi các công cụ và hiển thị một lỗi phiên bản không được hỗ trợ. Mọi yêu cầu hiện đại đều lặp lại metadata. Mọi thành công đều gồm danh tính máy chủ.

## Đóng gói

Bài này đóng gói `outputs/skill-mcp-server-scaffolder.md`. Nó tạo ra một kế hoạch máy chủ hiện đại gồm hợp đồng khám phá, kiểm tra theo từng yêu cầu, danh sách cache được tất định và một adapter legacy cô lập tuỳ chọn.

## Bài tập

1. Bỏ năng lực khỏi một yêu cầu và chứng minh máy chủ không dùng lại khai báo của yêu cầu trước.
2. Đảo ngược thứ tự chèn `TOOLS`, `PROMPTS` và các ghi chú. Xác nhận mọi kết quả danh sách vẫn ổn định.
3. Thêm công cụ `notes_delete` có tính phá huỷ và buộc phải kiểm tra phân quyền bên trong executor. Giữ `destructiveHint` chỉ như gợi ý UX.
4. Thêm `resources/templates/list` với `ttlMs`, `cacheScope` và thứ tự tất định.
5. Xây một adapter legacy riêng cho `2025-11-25`. Thêm test chứng minh yêu cầu hiện đại không bao giờ đi vào đó.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa thực sự |
|------|---------|
| Stateless server | Xử lý mỗi yêu cầu từ metadata của chính nó mà không cần bộ nhớ phiên giao thức |
| `server/discover` | Phương thức hiện đại bắt buộc, công bố các phiên bản và năng lực |
| Complete result | Kết quả hiện đại thành công với `resultType: "complete"` |
| Cacheable result | Kết quả discovery, list hoặc resource-read có `ttlMs` và `cacheScope` |
| Deterministic list | Cùng một registry lô-gic luôn tạo ra cùng thứ tự mục |
| Server identity | `io.modelcontextprotocol/serverInfo` khuyến nghị trong `_meta` của kết quả |
| Tool error | Lời gọi công cụ hợp lệ trả về nội dung với `isError: true` |
| Protocol error | Yêu cầu JSON-RPC hoặc MCP không hợp lệ được trả qua `error` |

## Đọc thêm

- [Đặc tả MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [Khám phá máy chủ MCP](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [Công cụ MCP](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [Tài nguyên MCP](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [Prompt MCP](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [Transport stdio của MCP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)

---

[← Bài trước: Kiến thức nền tảng MCP: Yêu cầu không trạng thái và JSON-RPC](../../06-mcp-fundamentals/docs/vi.md) · [Bài sau: Xây dựng MCP client: khám phá, định tuyến và dự phòng hai thời kỳ giao thức →](../../08-building-an-mcp-client/docs/vi.md)
