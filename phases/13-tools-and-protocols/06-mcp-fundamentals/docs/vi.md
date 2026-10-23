# Kiến thức nền tảng MCP: Yêu cầu không trạng thái và JSON-RPC

> MCP hiện đại không có bắt tay và không có phiên giao thức. Mỗi yêu cầu phải mang đủ metadata để tự nó được hiểu, được cấp quyền, được định tuyến và được thử lại.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 13, Bài 01-05
**Time:** ~55 phút

## Mục tiêu học tập

- Phân biệt các primitive phía server (máy chủ) của MCP (Model Context Protocol) với các tính năng phía client (máy khách).
- Xây dựng yêu cầu và phản hồi JSON-RPC 2.0 hợp lệ cho MCP `2026-07-28`.
- Đính kèm phiên bản giao thức, capability (năng lực) của client và danh tính client vào mỗi yêu cầu.
- Dùng `server/discover` và xử lý `UnsupportedProtocolVersionError` mà không cần bắt tay.
- Truy vết một yêu cầu độc lập từ bước kiểm tra hợp lệ đến khi có kết quả hoàn chỉnh.

## Vấn đề

Một server MCP có thể nhận hai yêu cầu liên tiếp từ các client khác nhau, với các capability khác nhau, trên cùng một process hoặc cùng một HTTP worker. Nếu server ghi nhớ những gì yêu cầu trước đã khai báo, nó có thể áp dụng sai quyền hoặc trả về sai dạng dữ liệu trên đường truyền.

MCP `2026-07-28` xoá bỏ sự mơ hồ đó. Lõi giao thức là không trạng thái. Server phải quyết định cách xử lý yêu cầu hiện tại dựa trên chính yêu cầu hiện tại, không phải từ lịch sử kết nối.

Điều này thay đổi mô hình tư duy. Trình tự cũ là kết nối trước, bắt tay sau, thao tác cuối. Trình tự hiện đại đơn giản hơn:

1. Client gửi một yêu cầu tự mô tả.
2. Server kiểm tra phiên bản và capability của yêu cầu đó.
3. Server xử lý phương thức.
4. Server trả về một kết quả có kiểu hoặc một lỗi JSON-RPC.

Yêu cầu tiếp theo lặp lại đúng quy trình đó từ đầu.

## Khái niệm

### Primitive phía server

Server MCP cung cấp ba primitive chính:

1. **Tools** (công cụ) là các hành động do model điều khiển, được khám phá bằng `tools/list` và gọi bằng `tools/call`.
2. **Resources** (tài nguyên) là dữ liệu định địa chỉ bằng URI, được khám phá bằng `resources/list` và lấy về bằng `resources/read`.
3. **Prompts** là các mẫu prompt (prompt template) tái sử dụng, được khám phá bằng `prompts/list` và kết xuất bằng `prompts/get`.

Roots, sampling và logging vẫn còn trong schema `2026-07-28` để đảm bảo tương thích, nhưng chúng đã lỗi thời. Các triển khai mới nên dùng đầu vào công cụ hoặc tài nguyên tường minh cho roots, dùng API trực tiếp của nhà cung cấp model cho sampling, và dùng stderr hoặc OpenTelemetry cho logging. Elicitation vẫn khả dụng qua Multi Round-Trip Requests, trong đó server trả về một yêu cầu nhập liệu và client thử lại thao tác ban đầu. Một server hiện đại không bao giờ tự khởi tạo một yêu cầu JSON-RPC độc lập.

### Phong bì JSON-RPC

MCP dùng JSON-RPC 2.0:

- Request: `{jsonrpc, id, method, params}`
- Response: `{jsonrpc, id, result}` hoặc `{jsonrpc, id, error}`
- Notification: `{jsonrpc, method, params}` không có `id`

`id` của yêu cầu tương ứng với đúng một phản hồi. Nó không tạo ra phiên giao thức.

### Metadata bắt buộc của yêu cầu

Mọi yêu cầu hiện đại đều mang một object `_meta` bên trong `params`:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Phiên bản giao thức và capability của client là bắt buộc. Danh tính client là khuyến nghị. Đó là dữ liệu tự khai để hiển thị và gỡ lỗi, không phải thông tin xác thực bảo mật.

Server không được suy ra bất kỳ giá trị nào trong số này từ một yêu cầu trước đó, từ một process stdio, từ một kết nối HTTP, hay chỉ từ header của transport (kênh truyền).

### Kết quả hoàn chỉnh và danh tính server

Mọi kết quả hiện đại thành công đều có `resultType`. Một kết quả cuối cùng bình thường dùng `"complete"`. Server cũng nên tự nhận diện mình trong metadata của kết quả:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "resultType": "complete",
    "tools": [],
    "ttlMs": 30000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "notes-server",
        "version": "1.0.0"
      }
    }
  }
}
```

`tools/list`, `resources/list`, `prompts/list`, `resources/templates/list`, `resources/read` và `server/discover` là những kết quả có thể cache. Chúng bao gồm `ttlMs` và `cacheScope`. Giá trị mặc định an toàn là `ttlMs: 0` và `cacheScope: "private"`. Các item trong danh sách nên có thứ tự xác định để những phản hồi tương đương tạo ra cache key ổn định và ngữ cảnh model ổn định.

### Khám phá mà không cần bắt tay

Mọi server hiện đại phải triển khai `server/discover`. Client có thể gọi nó trước một phương thức khác để lấy:

- `supportedVersions`
- `capabilities` của server
- `instructions` hướng dẫn sử dụng (không bắt buộc)
- danh tính server trong `_meta` của kết quả
- các gợi ý cache

Khám phá thì hữu ích, nhưng nó không phải cổng chặn. Client có thể gửi `tools/list` trước vì yêu cầu đó đã mang theo phiên bản giao thức và capability của nó.

Nếu phiên bản được yêu cầu không được hỗ trợ, server trả về mã JSON-RPC `-32022` kèm:

```json
{
  "requested": "2027-01-01",
  "supported": ["2026-07-28"]
}
```

Client chọn một phiên bản hiện đại mà cả hai bên cùng hỗ trợ và thử lại với id yêu cầu JSON-RPC mới.

### Vòng đời của một yêu cầu

Truy vết một yêu cầu hiện đại theo thứ tự này:

1. Phân tích một phong bì JSON-RPC.
2. Xác nhận `jsonrpc` là `"2.0"`, có `id`, `method` là string và `params` là object.
3. Yêu cầu chuỗi phiên bản và object capability trong `params._meta`; metadata sai định dạng hoặc thiếu là `-32602`.
4. Tại biên HTTP, so sánh phiên bản, phương thức và các header tên áp dụng với phần body. Sự khác nhau là `-32020`, kể cả khi một trong hai giá trị phiên bản không được hỗ trợ.
5. Sau khi đã xác lập bằng nhau, từ chối phiên bản khớp nhưng không được hỗ trợ bằng `-32022`.
6. Kiểm tra các capability cần thiết, rồi định tuyến theo `method` và kiểm tra các đối số riêng của phương thức.
7. Xác thực và cấp quyền cho thao tác cụ thể trước khi handler của nó chạy.
8. Trả về một kết quả hoàn chỉnh kèm danh tính server.
9. Quên metadata giao thức gắn với yêu cầu đó.

Thứ tự đó ngăn hai thành phần diễn giải hai lệnh gọi khác nhau. Một gateway không được cấp quyền cho `Mcp-Name: notes.read` trong khi origin thực thi `params.name: notes.delete`. Nó cũng giữ cho đầu vào sai định dạng, nhầm lẫn header, thương lượng phiên bản, lỗi capability, cấp quyền và lỗi handler thành những bằng chứng riêng biệt.

Đóng stdin hoặc một phản hồi HTTP sẽ kết thúc hoạt động của transport, chứ không kết thúc một phiên giao thức — vì MCP hiện đại không có phiên giao thức.

### Tương thích lỗi thời tường minh

Các phiên bản đến `2025-11-25` dùng `initialize`, `notifications/initialized`, capability gắn với kết nối và, trên Streamable HTTP đời trước, phiên giao thức không bắt buộc. Hành vi đó vẫn còn liên quan khi một client hai thời kỳ nói chuyện với một server cũ.

Hãy tách biệt hai thời kỳ. Một yêu cầu hiện đại được nhận diện bằng metadata bắt buộc theo từng yêu cầu. Một kết nối lỗi thời chỉ được chọn qua đường fallback đã được tài liệu hoá. Đừng gửi `initialize` như mặc định cho một server `2026-07-28`.

Vì vậy “không trạng thái” mang nghĩa khác nhau theo từng thời kỳ. Trong `2026-07-28`, đó là một bất biến của giao thức: mọi yêu cầu thông thường đều diễn giải được một cách độc lập và không tồn tại phiên MCP. Ở các phiên bản đến `2025-11-25`, bước khởi tạo và capability đã thương lượng thuộc về một kết nối, nên một adapter tương thích có thể giữ trạng thái kết nối cũ đó. Một triển khai hai thời kỳ không phải là một máy trạng thái dễ dãi duy nhất. Nó là một lõi hiện đại không trạng thái đặt cạnh một adapter lỗi thời tách biệt, với một quyết định lựa chọn tường minh trước khi bất kỳ parser nào chạy.

Cả hai nghĩa đều không cấm trạng thái ứng dụng bền vững. Một workflow, task hay bản nháp có thể nằm sau một handle không trong suốt trong kho dùng chung. Client gửi handle đó như đầu vào thông thường, và mọi replica đều xác thực và cấp quyền khi dùng nó. Ngữ cảnh giao thức không được rò rỉ vào kho đó để thay thế cho phiên đã bị loại bỏ.

```figure
mcp-tool-call
```

## Áp dụng

`code/main.py` xây dựng, kiểm tra, truy vết và điều phối các thông điệp MCP hiện đại mà không cần framework. Chạy:

```bash
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Hãy chú ý ba bất biến trong output:

- Mọi yêu cầu đều lặp lại các trường `_meta` của nó.
- Mọi kết quả thành công đều là `resultType: "complete"` và có danh tính server.
- Kết quả danh sách có thứ tự xác định và có các gợi ý cache tường minh.

## Đóng gói

Bài học này đóng gói `outputs/skill-mcp-handshake-tracer.md`. Tên file lịch sử được giữ ổn định, nhưng artifact giờ là một tracer yêu cầu không trạng thái. Nó kiểm tra từng thông điệp một cách độc lập và chỉ gắn nhãn lưu lượng bắt tay lỗi thời khi thực sự có.

## Bài tập

1. Đổi phiên bản giao thức của một yêu cầu thành `2027-01-01`. Xác nhận mã lỗi là `-32022` và phần data quảng bá phiên bản được hỗ trợ.
2. Xoá `io.modelcontextprotocol/clientCapabilities` khỏi yêu cầu thứ hai. Xác nhận server không dùng lại capability của yêu cầu thứ nhất.
3. Đảo ngược registry công cụ trong bộ nhớ. Xác nhận `tools/list` vẫn trả về cùng thứ tự xác định.
4. Đổi `cacheScope` từ `public` sang `private`. Giải thích trong mỗi trường hợp, những ngữ cảnh cấp quyền nào có thể dùng lại phản hồi.
5. Thêm một test bỏ qua `clientInfo` không bắt buộc. Yêu cầu vẫn phải hợp lệ vì danh tính client là khuyến nghị, không bắt buộc.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa thực sự |
|------|---------|
| Stateless protocol | Mọi yêu cầu đều cung cấp metadata cần thiết để diễn giải nó |
| Request metadata | Phiên bản, capability của client và danh tính client (khuyến nghị) trong `params._meta` |
| `server/discover` | Phương thức server bắt buộc cho phiên bản, capability, instructions và danh tính |
| `resultType` | Bộ phân biệt trên mọi kết quả hiện đại thành công |
| Cacheable result | Kết quả bao gồm các gợi ý bắt buộc `ttlMs` và `cacheScope` |
| Protocol era | Metadata theo từng yêu cầu (hiện đại) hoặc khởi tạo gắn với kết nối (lỗi thời) |
| Transport lifetime | Vòng đời của process, kết nối hoặc luồng phản hồi, không phải trạng thái phiên giao thức |
| `-32022` | Lỗi phiên bản giao thức không được hỗ trợ, kèm phiên bản yêu cầu và phiên bản hỗ trợ |

## Đọc thêm

- [MCP Architecture](https://modelcontextprotocol.io/specification/2026-07-28/architecture)
- [MCP Base Protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP 2026-07-28 Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)

---

[← Bài trước: Thiết kế lược đồ công cụ](../../05-tool-schema-design/docs/vi.md) · [Bài sau: Xây dựng một MCP server: Python và TypeScript không trạng thái →](../../07-building-an-mcp-server/docs/vi.md)
