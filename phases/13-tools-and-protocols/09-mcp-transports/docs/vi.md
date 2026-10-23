# Transport MCP: stdio và Streamable HTTP không trạng thái

> Transport (kênh truyền) mang thông điệp MCP. Nó không cấp trạng thái giao thức còn thiếu. Trong `2026-07-28`, stdio (kênh vào/ra chuẩn của tiến trình con) cục bộ và Streamable HTTP (HTTP dạng luồng) từ xa đều mang các request tự mô tả.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 13, Bài 07 và 08
**Time:** ~65 phút

## Mục tiêu học tập

- Chọn stdio cho tiến trình con cục bộ và Streamable HTTP cho dịch vụ mạng.
- Cài đặt đúng hợp đồng Streamable HTTP hiện đại: một endpoint duy nhất, chỉ POST.
- Sao chép và kiểm tra các header phiên bản, method và name của MCP so với body JSON-RPC.
- Truyền đúng SSE (luồng sự kiện do máy chủ đẩy) theo phạm vi request và các luồng `subscriptions/listen` sống lâu.
- Di trú các triển khai dựa trên session và HTTP+SSE kiểu cũ mà không trình bày hành vi cũ như thể là hiện đại.

## Vấn đề

Các bản Streamable HTTP trước đây gộp đàm phán giao thức với hành vi kết nối và phiên. Máy chủ có thể tạo `Mcp-Session-Id`, mở một luồng GET riêng, chấp nhận DELETE để kết thúc phiên, và nối lại SSE bằng `Last-Event-ID`.

MCP `2026-07-28` loại bỏ các cơ chế đó khỏi đường truyền hiện đại. Mọi request có thể rơi vào bất kỳ worker khỏe mạnh nào vì phiên bản giao thức và năng lực client của nó nằm ngay trong body. Header HTTP phản chiếu một số trường để định tuyến và áp chính sách, nhưng máy chủ kiểm tra các header đó với body trước khi thực thi.

Kết quả là dễ mở rộng quy mô hơn và dễ suy luận hơn. Nó cũng có nghĩa là một máy chủ dạy transport của năm 2025 như thể là hiện hành đang dạy sai mô hình lỗi và mô hình bảo mật.

## Khái niệm

### stdio

Ràng buộc stdio dành cho một tiến trình con do client khởi chạy:

- Client ghi một thông điệp JSON-RPC UTF-8 mỗi dòng vào stdin.
- Máy chủ ghi một thông điệp JSON-RPC UTF-8 mỗi dòng ra stdout.
- Máy chủ ghi chẩn đoán ra stderr.
- Máy chủ thoát ngay khi stdin gặp EOF.
- Mọi request hiện đại mang phiên bản và năng lực client trong `params._meta`.

Tiến trình có thể sống qua nhiều lần gọi, nhưng nó không phải một phiên giao thức hiện đại. Nếu nó thoát bất ngờ, các request đang bay sẽ bị mất. Hãy khởi động lại tiến trình, khám phá lại, liệt kê lại, mở lại các subscription, và thử lại các thao tác an toàn với id request mới.

### Streamable HTTP trong 2026-07-28

Máy chủ hiện đại mở một endpoint MCP, ví dụ `/mcp`, nhận POST.

Mỗi request hoặc notification JSON-RPC là một HTTP POST mới. Body chứa một thông điệp JSON-RPC. Client không gửi response JSON-RPC cho máy chủ.

Với một request, máy chủ trả về một trong hai:

- `Content-Type: application/json` kèm một response JSON-RPC; hoặc
- `Content-Type: text/event-stream` kèm các notification liên quan tới request đó, tiếp theo là response JSON-RPC cuối cùng.

Với một notification được chấp nhận, máy chủ trả `202 Accepted` không có body.

Client khai báo cả hai loại response:

```http
Accept: application/json, text/event-stream
```

### Chỉ POST nghĩa là chỉ POST

Streamable HTTP hiện đại không có luồng GET riêng và không có endpoint DELETE cho phiên.

- `GET /mcp` trả `405 Method Not Allowed`.
- `DELETE /mcp` trả `405 Method Not Allowed`.
- `Mcp-Session-Id` bị bỏ qua, không bao giờ được tạo hay dội lại.
- `Last-Event-ID` bị bỏ qua vì các luồng hiện đại không thể nối lại.

Nếu một luồng theo phạm vi request đứt trước response cuối cùng, client đã mất request đang bay đó. Nó có thể phát một request mới với id JSON-RPC mới khi việc thử lại là an toàn. Nó không được cố nối lại luồng.

### Kiểm tra Origin

Máy chủ kiểm tra `Origin` trên các kết nối đến để chống DNS rebinding. Nếu header có mặt và không được cho phép rõ ràng, trả `403 Forbidden`. Client không phải trình duyệt có thể bỏ `Origin`, và quy tắc transport chính thức cho phép điều đó.

Máy chủ cục bộ nên bind vào `127.0.0.1`, không phải mọi interface. Dịch vụ mạng vẫn cần xác thực và cấp quyền trên từng request. Kiểm tra Origin không phải xác thực.

Hãy so khớp origin chính xác sau khi chuẩn hoá cấu hình. Các kiểm tra theo tiền tố như `origin.startswith("https://trusted.example")` không an toàn vì chúng có thể chấp nhận hậu tố do kẻ tấn công kiểm soát.

### Các header metadata HTTP bắt buộc

Mọi request POST hiện đại đều chứa:

```http
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes_search
```

Quy tắc header:

- `MCP-Protocol-Version` là bắt buộc và phải bằng `params._meta.io.modelcontextprotocol/protocolVersion`.
- `Mcp-Method` là bắt buộc và phải bằng `method` của JSON-RPC.
- `Mcp-Name` là bắt buộc với `tools/call`, `resources/read`, và `prompts/get`.
- `Mcp-Name` bằng `params.name`, hoặc `params.uri` với `resources/read`.
- Giá trị header phân biệt chữ hoa chữ thường dù tên header thì không.

Các giá trị `Mcp-Name` không an toàn hoặc không phải ASCII dùng đúng sentinel Base64 UTF-8:

```text
=?base64?{Base64EncodedValue}?=
```

Máy chủ giải mã giá trị đó trước khi so sánh với body.

Header phản chiếu bị thiếu, sai định dạng hoặc lệch sẽ trả HTTP `400` với mã JSON-RPC `-32020`. Nếu header và body khớp nhau về một phiên bản mà máy chủ không hỗ trợ, trả HTTP `400` với `-32022` và data lỗi chính xác như `{"supported":["2026-07-28"],"requested":"2027-01-01"}`.

Một method hiện đại không xác định trả HTTP `404` với JSON-RPC `-32601`. Body JSON-RPC là quan trọng vì client lưỡng thời dùng nó để phân biệt lỗi hiện đại với việc trượt endpoint kiểu cũ.

### SSE theo phạm vi request

Máy chủ có thể chọn SSE cho một request chạy lâu:

```text
POST tools/call id=41
  <- notifications/progress related to id=41
  <- notifications/progress related to id=41
  <- JSON-RPC response id=41
stream closes
```

Máy chủ không được gửi các request JSON-RPC độc lập trên luồng này. Tương tác sampling, elicitation và roots dùng kết quả Multi Round-Trip Request. Đóng luồng response sẽ huỷ request đó.

Đừng thêm event id SSE để phát lại. Nối lại bằng `Last-Event-ID` không thuộc bản hiện đại.

### Thay đổi sống lâu dùng subscriptions/listen

Notification thay đổi dùng một request do client mở, không phải GET riêng:

```json
{
  "jsonrpc": "2.0",
  "id": "listen-1",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true,
      "resourceSubscriptions": ["notes://note-1"]
    },
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

Response POST là một luồng SSE sống lâu. Thông điệp giao thức đầu tiên của nó là `notifications/subscriptions/acknowledged`. Thông điệp xác nhận, mọi notification thay đổi, và kết quả cuối cùng đều mang `io.modelcontextprotocol/subscriptionId` trong `_meta`, bằng id của request listen. Máy chủ có thể phát comment SSE làm keepalive. Khi luồng đứt, client phát lại `subscriptions/listen` với id request mới và lấy lại dữ liệu bị ảnh hưởng.

`resources/subscribe` và `resources/unsubscribe` thuộc về thời kỳ cũ. Đừng dùng chúng trên một kết nối hiện đại.

### Trạng thái ứng dụng tường minh

Việc bỏ session giao thức không cấm các luồng công việc có trạng thái. Máy chủ có thể tạo một handle trạng thái mờ và trả nó như kết quả tool bình thường. Client truyền handle đó như một đối số tường minh ở các lần gọi sau.

Hãy gắn handle với principal đã xác thực, làm chúng không thể đoán, đặt hạn dùng, và cấp quyền cho mọi lần dùng. Điều này làm trạng thái hiện ra ở tầng ứng dụng thay vì ẩn nó trong ràng buộc kết nối của transport.

Lỗi do trạng thái bản sao bị ẩn gây ra mang tính cơ học:

1. Request A tới replica 1 và tạo một bản nháp trong bộ nhớ của tiến trình đó.
2. Response không trả về handle bản nháp vì bản cài đặt cho rằng kết nối đã định danh bản nháp.
3. Request B là một POST mới và tới replica 2.
4. Replica 2 có metadata giao thức hợp lệ nhưng không có cách nào đặt tên hay nạp bản nháp, nên luồng công việc thất bại hoặc đọc nhầm đối tượng cục bộ.
5. Định tuyến dính dường như sửa được triệu chứng cho tới khi một lần khởi động lại, triển khai, lên lịch lại hay chuyển đổi dự phòng đẩy request tiếp theo đi chỗ khác.

Ranh giới đúng có hai phần. Ngữ cảnh giao thức ở lại trong từng request. Trạng thái ứng dụng bền vững sống trong một kho lưu trữ chung dưới một handle do máy chủ tạo và trả cho client. Lần gọi sau truyền handle đó, mọi replica nạp cùng bản ghi, và cấp quyền gắn bản ghi với principal và tenant đã xác thực. Bộ nhớ replica có thể cache một bản ghi, nhưng nó không thể là bản sao duy nhất cần cho tính đúng đắn.

Hãy chọn cơ chế trạng thái theo vòng đời. Biến cục bộ của request phục vụ được một lần gọi. Một phần tiếp nối MRTR ngắn có thể dùng `requestState` được bảo vệ toàn vẹn. Một bản nháp hay tác vụ bền vững cần handle tường minh cùng lưu trữ chung, hạn dùng, kiểm soát đồng thời và tính idempotent. Không đối tượng nào trong số đó là session giao thức MCP.

### Tương thích lưỡng thời trên HTTP

Client hỗ trợ cả máy chủ hiện đại lẫn kiểu cũ sẽ thử POST hiện đại trước. Nếu nhận HTTP `400`, `404` hoặc `405`, nó xem body:

- Một lỗi JSON-RPC hiện đại được nhận diện chứng minh máy chủ là hiện đại. Hãy sửa request hoặc thử lại với một phiên bản được quảng bá. Đừng hạ cấp.
- Body rỗng hoặc một response không nhận diện được có thể là dấu hiệu của máy chủ HTTP+SSE kiểu cũ. Chỉ khi đó mới thử endpoint GET cũ và chờ event `endpoint` kiểu cũ của nó.

Máy chủ có thể hỗ trợ cả hai thời kỳ trong lúc di trú bằng cách định tuyến metadata hiện đại tới bản cài đặt chỉ POST hiện đại và giữ các endpoint cũ riêng cho client cũ. Không bao giờ mô tả hành vi GET, DELETE, session id (mã phiên) hay phát lại kiểu cũ như một phần của `2026-07-28`.

```figure
tp-transport-handshake
```

## Áp dụng

`code/main.py` cài đặt một máy chủ Streamable HTTP hiện đại, hữu hạn bằng thư viện chuẩn Python. Nó kiểm tra Origin và các header phản chiếu, bỏ qua các header phiên đã bị loại, trả JSON cho các lần gọi thường, và minh hoạ một luồng SSE `subscriptions/listen` hữu hạn.

```bash
cd code
python3 main.py --probe
python3 -m unittest discover tests -v
```

Bản probe kiểm tra:

- Origin không hợp lệ bị từ chối;
- khám phá thành công mà không cần session id;
- `Mcp-Session-Id` và `Last-Event-ID` bị bỏ qua;
- header lệch trả `-32020`;
- phiên bản không hỗ trợ trả `-32022` với data `supported` và `requested` chính xác;
- một notification không có id được chấp nhận trả HTTP `202` không body;
- GET và DELETE trả `405`;
- `subscriptions/listen` là một luồng response POST mà thông điệp xác nhận, các notification và kết quả cuối đều mang id subscription của nó.

## Đóng gói

Bài này đóng gói `outputs/skill-mcp-transport-migrator.md`. Nó bỏ session giao thức hiện đại, thêm kiểm tra header-body, thay GET riêng bằng `subscriptions/listen`, và giữ mọi cầu nối kiểu cũ tách bạch rõ ràng.

## Bài tập

1. Bỏ `Mcp-Method` khỏi một POST. Xác nhận HTTP `400` và lỗi `-32020`.
2. Gửi header và body cùng phiên bản `2027-01-01`. Xác nhận HTTP `400`, lỗi `-32022`, và data chính xác `{"supported":["2026-07-28"],"requested":"2027-01-01"}`.
3. Gửi một `Mcp-Name` dạng sentinel Base64 cho một URI tài nguyên không phải ASCII. Xác nhận giá trị đã giải mã được so sánh với `params.uri`.
4. Làm đứt luồng listen hữu hạn trước response cuối cùng. Phát lại nó với id JSON-RPC mới và lấy lại danh sách tools.
5. Thêm một handle luồng công việc tường minh vào tool ping. Gắn nó với một chủ thể cấp quyền mà không dùng ràng buộc kết nối.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa thực sự |
|------|---------|
| stdio | JSON-RPC phân tách bằng dòng mới trên một tiến trình con do client khởi chạy |
| Streamable HTTP | Một endpoint duy nhất, mỗi thông điệp hiện đại là một POST mới |
| Request-scoped SSE | Luồng response của POST chứa các notification liên quan và response cuối |
| `subscriptions/listen` | Request POST sống lâu để nhận các notification thay đổi đã đăng ký |
| Header mismatch | HTTP `400` và JSON-RPC `-32020` khi header phản chiếu không khớp body |
| Origin validation | Phòng thủ DNS-rebinding cho kết nối đến, không phải xác thực |
| Explicit state handle | Token ứng dụng được truyền như một đối số thường thay vì trạng thái phiên ẩn |
| Legacy bridge | Hành vi thời kỳ cũ được giữ riêng chỉ để tương thích |

## Đọc thêm

- [Tổng quan transport MCP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)
- [Transport stdio của MCP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [Streamable HTTP của MCP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Subscription của MCP](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [Changelog MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/changelog)

---

[← Bài trước: Xây dựng MCP client: khám phá, định tuyến và dự phòng hai thời kỳ giao thức](../../08-building-an-mcp-client/docs/vi.md) · [Bài sau: Tài nguyên và Prompt MCP: Ngữ cảnh có địa chỉ cho máy chủ phi trạng thái →](../../10-mcp-resources-and-prompts/docs/vi.md)
