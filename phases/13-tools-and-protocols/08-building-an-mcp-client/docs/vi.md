# Xây dựng MCP client: khám phá, định tuyến và dự phòng hai thời kỳ giao thức

> Một MCP client hiện đại lặp lại hợp đồng của nó trong mọi request (yêu cầu). Quyết định tương thích khó nhất của nó là biết khi nào một server cũ thực sự cũ và khi nào một server hiện đại đang báo một lỗi có thể sửa được.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 13, Bài 07
**Time:** ~85 phút

## Mục tiêu học tập

- Xây dựng mọi request MCP `2026-07-28` với metadata hiện hành.
- Thăm dò các server stdio bằng `server/discover` và chọn phiên bản mà cả hai bên cùng hỗ trợ.
- Chỉ cấp quyền thăm dò legacy có giới hạn cho những peer nằm trong allowlist tường minh.
- Chỉ chấp nhận một thời kỳ legacy sau khi xác thực kết quả `initialize` (khởi tạo) dương tính cho một revision được hỗ trợ.
- Hợp nhất các danh sách công cụ một cách tất định mà không lặng lẽ ghi đè khi trùng tên.
- Định tuyến lệnh gọi tới peer sở hữu từng công cụ mà không tự tạo ra các phiên (session) giao thức.

## Vấn đề

Một agent host thường nói chuyện với nhiều hơn một MCP server. Nó phải khám phá từng server, hợp nhất các catalog công cụ, xử lý tên trùng, định tuyến lệnh gọi và phục hồi khi transport (kênh vận chuyển) thất bại.

Revision `2026-07-28` làm trạng thái ổn định đơn giản hơn vì mỗi request tự chứa đầy đủ. Phần tương thích khiến việc khởi động tinh tế hơn. Một client có thể gặp:

- một server hiện đại hỗ trợ phiên bản ưu tiên;
- một server hiện đại trả về lỗi phiên bản hoặc header mà nó nhận ra;
- một server legacy chưa từng nghe tới `server/discover`;
- một server legacy im lặng cho tới khi nhận được `initialize`.

Coi mọi lỗi thăm dò là legacy thì nguy hiểm. Một request hiện đại sai định dạng, một server quá tải, một tiến trình đã chết và một server cũ đều có thể tạo ra cùng một timeout hoặc cùng một lần đóng kết nối. Những tín hiệu đó là mơ hồ. Client phải kết hợp ý định tường minh của người vận hành với bằng chứng giao thức dương tính trước khi chọn thời kỳ legacy.

## Khái niệm

### Một peer, không phải một phiên giao thức

Giữ một bản ghi peer transport cho mỗi tiến trình server hoặc endpoint:

- handle transport hoặc hàm gửi;
- thời kỳ và phiên bản giao thức đã chọn;
- các năng lực (capabilities) của server được khám phá lần cuối;
- danh sách công cụ tất định lần cuối;
- các id request đang chờ để tương quan;
- tình trạng sức khoẻ của transport.

Đây là việc ghi sổ phía client. Nó không phải trạng thái phiên giao thức. Với MCP hiện đại, server vẫn nhận phiên bản và năng lực hiện hành trong mọi request.

### Xây dựng mọi request hiện đại từ đầu

```python
def modern_request(request_id, method, params, version, capabilities):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {
            **params,
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": version,
                "io.modelcontextprotocol/clientCapabilities": capabilities,
                "io.modelcontextprotocol/clientInfo": CLIENT_INFO,
            },
        },
    }
```

Đừng gắn metadata một lần vào một đối tượng kết nối rồi cho rằng nó đã lên tới đường truyền. Hãy đóng dấu và kiểm tra request cuối cùng sau khi serialize.

### Khám phá hiện đại

`server/discover` trả về các phiên bản được hỗ trợ, năng lực của server, hướng dẫn, gợi ý cache và danh tính server được khuyến nghị. Client chọn phiên bản hiện đại cao nhất mà cả hai bên cùng hỗ trợ.

Khám phá là tuỳ chọn với một client chỉ dùng hiện đại, nhưng được khuyến nghị trên stdio. Một số server legacy chấp nhận một thao tác trước khi khởi tạo, nên gửi `tools/list` trước có thể tạo ra một thành công mơ hồ. `server/discover` tạo ranh giới thời kỳ rõ ràng.

### Thăm dò tương thích trên stdio

Một client stdio hai thời kỳ gửi `server/discover` với metadata hiện đại ưu tiên của nó trước mọi request khác. Có ba lớp kết quả:

1. **DiscoverResult.** Server là hiện đại. Chọn một phiên bản cùng hỗ trợ và tiếp tục với metadata theo từng request.
2. **Lỗi hiện đại được nhận ra.** Server là hiện đại. Với `-32022`, hãy chọn từ `data.supported` và thử lại với một id request mới. Với lỗi header hoặc năng lực, hãy sửa request. Đừng gửi `initialize`.
3. **Tín hiệu mơ hồ.** Một lỗi JSON-RPC không nhận ra, timeout, đóng kết nối hoặc phản hồi rỗng không xác định được thời kỳ. Hãy fail closed (từ chối an toàn) trừ khi peer chính xác đó được cấu hình cho tương thích legacy.

Các lỗi giao thức hiện đại được nhận ra gồm:

- `-32020` HeaderMismatch
- `-32021` MissingRequiredClientCapability
- `-32022` UnsupportedProtocolVersion

Các lỗi hiện đại được nhận ra vẫn là hiện đại kể cả khi peer nằm trong allowlist legacy. Khi một server đã chứng minh rằng nó hiểu bộ từ vựng lỗi hiện đại, gửi `initialize` sẽ là một bước hạ cấp.

Đừng coi `-32601` là bằng chứng legacy dương tính. Nó chỉ làm cho một peer nằm trong allowlist tường minh đủ điều kiện cho một lần thăm dò legacy. Quy tắc tương tự áp dụng cho timeout, đóng kết nối hoặc phản hồi rỗng.

### Allowlist là ý định của người vận hành, không phải bằng chứng

Tương thích legacy phải là một thuộc tính tường minh của một cấu hình peer được ghim:

```python
client.add_server("archive", archive_transport, allow_legacy=True)
```

Hãy buộc lựa chọn đó vào lệnh hoặc endpoint đã cấu hình. Đừng dùng wildcard cho phép một server tuỳ ý tự đưa mình vào ngữ nghĩa yếu hơn. Một peer không có `allow_legacy=True` sẽ thất bại sau một kết quả khám phá mơ hồ và không bao giờ nhận `initialize`.

Allowlist cấp quyền thăm dò. Nó không chọn thời kỳ. Client gửi một `initialize` trong một deadline do transport áp đặt, rồi yêu cầu tất cả những điều sau:

- một phản hồi JSON-RPC `2.0` khớp id request;
- đúng một `result` và không có `error`;
- một `protocolVersion` nằm trong tập revision legacy đã cấu hình của client;
- một trường `capabilities` kiểu đối tượng;
- một đối tượng `serverInfo` với các trường `name` và `version` là chuỗi không rỗng.

Timeout, đóng kết nối, phản hồi lỗi, kết quả sai định dạng, id không khớp hoặc revision không hỗ trợ đều fail closed. Chỉ một kết quả dương tính hợp lệ về cấu trúc mới chọn thời kỳ legacy. Code truyền `legacy_probe_timeout_ms` cho adapter transport; một adapter stdio hoặc HTTP thực sự phải áp đặt deadline đó chứ không chỉ ghi lại nó.

Hãy cache thời kỳ đã chọn cho peer transport. Đừng thăm dò lại trước mọi lệnh gọi.

### Legacy là một nhánh tương thích

Khi lần thăm dò có giới hạn trả về bằng chứng legacy dương tính hợp lệ, client dùng phiên bản legacy đã chọn đúng như revision đó định nghĩa:

1. Kiểm tra envelope phản hồi và id tương quan.
2. Kiểm tra revision đã thương lượng nằm trong tập legacy đã cấu hình.
3. Ghi lại năng lực và danh tính server đã xác thực.
4. Chỉ gửi `notifications/initialized` sau khi mọi kiểm tra đều qua.
5. Dùng các dạng request legacy cho suốt vòng đời của transport đó.

Nhánh này tồn tại để tương tác với những peer đã biết. Nó không phải thiết kế mặc định cho server mới hay request mới. Nếu transport khởi động lại hoặc endpoint của nó thay đổi, hãy bỏ cache thời kỳ của peer và thương lượng năng lực (capability negotiation) lại.

### Khám phá và cache công cụ

Với mỗi peer đang hoạt động, gọi `tools/list` (liệt kê công cụ). Kết quả hiện đại gồm `resultType`, `ttlMs` và `cacheScope`. Hãy tôn trọng gợi ý độ mới trong đúng ngữ cảnh cấp quyền. Tải lại sau khi hết hạn hoặc sau một sự kiện thay đổi danh sách đã đăng ký.

Client phải coi một `resultType` bị thiếu từ server legacy là `"complete"`. Đừng đòi các trường cache hiện đại trên một phản hồi từ thời kỳ đã thương lượng trước đó.

Server nên trả về thứ tự tất định. Client cũng nên sắp xếp trước khi hợp nhất để thứ tự registry cục bộ không phụ thuộc vào thời điểm khởi động tiến trình.

### Hợp nhất namespace an toàn với trùng tên

Hai server có thể cùng cung cấp `search`. Hãy chọn một chính sách đã công bố:

1. **Thêm tiền tố khi trùng.** Giữ tên canonical đầu tiên và đưa các lần trùng sau ra dưới dạng `<server>/<tool>`.
2. **Từ chối khi trùng.** Không nạp bản trùng và nêu một lỗi cấu hình rõ ràng.
3. **Ghi đè lặng lẽ.** Không bao giờ dùng cách này. Nó giấu việc server nào nhận một hành động do model chọn.

Lưu cả tên canonical và tên cục bộ. Model thấy tên canonical. Lệnh `tools/call` gửi đi dùng tên cục bộ mà server sở hữu đã công bố.

### Định tuyến một lệnh gọi

Định tuyến là một phép tra cứu thuần:

```text
canonical tool name
  -> peer name + local tool name
  -> new JSON-RPC request id
  -> modern request metadata or explicit legacy shape
  -> matching response id
```

Đừng gửi lệnh gọi khi transport sở hữu nó không khả dụng. Hãy kết nối lại hoặc khởi động lại transport, rồi chạy lại khám phá và `tools/list`. Các request hiện đại đang bay bị mất trên một transport gãy có thể được thử lại với một id JSON-RPC mới khi chính sách an toàn của thao tác cho phép.

### Thông báo và đăng ký

Các thay đổi danh sách và tài nguyên hiện đại chỉ đến trên một luồng `subscriptions/listen` do client mở. Client gửi bộ lọc thông báo, chờ `notifications/subscriptions/acknowledged`, và tương quan các sự kiện với id request listen trong metadata của thông báo.

Khi mất kết nối, hãy mở một request listen mới và tải lại các danh sách hoặc tài nguyên liên quan. Các luồng hiện đại không tiếp tục bằng `Last-Event-ID`.

### Không có request do server khởi xướng

Server hiện đại không gọi client bằng các request JSON-RPC độc lập cho sampling, elicitation hay roots. Chúng trả về `input_required`, và client thử lại request gốc sau khi đáp ứng các request đầu vào được nhúng.

Đừng chặn bộ đọc phản hồi của peer trong khi đáp ứng đầu vào. Giữ tương quan và tạo một id JSON-RPC mới cho lần thử lại.

```figure
tp-client-merge
```

## Áp dụng

`code/main.py` dùng các hàm peer trong cùng tiến trình để các quyết định giao thức luôn nhìn thấy được. Nó kết nối tới hai peer hiện đại và một peer legacy được allowlist có chủ ý, rồi hợp nhất và định tuyến công cụ của chúng. Callable transport nhận một ngân sách timeout để nhánh tương thích không thể giấu một lần thăm dò không giới hạn.

```bash
cd code
python3 main.py
python3 -m unittest discover tests -v
```

Các test chứng minh những ranh giới mà demo thông thường bỏ qua:

- request hiện đại lặp lại metadata;
- `-32022` thử lại khám phá hiện đại mà không khởi tạo;
- lỗi hiện đại được nhận ra không bao giờ hạ cấp, kể cả với peer trong allowlist;
- timeout, đóng kết nối, phản hồi rỗng và lỗi không nhận ra không kích hoạt `initialize` khi không có allowlist;
- một peer trong allowlist chỉ trở thành legacy sau một kết quả `initialize` hợp lệ và được hỗ trợ;
- kết quả legacy sai định dạng và không được hỗ trợ để peer ở trạng thái không khả dụng;
- một thời kỳ được chọn thành công sẽ được cache cho vòng đời của transport.

## Đóng gói

Bài này đóng gói `outputs/skill-mcp-client-harness.md`. Nó dựng bộ khung cho việc đóng dấu request hiện đại, thương lượng thời kỳ trên stdio, hợp nhất namespace tất định, định tuyến và một nhánh tương thích legacy fail closed.

## Bài tập

1. Hãy làm một server giả trả về `-32022` mà không có phiên bản nào cùng hỗ trợ. Xác nhận client thất bại thay vì gửi `initialize`.
2. Đưa một server legacy giả vào allowlist, làm lần thăm dò `initialize` có giới hạn của nó bị timeout, và chứng minh peer vẫn ở trạng thái `unknown` và không khả dụng.
3. Thêm danh sách công cụ `cacheScope: "private"` cho hai ngữ cảnh cấp quyền. Xác nhận client không bao giờ chia sẻ kết quả cache của ngữ cảnh này cho ngữ cảnh kia.
4. Đổi chính sách trùng tên sang từ chối và làm cho việc khởi động thất bại với cả hai tên peer trong thông báo lỗi.
5. Thêm một trình mô phỏng `subscriptions/listen` hữu hạn. Khi luồng mất, hãy listen lại với một id request mới và tải lại công cụ.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa thực sự |
|------|---------|
| Peer | Bản ghi phía client cho một transport server và dữ liệu đã khám phá của nó |
| Protocol era | Metadata theo từng request của thời kỳ hiện đại hoặc ngữ nghĩa khởi tạo của thời kỳ legacy |
| Discovery probe | Lần `server/discover` đầu tiên dùng để nhận diện thời kỳ stdio |
| Recognized modern error | Lỗi chứng minh hành vi hiện đại và cấm dự phòng legacy |
| Legacy allowlist | Cấu hình của người vận hành cho phép một lần thăm dò tương thích có giới hạn cho một peer được ghim |
| Positive legacy evidence | Kết quả `initialize` hợp lệ và có tương quan cho một revision legacy được hỗ trợ tường minh |
| Merged namespace | Tên công cụ canonical trên tất cả peer đang hoạt động |
| Collision policy | Quy tắc thêm tiền tố hoặc từ chối cho tên công cụ trùng nhau |
| Era cache | Hành vi hiện đại hoặc legacy đã chọn, được lưu cho một peer transport |
| Transport recovery | Khởi động lại hoặc kết nối lại, khám phá lại, liệt kê lại và thử lại an toàn với id mới |

## Đọc thêm

- [Đặc tả MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [Khám phá server MCP](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [Transport stdio của MCP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [Đánh phiên bản MCP](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [Công cụ MCP](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)

---

[← Bài trước: Xây dựng một MCP server: Python và TypeScript không trạng thái](../../07-building-an-mcp-server/docs/vi.md) · [Bài sau: Transport MCP: stdio và Streamable HTTP không trạng thái →](../../09-mcp-transports/docs/vi.md)
