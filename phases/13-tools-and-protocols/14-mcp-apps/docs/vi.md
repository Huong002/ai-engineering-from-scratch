# MCP Apps trên giao thức phi trạng thái

> Một kết quả tương tác vẫn là một trao đổi công cụ và tài nguyên MCP. Phần lõi 2026-07-28 khiến trao đổi đó tự chứa, còn phần mở rộng Apps thêm bề mặt trình duyệt được sandbox.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 13, Bài 07 (máy chủ MCP), Giai đoạn 13, Bài 10 (tài nguyên)
**Time:** ~75 phút

## Mục tiêu học tập

- Công bố MCP app (ứng dụng MCP) qua `server/discover` và năng lực phần mở rộng theo từng yêu cầu.
- Khai báo một tài nguyên `ui://` trên một công cụ trước khi công cụ đó được gọi.
- Trả về kết quả công cụ và tài nguyên đầy đủ trên đường truyền phi trạng thái 2026-07-28.
- Tách thông điệp cầu nối `ui/initialize` của Apps khỏi phần bắt tay lõi MCP đã bị bỏ.
- Áp dụng kiểm tra origin, sandbox (hộp cát), CSP và quyền tối thiểu.

## Vấn đề

Một kết quả văn bản có thể mô tả một dòng thời gian. Nó không thể cho người dùng một dòng thời gian mà họ có thể lọc, xem xét hay thao tác.

MCP Apps giải bài toán trình bày bằng một phần mở rộng tùy chọn. Một định nghĩa công cụ trỏ tới tài nguyên `ui://`. Host (máy chủ nhúng) có thể tải và xem xét tài nguyên đó trước khi công cụ chạy, hiển thị nó trong một iframe (khung nhúng) đã được sandbox, và làm trung gian cho mọi hành động của ứng dụng qua một cầu nối JSON-RPC.

Giao thức lõi đã thay đổi trong 2026-07-28. Đừng bọc một App trong vòng đời kết nối cũ:

- Không có yêu cầu `initialize` ở lõi hay thông báo `notifications/initialized`.
- Không có header `Mcp-Session-Id`.
- Mỗi yêu cầu mang phiên bản giao thức và năng lực client trong `params._meta`.
- Máy chủ hiện thực `server/discover` để client kiểm tra phiên bản, năng lực lõi và phần mở rộng.
- Mọi kết quả thành công đều có bộ phân biệt `resultType`.
- Streamable HTTP dùng một POST cho mỗi yêu cầu. Các điểm vào GET và DELETE hiện đại trả về 405.

Cầu nối Apps vẫn có một phương thức tên `ui/initialize`. Nó thuộc phương ngữ postMessage của iframe. Nó không tạo lại một phiên MCP lõi.

## Khái niệm

### Hai giao thức, một tính năng

Giữ các tầng tường minh:

1. Lõi MCP mang `server/discover`, `tools/list`, `tools/call`, `resources/list` và `resources/read`.
2. Phần mở rộng MCP Apps khai báo giao diện và định nghĩa cầu nối từ iframe tới host.
3. Các quy tắc sandbox của trình duyệt giới hạn những gì giao diện có thể chạm tới.

Mã định danh phần mở rộng là `io.modelcontextprotocol/ui`. Cả hai phía đều tự nguyện tham gia. Client gửi khai báo hỗ trợ phần mở rộng bên trong đối tượng capabilities ở mỗi yêu cầu:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "server/discover",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/ui": {}
        }
      },
      "io.modelcontextprotocol/clientInfo": {
        "name": "timeline-host",
        "version": "1.0.0"
      }
    }
  }
}
```

`clientInfo` được khuyến nghị cho mục đích chẩn đoán. Đó là dữ liệu tự khai, không phải danh tính cấp quyền.

### Khám phá trước khi hiển thị

Kết quả khám phá của máy chủ công bố phần mở rộng:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {},
    "resources": {},
    "extensions": {
      "io.modelcontextprotocol/ui": {}
    }
  },
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "timeline-app-server",
      "version": "2.0.0"
    }
  }
}
```

Máy chủ buộc phải hỗ trợ khám phá. Client không bị buộc gọi khám phá trước mỗi hành động vì mỗi hành động đều mang năng lực của riêng nó.

### Khai báo giao diện trên định nghĩa công cụ

Hợp đồng Apps hiện đại gắn giao diện với công cụ ngay trong `tools/list`:

```json
{
  "name": "notes_timeline",
  "description": "Render a timeline of notes.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  },
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline.html"
    }
  }
}
```

Đây cố ý là metadata có từ trước khi gọi. Host có thể tải trước, cache và rà soát bảo mật HTML trước khi một kết quả yêu cầu hiển thị nó. Các khóa metadata phẳng cũ có thể được mã tương thích chấp nhận, nhưng máy chủ mới nên phát ra dạng lồng `_meta.ui.resourceUri`.

`tools/list` có thể cache trong lõi hiện tại. Hãy kèm thứ tự tất định, `ttlMs` và `cacheScope`. Dùng `private` khi tập công cụ nhìn thấy được thay đổi theo người dùng hoặc token.

### Trả dữ liệu, rồi để host gắn giao diện

Lời gọi công cụ trả về nội dung thông thường cộng thêm dữ liệu có cấu trúc:

```json
{
  "resultType": "complete",
  "content": [
    {"type": "text", "text": "Timeline ready."}
  ],
  "structuredContent": {
    "notes": [
      {"id": "note-1", "title": "Discover", "created": "2026-07-28"}
    ]
  },
  "isError": false
}
```

Host đã biết khung nhìn nào thuộc về công cụ. Đừng tạo ra một khối nội dung mới chỉ để lặp lại URI.

### Phục vụ ứng dụng dưới dạng tài nguyên

Máy chủ công bố `resources` trong kết quả khám phá, nên nó cũng hiện thực thao tác bắt buộc `resources/list`. Mục danh sách tất định của nó gồm URI chuẩn, tên ổn định, mô tả và kiểu MIME. Kết quả danh sách gồm `resultType`, metadata danh tính máy chủ, `ttlMs` và `cacheScope`, giống hệt danh sách công cụ tất định.

Host gửi `resources/read`. Trên Streamable HTTP, yêu cầu có:

```text
POST /mcp
MCP-Protocol-Version: 2026-07-28
Mcp-Method: resources/read
Mcp-Name: ui://notes/timeline.html
```

Giá trị header và body JSON-RPC phải khớp nhau. Lệch nhau là lỗi giao thức `-32020`.

Kết quả chứa tài nguyên HTML và các gợi ý cache:

```json
{
  "resultType": "complete",
  "contents": [
    {
      "uri": "ui://notes/timeline.html",
      "mimeType": "text/html;profile=mcp-app",
      "text": "<!doctype html>...",
      "_meta": {
        "ui": {
          "csp": {
            "connectDomains": [],
            "resourceDomains": [],
            "frameDomains": [],
            "baseUriDomains": []
          },
          "permissions": {}
        }
      }
    }
  ],
  "ttlMs": 60000,
  "cacheScope": "public"
}
```

### Cache UI resource như nội dung có thể thực thi

Một UI resource (tài nguyên giao diện) của App không thể hoán đổi với văn xuôi thông thường. Mục cache của nó có thể thực thi mã cầu nối, hiển thị dữ liệu công cụ và yêu cầu các hành động do host làm trung gian. Hãy khóa cache theo URI `ui://` chuẩn, danh tính và phiên bản máy chủ đã được chấp nhận, digest nội dung tài nguyên, và ngữ cảnh cấp quyền khi `cacheScope` là private. Không bao giờ dùng lại một tài nguyên App private giữa các chủ thể khác nhau vì HTML hoặc metadata chính sách của nó có thể khác dù URI giống hệt.

Vô hiệu hóa mục cache khi `ttlMs` hết hạn, khi ràng buộc `_meta.ui.resourceUri` của công cụ thay đổi, khi phiên bản máy chủ hoặc mốc ghim descriptor đã chấp nhận thay đổi, hoặc khi một đăng ký thay đổi tài nguyên đã được xác nhận nêu đúng URI đó. Tải lại và áp lại việc rà soát CSP cùng quyền trước khi gắn lại. Một iframe cũ không được giữ quyền rộng hơn chỉ vì phiên bản tài nguyên mới chưa tải xong.

### Từ chối sự mơ hồ trên đường truyền trước khi xét chính sách tính năng

Việc kiểm tra có thứ tự cố ý. Đầu tiên kiểm tra hình dạng JSON-RPC và yêu cầu metadata giao thức kiểu chuỗi cùng một ánh xạ năng lực client kiểu đối tượng. Tiếp theo so sánh header định tuyến với body. Chỉ sau đó mới quyết định phiên bản giao thức khớp được có hỗ trợ hay không. Thứ tự này ngăn proxy và máy chủ diễn giải thành hai yêu cầu khác nhau.

| Điều kiện | HTTP | Lỗi JSON-RPC |
|-----------|------|----------------|
| Phiên bản, phương thức hoặc tên ở header và body không khớp | 400 | `-32020` |
| Header và body cùng thống nhất một phiên bản không được hỗ trợ | 400 | `-32022`, với `data` đúng bằng `{"supported":["2026-07-28"],"requested":"<actual>"}` |
| `resources/read` thiếu năng lực phần mở rộng Apps | 400 | `-32021`, với `data.requiredCapabilities.extensions.io.modelcontextprotocol/ui` |
| Phương thức không xác định | 404 | `-32601` |

Một notification JSON-RPC không có `id`, nên máy chủ không bao giờ phát phản hồi JSON-RPC cho nó. Một notification HTTP được chấp nhận trả về 202 với body rỗng. Lỗi có thể đổi mã trạng thái HTTP, nhưng vẫn không thể tạo body lỗi JSON-RPC cho một notification.

### Sandbox là ranh giới, không phải phán quyết tin cậy

Host kiểm soát iframe. App không thể đọc trực tiếp cookie của host, local storage hay DOM của trang. Mọi việc cần quyền phải đi qua cầu nối.

Hãy dùng các mặc định sau:

- Để mọi danh sách miền CSP trống, rồi chỉ thêm những origin mà App cần. Dùng `connectDomains` cho fetch, XHR và WebSocket; dùng `resourceDomains` cho script, style, hình ảnh và font.
- Đóng gói mã và dữ liệu lại khi thấy hợp lý.
- Không xin quyền camera, micro hay vị trí trừ khi một tính năng hiển thị rõ cần tới.
- Ghim `postMessage` vào đúng origin của đối tác và từ chối sự kiện từ mọi origin khác.
- Coi tham số công cụ, kết quả công cụ, văn bản tài nguyên và thông điệp cầu nối là đầu vào không đáng tin.
- Giữ sự đồng ý của người dùng ở phía host. iframe không thể tự phê duyệt hành động hệ trọng của chính nó.

Đừng sao chép một thuộc tính `sandbox` cố định từ hướng dẫn nào đó vào mọi host. Host phải chọn cờ dựa trên mô hình origin của App và thiết kế cách ly của chính nó.

Một miền được phép vẫn là một đường rò rỉ dữ liệu. `connectDomains: ["https://api.example.com"]` nghĩa là mọi script chạy bên trong App đều có thể gửi dữ liệu được phép tới đó. Khớp origin chính xác ngăn nhầm lẫn đích, nhưng không quyết định dữ liệu gửi đi có thích hợp hay không. Mặc định hãy để quyền kết nối trống, tránh đặt bearer token trong iframe, chuyển tiếp các thao tác hẹp qua host khi thấy hợp lý, giới hạn kích thước yêu cầu và phản hồi, và ghi lại hành động người dùng nào gây ra từng yêu cầu ra ngoài. Hãy xử lý `resourceDomains` tách biệt với `connectDomains`; quyền tải một font hay script không được cấp quyền tải dữ liệu tùy ý lên.

### Cầu nối Apps có vòng đời riêng

Cầu nối Apps là một phương ngữ JSON-RPC trên `postMessage`. Nó trao đổi được các notification `ui/initialize` và `ui/*`, và làm proxy cho những phương thức trông giống lõi như `tools/call`.

View gửi `ui/initialize` kèm `appInfo` và một đối tượng `appCapabilities`. Host trả về năng lực và ngữ cảnh host của nó. Chỉ sau phản hồi đó, View mới gửi `ui/notifications/initialized`. Host phải đợi notification Apps này trước khi gửi thông điệp tới View.

Cú bắt tay cục bộ đó tạo một cầu nối giữa một iframe và một host frame. Nó không thương lượng phiên bản giao thức MCP, không tạo trạng thái máy chủ, cũng không cấp một phiên truyền tải. Hãy để ý tiền tố chính xác: `notifications/initialized` của lõi đã bị bỏ, còn `ui/notifications/initialized` của Apps vẫn giữ. Một yêu cầu lõi sinh ra từ một lời gọi công cụ qua cầu nối là một yêu cầu tự chứa mới, với id JSON-RPC mới và metadata yêu cầu đầy đủ.

### Ngữ cảnh host, hành động và việc thu hồi

Host vẫn là bên có thẩm quyền sau khi cầu nối được khởi tạo. View chỉ có thể yêu cầu một hành động công cụ, điều hướng, dùng clipboard hay một tác dụng cần quyền khác thông qua năng lực mà host đã công bố. Host kiểm tra yêu cầu có kiểu, người dùng hiện tại, đích và tham số, áp dụng chính sách phê duyệt, và có thể từ chối. Một cú nhấn nút và một thông điệp cầu nối hợp lệ thể hiện ý định; cả hai đều không cấp thẩm quyền.

Coi theme, kích thước và khả năng tiếp cận là ngữ cảnh host thay đổi liên tục, không phải đầu vào hiển thị một lần:

- Áp dụng token màu sắc và kiểu chữ do host cung cấp, rồi phản ứng khi theme hoặc tùy chọn độ tương phản thay đổi.
- Cho View báo kích thước mong muốn, nhưng để host giới hạn và áp kích thước iframe để nội dung không thể thoát khỏi bố cục hay tạo lớp phủ đánh lừa.
- Giữ nguyên thứ tự bàn phím, focus nhìn thấy được, tên dễ tiếp cận, trạng thái trình đọc màn hình, độ tương phản đủ, zoom và hành vi giảm chuyển động bên trong iframe.
- Kiểm tra lại việc chuyển focus giữa điều khiển của host và điều khiển của View sau khi đổi kích thước và hiển thị lại.

Năng lực có thể bị thu hồi trong khi App đang mở vì người dùng đổi tài khoản, chính sách thay đổi, một máy chủ bị cách ly, hoặc host thu hẹp phạm vi đồng ý. Hãy kiểm tra năng lực và cấp quyền ngay tại thời điểm hành động, không chỉ lúc `ui/initialize`. Khi bị thu hồi, hãy từ chối các lời gọi cần quyền đang chờ, dừng hoạt động mạng không còn hợp chính sách, xóa trạng thái hiển thị nhạy cảm, và gắn lại hoặc quay về văn bản khi bản thân UI resource không còn được chấp nhận. View phải xử lý việc bị từ chối như một kết quả bình thường, không thử lại cho tới khi host nhượng bộ.

### Phương án dự phòng là một phần của hợp đồng

Một máy chủ hiểu Apps vẫn có thể phục vụ những host không công bố phần mở rộng UI:

- Trả cùng công cụ đó mà không có `_meta.ui` trong `tools/list`.
- Giữ một kết quả văn bản hữu ích cho `tools/call`.
- Từ chối `resources/read` cho giao diện bằng lỗi thiếu năng lực.
- Đừng bao giờ giả định có iframe khi quyết định công cụ đã hoàn tất hay chưa.

```figure
t3-ui-sandbox
```

## Tự tay làm

`code/main.py` dựng một mô hình giao thức nhỏ chạy trong tiến trình, không dùng SDK. Nó kiểm tra phong bì yêu cầu hiện tại và các giá trị định tuyến Streamable HTTP, công bố Apps qua `server/discover`, liệt kê công cụ và tài nguyên, thực thi công cụ, và phục vụ một tài nguyên HTML tự chứa.

Mô hình nhận body và header định tuyến đã được phân tích sẵn. Nó không phải một adapter HTTP đầy đủ và không phân tích `Content-Type` hay `Accept`. Xem Bài 09 để có adapter Streamable HTTP đầy đủ, yêu cầu `Content-Type: application/json` và giá trị `Accept` chứa cả `application/json` lẫn `text/event-stream`.

Chạy nó:

```bash
cd phases/13-tools-and-protocols/14-mcp-apps
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Hãy xem bốn điểm trong đầu ra:

1. Mỗi lời gọi là độc lập.
2. Mỗi yêu cầu đều có năng lực trong `_meta`.
3. `resources/list` trả về descriptor ổn định trước mọi lần đọc tài nguyên.
4. Mỗi kết quả đều có `resultType` và metadata danh tính máy chủ.
5. Không xuất hiện mã định danh phiên lõi nào.

## Áp dụng

Bắt đầu với `server/discover`. Xác nhận `io.modelcontextprotocol/ui` xuất hiện trong ánh xạ phần mở rộng của máy chủ. Sau đó gọi `tools/list` hai lần, một lần có năng lực Apps và một lần không. Phản hồi đầu tiên khai báo tài nguyên. Phản hồi thứ hai vẫn là một công cụ chỉ văn bản dùng được.

Đọc `ui://notes/timeline.html`. Tìm trong HTML hai chỗ `hostOrigin` và chốt kiểm tra `event.origin`. Hai dòng đó là bằng chứng tối thiểu nhìn thấy được rằng cầu nối không dùng đích đại diện tùy ý (wildcard).

## Đóng gói

Bài này đóng gói `outputs/skill-mcp-apps-spec.md`. Dùng nó để rà soát hợp đồng App trước khi viết mã khung. Nó buộc người viết phải nêu rõ phong bì lõi hiện tại, thương lượng phần mở rộng, phương án dự phòng, UI resource, chính sách cache, CSP, quyền, các phương thức cầu nối và ranh giới đồng ý.

## Bài tập

1. Đổi năng lực client thành một ánh xạ phần mở rộng rỗng. Xác nhận `tools/list` vẫn giữ công cụ nhưng bỏ ràng buộc giao diện.
2. Gửi `Mcp-Name: ui://notes/other.html` với body đọc dòng thời gian. Xác nhận lỗi `-32020`.
3. Đổi tài nguyên sang `cacheScope: private`. Mô tả điều kiện riêng theo người dùng biện minh cho lựa chọn đó.
4. Chuyển script sang `https://static.example.com/app.js`. Thêm origin đó vào `resourceDomains` và giải thích rủi ro chuỗi cung ứng mới.
5. Thêm một công cụ `notes_open` và định tuyến cú nhấn nút qua host. Giữ việc người dùng phê duyệt ở phía host.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa thực sự |
|------|---------|
| MCP Apps | Phần mở rộng tùy chọn cho HTML tương tác do host MCP hiển thị |
| `io.modelcontextprotocol/ui` | Mã định danh phần mở rộng được cả hai phía công bố |
| `ui://` | Lược đồ tài nguyên cho mẫu giao diện của một App |
| `text/html;profile=mcp-app` | Kiểu MIME cho HTML của MCP App |
| `server/discover` | RPC hiện tại để khám phá giao thức và năng lực |
| `resources/list` | Phương thức liệt kê tài nguyên bắt buộc khi máy chủ công bố tài nguyên |
| `resultType` | Bộ phân biệt bắt buộc cho kết quả thành công kiểu mới |
| `ui/initialize` | Yêu cầu cầu nối Apps đầu tiên, tách khỏi phần khởi tạo lõi đã bị bỏ |
| `ui/notifications/initialized` | Notification báo View của Apps đã sẵn sàng, gửi sau khi host phản hồi |
| CSP | Chính sách trình duyệt giới hạn script, style, hình ảnh và origin mạng |
| Text fallback | Hành vi của công cụ giữ lại cho host không hỗ trợ Apps |

## Đọc thêm

- [MCP 2026-07-28 base protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP Apps overview](https://modelcontextprotocol.io/extensions/apps/overview)
- [MCP Apps build guide](https://modelcontextprotocol.io/extensions/apps/build)
- [Official extension support matrix](https://modelcontextprotocol.io/extensions/client-matrix)

---

[← Bài trước: Phần mở rộng MCP Tasks: công việc bền vững trên một lõi không trạng thái](../../13-mcp-async-tasks/docs/vi.md) · [Bài sau: Bảo mật MCP: Metadata bị đầu độc, định tuyến và trạng thái MRTR →](../../15-mcp-security-tool-poisoning/docs/vi.md)
