# Tài nguyên và Prompt MCP: Ngữ cảnh có địa chỉ cho máy chủ phi trạng thái

> Tool thực hiện thao tác. Resource (tài nguyên) phơi ra nội dung có địa chỉ. Prompt đóng gói các mẫu tin nhắn do người dùng chọn. Một máy chủ MCP tốt giữ những hợp đồng đó tách bạch và dễ đoán.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 13, Bài 07 (Xây dựng máy chủ MCP), Giai đoạn 13, Bài 09 (kênh truyền MCP)
**Time:** ~60 phút

## Mục tiêu học tập

- Chọn giữa tool, resource và prompt dựa trên ý định của bên tiêu thụ.
- Quảng bá bề mặt resource và prompt qua `server/discover` bắt buộc.
- Xây dựng kết quả `resources/list` và `prompts/list` mang tính xác định.
- Áp dụng `ttlMs` và `cacheScope` mà không làm rò rỉ dữ liệu riêng của người dùng.
- Trả lỗi JSON-RPC `-32602` khi resource URI (URI tài nguyên) không hợp lệ hoặc không xác định.
- Mở luồng phản hồi sau POST của `subscriptions/listen` và đối chiếu từng sự kiện bằng ID subscription.
- Coi nội dung resource và prompt template (mẫu prompt) là đầu ra không đáng tin từ máy chủ.

## Bắt đầu từ bên tiêu thụ

Cách dễ nhất để dùng sai MCP là bắt đầu bằng mã cài đặt. Một truy vấn cơ sở dữ liệu trở thành tool vì hàm thì quen thuộc. Một quy trình tái sử dụng trở thành resource vì nó được lưu trong tệp. Một prompt trở thành chính sách ẩn vì host có thể chèn nó vào.

Hãy bắt đầu từ việc ai chọn và họ mong đợi gì.

| Primitive | Ý định chính | Bên chọn | Kết quả điển hình |
|---|---|---|---|
| Tool | Thực hiện một thao tác | Model hoặc ứng dụng | Kết quả hành động có cấu trúc |
| Resource | Đọc nội dung tại một URI | Host, ứng dụng hoặc người dùng | Nội dung văn bản hoặc nhị phân |
| Prompt | Khởi động một quy trình tin nhắn tái sử dụng | Người dùng qua giao diện host | Một hoặc nhiều tin nhắn prompt |

Một ghi chú tại `notes://note-1` là resource vì nó là nội dung có địa chỉ. `delete_note` là tool vì nó thay đổi trạng thái. `review_note` là prompt vì người dùng chọn một quy trình xem xét đã chuẩn bị sẵn.

Đừng phơi một thao tác dưới cả ba dạng chỉ để trông có vẻ đầy đủ. Mỗi bề mặt thêm vào đều cần khám phá, phân quyền, cache, xử lý lỗi, kiểm thử và tài liệu.

## Phong bì phi trạng thái 2026-07-28

Bài học này nhắm tới bản sửa đổi giao thức MCP `2026-07-28`. Trong cấu hình này không có bắt tay khởi tạo hay phiên giao thức. Mọi yêu cầu mang theo phiên bản giao thức và năng lực của client trong các khóa `_meta` dành riêng.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "resources/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      },
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

Máy chủ phải cài đặt `server/discover`. Kết quả của nó quảng bá các phiên bản
được hỗ trợ, năng lực resource và prompt, danh tính cài đặt, và gợi ý cache.
Client có thể gọi trực tiếp một phương thức khác, nhưng khám phá cho nó một
ảnh chụp ổn định trước khi nó dựng giao diện.

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "resources": {"listChanged": true, "subscribe": true},
    "prompts": {"listChanged": true}
  },
  "ttlMs": 3600000,
  "cacheScope": "public"
}
```

Một kết quả bình thường khai báo `"resultType": "complete"`. `_meta` của phản hồi nhận diện cài đặt đang phục vụ qua `io.modelcontextprotocol/serverInfo`. Thông tin này hữu ích cho chẩn đoán. Nó không phải danh tính xác thực. Một yêu cầu mang bản sửa đổi không được hỗ trợ sẽ trả về `-32022` kèm cả bản sửa đổi được yêu cầu và các bản sửa đổi mà máy chủ hỗ trợ.

Hợp đồng phi trạng thái thay đổi bản năng thiết kế của bạn. Một danh sách không thể phụ thuộc vào lời gọi trước đó trên cùng một kết nối. Phân quyền có thể làm thay đổi tập hiển thị vì thông tin xác thực là đầu vào của yêu cầu, nhưng lịch sử kết nối thì không được.

## Resource là những hợp đồng URI ổn định

Resource là nội dung được nhận diện bằng một URI. Hãy thiết kế URI trước khi viết handler.

Các tính chất tốt của một URI:

- Đủ ổn định để đánh dấu hoặc trao đổi giữa các yêu cầu.
- Có namespace thuộc miền của máy chủ.
- Độc lập với ID tiến trình hoặc kết nối.
- Được kiểm tra hợp lệ trước khi truy cập kho lưu trữ.
- Được phân quyền ở mỗi lần đọc.

`notes://note-1` tốt hơn `note-1` vì namespace của nó tường minh. Một máy chủ tệp có thể dùng URI `file://`, nhưng vẫn phải kiểm tra ranh giới thư mục đã cấu hình sau khi phân giải symlink và các đoạn đường dẫn tương đối.

`resources/list` trả về những resource hiện đang hiển thị với bên gọi. Hãy sắp xếp theo một khóa ổn định như URI. Thứ tự xác định ngăn cache miss ồn ào, ngăn ảnh chụp bị thay đổi, và ngăn giao diện host nhảy qua nhảy lại giữa các lần làm mới.

```json
{
  "resultType": "complete",
  "resources": [
    {
      "uri": "notes://note-1",
      "name": "Architecture decision",
      "description": "Why the service uses a stateless boundary",
      "mimeType": "text/markdown"
    }
  ],
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

`resources/read` trả về một hoặc nhiều mục nội dung. URI không xác định không phải là một lần đọc rỗng thành công. Đặc tả Resource hiện hành gán resource URI không hợp lệ hoặc không xác định cho lỗi tham số không hợp lệ của JSON-RPC, mã `-32602`.

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "error": {
    "code": -32602,
    "message": "Unknown or invalid resource URI",
    "data": {
      "uri": "notes://missing"
    }
  }
}
```

Sự phân biệt đó cho phép client tách "không tồn tại" khỏi một tài liệu rỗng hợp lệ. Nó cũng ngăn việc vô tình rơi về một truy vấn tra cứu rộng hơn.

### Resource template

Một resource template mô tả một họ URI có tham số. Dùng nó khi liệt kê từng mục cụ thể sẽ tốn kém hoặc không có giới hạn. Ví dụ, `notes://projects/{project}/decisions/{decision}` cho client biết cách tạo một địa chỉ hợp lệ mà không phải trả về mọi quyết định.

Template không làm yếu việc kiểm tra hợp lệ. Hãy phân tích biến, áp dụng phân quyền, thực thi giới hạn độ dài và ký tự, và dựng truy vấn lưu trữ bằng tham số có kiểu. Không bao giờ nối một đuôi URI tùy ý vào đường dẫn hệ thống tệp hay câu lệnh cơ sở dữ liệu.

### Nội dung không phải chỉ thị đáng tin

Văn bản của resource có thể chứa prompt injection, bí mật, chỉ dẫn gây nhầm lẫn, hoặc đánh dấu sai định dạng. Host nên giữ nguồn gốc và coi nội dung resource là dữ liệu. Máy chủ nên giới hạn kích thước nội dung, trả về MIME type chính xác, che những trường mà bên gọi không được phép xem, và tránh trả về các bản ghi không liên quan.

## Prompt là mẫu do người dùng kiểm soát

Prompt MCP được thiết kế để người dùng chọn một cách tường minh. Host có thể hiển thị chúng dưới dạng lệnh slash, mục menu, hoặc nút quy trình. Giao thức không bắt buộc một giao diện duy nhất.

`prompts/list` phải xác định với cùng quyền của yêu cầu. Mỗi prompt cần một tên ổn định, một mô tả hữu ích, và các khai báo argument (đối số) để host thu thập đầu vào trước `prompts/get`.

```json
{
  "resultType": "complete",
  "prompts": [
    {
      "name": "review_note",
      "title": "Review a note",
      "description": "Review one note for a named concern",
      "arguments": [
        {
          "name": "uri",
          "description": "The note resource URI",
          "required": true
        }
      ]
    }
  ],
  "ttlMs": 600000,
  "cacheScope": "public"
}
```

`prompts/get` phân giải argument thành các tin nhắn. Nó không thay thế chỉ dẫn hệ thống của host. Host quyết định cách các tin nhắn trả về đi vào ngữ cảnh của model và giữ chính sách tin cậy của mình ở mức ưu tiên cao hơn.

Hãy kiểm tra argument của prompt ngay tại ranh giới máy chủ. Một prompt URI nên đi qua đúng bước kiểm tra phân quyền như khi đọc resource trực tiếp. Đừng biến prompt thành kênh phụ vòng qua kiểm soát truy cập resource.

## Gợi ý cache là một phần của tính đúng đắn

`ttlMs` cho client biết kết quả được tái sử dụng trong bao lâu. `cacheScope` mô tả ai được chia sẻ giá trị đã cache đó.

| Phạm vi | Ý nghĩa | Cách dùng điển hình |
|---|---|---|
| `public` | Có thể tái sử dụng giữa các người dùng khi phân quyền cho phép | Danh mục prompt công khai |
| `private` | Gắn với người dùng hoặc ngữ cảnh thông tin xác thực đang yêu cầu | Nội dung ghi chú riêng của người dùng |

Chọn TTL dựa trên tốc độ thay đổi của dữ liệu và mức thiệt hại khi dữ liệu đã cũ. Năm phút có thể phù hợp với một danh mục prompt công khai. Một lần đọc ghi chú riêng tư có thể dùng một phút.

MCP chỉ định nghĩa `public` và `private` làm giá trị `cacheScope`. Với kết quả chứa bí mật hoặc thay đổi nhanh, hãy trả `cacheScope: "private"` cùng `ttlMs: 0`, rồi áp dụng bất kỳ quy tắc no-store chặt hơn nào trong chính sách cache của host. Bản thân `no-store` không phải là giá trị `cacheScope` của MCP.

Gợi ý cache không bao giờ thay thế phân quyền. Khóa cache phải bao gồm mọi chiều của yêu cầu làm thay đổi khả năng hiển thị, gồm tenant, người dùng, scope, locale và con trỏ phân trang. Nếu một cache chia sẻ không thể diễn đạt an toàn những chiều đó, hãy dùng `private` với TTL bằng 0 và một chính sách no-store ở tầng host.

## Đăng ký theo dõi (subscription) dùng luồng phản hồi do client mở

Mẫu subscription hiện đại thay thế RPC `resources/subscribe` trước đây và endpoint sự kiện HTTP GET cũ.

Client gửi `subscriptions/listen` như một yêu cầu JSON-RPC bình thường. Trên Streamable HTTP, đây là một POST mà phản hồi vẫn mở dưới dạng luồng SSE. Đối tượng `notifications` là một allowlist. Máy chủ không được gửi những loại thông báo không được yêu cầu.

```json
{
  "jsonrpc": "2.0",
  "id": 17,
  "method": "subscriptions/listen",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    },
    "notifications": {
      "resourcesListChanged": true,
      "promptsListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

ID của yêu cầu chính là ID của subscription. Trước mọi sự kiện được yêu cầu, máy chủ gửi `notifications/subscriptions/acknowledged`. Bộ lọc của nó chỉ chứa tập con mà máy chủ đã chấp nhận.

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "notifications": {
      "resourcesListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

Mọi sự kiện về sau trên luồng đó đều mang cùng metadata.

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "uri": "notes://note-1"
  }
}
```

Thông báo cho biết resource đã thay đổi. Client đọc lại nó qua `resources/read`, tuân theo phân quyền hiện hành. Nó không giả định rằng sự kiện chứa tài liệu mới.

Nhiều subscription có thể chia sẻ một kênh stdio. ID subscription cho phép client phân kênh chúng. Trên HTTP, đóng luồng phản hồi sẽ hủy subscription. Một máy chủ kết thúc luồng một cách êm ái sẽ trả về phản hồi cuối cùng `resultType: "complete"` tương ứng với yêu cầu ban đầu.

Đừng dùng luồng subscription như một phiên giao thức. Một lần đọc sau đó vẫn là một yêu cầu trọn vẹn và có thể tới bất kỳ phiên bản máy chủ khỏe mạnh nào.

```figure
t3-primitive-sort
```

## Phòng thí nghiệm tương tác

Dùng hình để phân loại năm năng lực từ một công cụ theo dõi dự án: chi tiết issue, tạo issue, mẫu đánh giá sprint, chính sách dự án, và đóng issue. Sau đó quyết định danh sách nào có thể cache công khai, lần đọc nào phải giữ riêng tư, và resource nào xứng đáng nhận thông báo cập nhật.

Với mỗi phân loại, hãy nêu bên chọn. Nếu model thực hiện một thao tác, dùng tool. Nếu host đọc nội dung được định địa chỉ bằng URI, dùng resource. Nếu người dùng khởi động một quy trình tin nhắn đã chuẩn bị, dùng prompt.

## Phòng thí nghiệm thực hành

Chạy trình mô phỏng từ gốc kho mã:

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

Xem bản ghi theo thứ tự này:

1. Xác nhận `server/discover` quảng bá bản sửa đổi hiện hành và cả hai năng lực.
2. Xác nhận cả hai kết quả danh sách đều đã sắp xếp và dùng `resultType: "complete"`.
3. Xác nhận kết quả list và read mang gợi ý cache có chủ đích.
4. Đổi URI đọc thành `notes://missing` và quan sát `-32602`.
5. Xác nhận xác nhận subscription diễn ra trước sự kiện resource.
6. Xác nhận sự kiện và lần đóng êm ái đều mang subscription ID `5`.

Model Python không mở một kết nối HTTP thật. Nó biểu diễn những tin nhắn mà một SDK phải đặt lên luồng phản hồi theo phạm vi yêu cầu. Dùng SDK chính thức để đóng gói khung và truyền tải trong môi trường sản xuất.

## Sản phẩm đóng gói

`outputs/skill-primitive-splitter.md` là một bản đánh giá thiết kế tái sử dụng cho việc chọn primitive MCP. Nó giờ kiểm tra khám phá xác định, phạm vi cache, hành vi với URI không hợp lệ, và các bộ lọc subscription hiện đại.

Bài học cũng đóng gói `assets/primitive-split.svg`, một phiên bản tĩnh của ranh giới primitive và subscription để học ngoại tuyến.

## Kiểm chứng

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

Kết quả mong đợi: chương trình chính in ra một bản ghi JSON và lệnh kiểm thử báo cáo ít nhất mười hai bài kiểm thử đạt.

## Liên hệ đồ án tổng

Dùng hợp đồng này khi máy chủ đồ án tổng của bạn phơi tri thức có địa chỉ bên cạnh các hành động. Bao gồm một ảnh chụp danh mục xác định, một lần đọc resource được phân quyền, một lần phân giải prompt, một trường hợp URI không hợp lệ, và một bản ghi subscription.

Bằng chứng của bạn nên cho thấy không danh sách nào phụ thuộc vào lịch sử kết nối và một sự kiện subscription không bao giờ cấp quyền truy cập vào resource bên dưới.

## Bài tập

1. Thêm một resource template `notes://projects/{project}/notes/{id}` và kiểm tra hợp lệ cả hai biến.
2. Thêm phân trang cho `resources/list` mà vẫn giữ thứ tự xác định.
3. Đổi một resource sang `cacheScope: "private"` với `ttlMs: 0`, thêm một chính sách no-store ở tầng host, và giải thích mối đe dọa biện minh cho cả hai biện pháp kiểm soát.
4. Thêm subscription thay đổi danh sách prompt và chứng minh không sự kiện nào được gửi khi bộ lọc bỏ `promptsListChanged`.
5. Tạo hai subscription đồng thời và chứng minh mỗi sự kiện mang đúng ID yêu cầu.
6. Thêm chủ thể phân quyền vào handler đọc và chứng minh một mục cache không thể vượt qua chủ thể.

## Thuật ngữ chính

- **Resource:** Nội dung được định địa chỉ bằng URI, do máy chủ MCP phơi ra.
- **Prompt:** Một mẫu tin nhắn do người dùng kiểm soát, do máy chủ MCP phơi ra.
- **Deterministic list:** Kết quả khám phá có thành viên và thứ tự ổn định với cùng đầu vào yêu cầu.
- **`ttlMs`:** Thời lượng tươi của cache, tính bằng mili giây.
- **`cacheScope`:** Ranh giới chia sẻ của một kết quả đã cache.
- **`subscriptions/listen`:** Một yêu cầu sống lâu mà luồng phản hồi của nó gửi các thông báo đã được lọc tường minh.
- **Subscription ID:** ID của yêu cầu listen ban đầu, được lặp lại trong metadata thông báo.
- **Invalid parameters:** Lỗi JSON-RPC `-32602`, dùng cho resource URI không hợp lệ hoặc không xác định.
- **Unsupported protocol version:** Lỗi JSON-RPC `-32022`, gồm các bản sửa đổi `supported` và `requested`.
- **`server/discover`:** Phương thức máy chủ bắt buộc, trả về các bản sửa đổi được hỗ trợ, năng lực, danh tính, và gợi ý cache tùy chọn.

## Đọc thêm

- [Resource MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [Prompt MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [Subscription MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [Cache MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/caching)

---

[← Bài trước: Transport MCP: stdio và Streamable HTTP không trạng thái](../../09-mcp-transports/docs/vi.md) · [Bài sau: Đầu vào mô hình MCP: Di trú Sampling và MRTR không trạng thái →](../../11-mcp-sampling/docs/vi.md)
