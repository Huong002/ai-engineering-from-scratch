# Phạm vi tường minh và gợi hỏi không trạng thái

> Roots đã bị ngừng hỗ trợ trong MCP 2026-07-28 và chưa bao giờ là một sandbox bảo mật. Hãy đặt phạm vi vào tham số công cụ hoặc URI tài nguyên mà người ta thấy được, cấp quyền cho nó ở phía máy chủ, và dùng MRTR khi một công cụ thực sự cần người dùng nhập liệu. Người dùng thấy quyết định, model thấy handle, và bất kỳ phiên bản máy chủ nào cũng xử lý được lần thử lại.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 13, Bài 07 (máy chủ MCP), Giai đoạn 13, Bài 11 (MRTR không trạng thái)
**Time:** ~60 phút

## Mục tiêu học tập

- Thay root (gốc thư mục) đã bị ngừng hỗ trợ bằng tham số workspace tường minh, URI tài nguyên, hoặc cấu hình máy chủ.
- Tách biệt gợi ý phạm vi khỏi việc cấp quyền, việc giữ đường dẫn bên trong ranh giới, và sandbox của hệ điều hành.
- Phân phối `elicitation/create` ở chế độ form thông qua một kết quả `input_required` của MRTR.
- Khai báo hỗ trợ elicitation trong capability của từng yêu cầu và từ chối những chế độ không được hỗ trợ.
- Xác thực `accept`, `decline` và `cancel` như ba kết cục khác nhau.
- Gắn việc xác nhận thao tác phá huỷ với một principal đã xác thực, tham số gốc, tập ứng viên, và thời hạn.

## Hai vấn đề trông có vẻ giống nhau

Một công cụ ghi chú nhận yêu cầu: "Xoá báo cáo TPS cũ."

Máy chủ phải trả lời hai câu hỏi khác nhau.

1. Thao tác này được phép chạm tới workspace nào?
2. Người dùng muốn ghi chú nào trong ba ghi chú khớp?

Vấn đề thứ nhất là phạm vi và cấp quyền. Thứ hai là việc làm rõ nghĩa có tương tác. Trộn lẫn hai thứ này dẫn tới những thiết kế nguy hiểm, ví dụ coi một thư mục do client gửi lên là bằng chứng rằng bên gọi được phép xoá mọi thứ bên trong nó.

## Roots là bề mặt cần chuyển đổi

Các bản MCP trước đây cho phép client khai báo root và thông báo cho máy chủ khi danh sách thay đổi. Root chỉ là gợi ý mang tính thông tin. Chúng không giới hạn thứ mà tiến trình máy chủ được đọc, không cấp quyền cho bên gọi, và không tạo ra sandbox của hệ điều hành.

MCP 2026-07-28 ngừng hỗ trợ `roots/list` và `notifications/roots/list_changed` cho các thiết kế mới. Hãy ưu tiên một trong những lựa chọn thay thế tường minh sau:

- Tham số công cụ `workspaceUri` hoặc `directory` khi phạm vi thay đổi theo từng lần gọi.
- URI tài nguyên khi thao tác vốn đã nhắm tới một tài nguyên.
- Cấu hình máy chủ khi một lần triển khai sở hữu một workspace cố định.
- Sandbox tiến trình hoặc hệ thống tệp bị giam (jailed filesystem) khi mã phải thực sự không thể thoát ra.

Nếu một tích hợp 2026-07-28 hiện có vẫn cần `roots/list` trong giai đoạn chuyển đổi, máy chủ nhúng nó vào `inputRequests` của MRTR. Máy chủ không được gửi một yêu cầu ngược đang sống. Đó là một adapter chuyển đổi; các handler mới nên nhận phạm vi tường minh thay thế.

Model có thể thấy và lặp lại một handle tường minh. Phạm vi ẩn trong phiên transport thì khó kiểm tra, khó phát lại, khó audit và khó định tuyến hơn.

### Quy tắc ba lớp

Một URI tường minh vẫn không tự cấp quyền cho chính nó. Hãy thực thi cả ba lớp:

1. **Cấp quyền:** Principal đã xác thực này có được phép dùng workspace này không?
2. **Giữ trong ranh giới:** URI đích sau khi chuẩn hoá có nằm trong ranh giới (boundary) của workspace được cấp quyền không?
3. **Sandbox:** Hệ điều hành có chặn được một máy chủ đã bị xâm nhập thoát ra ngoài không?

Máy chủ chạy được giữ một allowlist các URI workspace được cấp quyền, chuẩn hoá đường dẫn mã hoá phần trăm, kiểm tra ranh giới theo thành phần đường dẫn thực, và kiểm tra lại việc giữ trong ranh giới ngay trước khi xoá.

Kiểm tra tiền tố chuỗi một cách ngây thơ là sai:

```text
allowed:   file:///work/notes
attacker:  file:///work/notes-evil/secret.md
traversal: file:///work/notes/%2e%2e/private.md
```

Cả hai đường dẫn thù địch đều bắt đầu bằng một chuỗi gây hiểu nhầm. Hãy chuẩn hoá trước, rồi so sánh các thành phần đường dẫn. Một máy chủ tệp (filesystem) cho môi trường thật cũng phải phòng chống race của symbolic link và ngữ nghĩa đường dẫn riêng của từng nền tảng.

## Elicitation vẫn tồn tại, nhưng cách phân phối đã đổi

Elicitation (gợi hỏi) là tính năng client hiện hành để thu thập dữ liệu người dùng nhập trong `tools/call`, `prompts/get` hoặc `resources/read`. Tên method vẫn là `elicitation/create`. Thứ đã thay đổi là hướng của luồng truyền tải.

Máy chủ 2026-07-28 không gửi một yêu cầu JSON-RPC ngược. Nó trả về một `InputRequiredResult`:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "delete_choice": {
        "method": "elicitation/create",
        "params": {
          "mode": "form",
          "message": "Choose one matching note and confirm deletion.",
          "requestedSchema": {
            "type": "object",
            "properties": {
              "note_id": {
                "type": "string",
                "enum": ["note-3", "note-7", "note-14"]
              },
              "confirm": {"type": "boolean"}
            },
            "required": ["note_id", "confirm"]
          }
        }
      }
    },
    "requestState": "integrity-protected-delete-state"
  }
}
```

Host hiển thị form. Người dùng có thể chấp nhận, từ chối rõ ràng, hoặc đóng form. Client sau đó thử lại `tools/call` ban đầu với một id mới:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes_delete",
    "arguments": {
      "workspaceUri": "file:///Users/alice/Documents/Notes",
      "title": "TPS report"
    },
    "inputResponses": {
      "delete_choice": {
        "action": "accept",
        "content": {"note_id": "note-14", "confirm": true}
      }
    },
    "requestState": "integrity-protected-delete-state",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

Không có phiên giao thức nào giữa hai lần gọi. Máy chủ xác minh state được echo lại, xác thực phản hồi theo schema mong đợi, kiểm tra rằng ghi chú được chọn nằm trong tập ứng viên đã ký, cấp quyền lại cho workspace, kiểm tra lại việc giữ trong ranh giới, rồi mới xoá.

## Thương lượng capability theo từng yêu cầu

Một client hỗ trợ elicitation chế độ form khai báo:

```json
{
  "io.modelcontextprotocol/clientCapabilities": {
    "elicitation": {"form": {}}
  }
}
```

Một capability elicitation rỗng, `"elicitation": {}`, vẫn tương đương hỗ trợ chỉ form để đảm bảo tương thích. Khai báo tường minh `"elicitation": {"form": {}}` cũng hỗ trợ chế độ form. Khai báo chỉ URL, `"elicitation": {"url": {}}`, thì không. Máy chủ không được nhúng một chế độ không có trong capability của yêu cầu hiện tại, kể cả khi một yêu cầu trước đó đã khai báo nó.

Mỗi yêu cầu cũng mang `io.modelcontextprotocol/protocolVersion`. Version thiếu hoặc không phải chuỗi trả về `-32602`. Một chuỗi không được hỗ trợ trả về `-32022` kèm dữ liệu `supported` và `requested` chính xác. Thiếu hỗ trợ elicitation hoặc chỉ hỗ trợ URL trả về `-32021` với `data.requiredCapabilities` đặt thành `{"elicitation":{"form":{}}}`.

Một envelope không có `id` của JSON-RPC là một notification. Hãy xử lý nó mà không phát ra phản hồi thành công hay lỗi JSON-RPC. Trên Streamable HTTP, một notification được chấp nhận nhận `202 Accepted` không có body.

Nên đưa `clientInfo` vào để chẩn đoán, nhưng nó do chính client tự khai và không thể dùng để xác định người dùng cho việc cấp quyền.

Máy chủ triển khai `server/discover` và trả về `supportedVersions`, capability, `ttlMs` và `cacheScope` với `resultType: "complete"`. Nó không khai báo Roots cho thiết kế hiện đại này. Vì có khai báo tools, nó cũng triển khai `tools/list` bắt buộc. Kết quả đó trả về descriptor `notes_delete` tất định, một `inputSchema` là object hợp lệ, metadata danh tính máy chủ, và các gợi ý cache công khai.

## Chế độ form

Chế độ form dùng một JSON Schema bị giới hạn, thiết kế cho những hộp thoại dễ dùng. Gốc là một object và các thuộc tính của nó là trường nguyên thuỷ phẳng hoặc mảng enum được hỗ trợ. Object lồng sâu và schema tài liệu đa dụng không thuộc về một hộp thoại xác nhận.

Dùng chế độ form để:

- chọn một trong nhiều ứng viên;
- xác nhận một thao tác phá huỷ;
- thu thập các tuỳ chọn không nhạy cảm;
- thu thập một số ít giá trị mà người dùng, không phải model, phải quyết định.

Không dùng chế độ form cho mật khẩu, API key, access token hoặc thông tin thanh toán. Những bí mật đó sẽ đi qua MCP client và có thể lọt vào log hoặc context của model.

Máy chủ xác thực lại nội dung trả về. Việc xác thực form ở phía client cải thiện UX nhưng không tạo ra niềm tin.

## Chế độ URL

Chế độ URL gửi một URL web an toàn cho một tương tác ngoài luồng:

```json
{
  "method": "elicitation/create",
  "params": {
    "mode": "url",
    "message": "Connect the report service to continue.",
    "url": "https://mcp.example.com/connect/report-service"
  }
}
```

Dùng nó khi thông tin nhạy cảm phải đi trực tiếp vào một luồng web do máy chủ kiểm soát, chẳng hạn cấp quyền cho bên thứ ba. Client hiển thị đầy đủ đích đến và lấy sự đồng ý của người dùng (user consent) trước khi mở. Client không được tải trước URL đó.

Một phản hồi `accept` nghĩa là người dùng đã đồng ý mở URL. Nó không chứng minh luồng bên ngoài đã hoàn tất. Khi thử lại, máy chủ kiểm tra state của chính nó và hoặc hoàn tất, hoặc trả về một kết quả `input_required` khác.

Elicitation chế độ URL không thay thế việc cấp quyền giữa MCP client và MCP server. Nó dành cho một tương tác bên ngoài mà MCP server cần thực hiện thay mặt người dùng. Máy chủ phải gắn người dùng trên trình duyệt với đúng principal đã xác thực khởi tạo thao tác MCP.

## Các nhánh phản hồi

Hãy coi các action này là quyết định sản phẩm, không phải bí danh của nhau:

| Action | Ý nghĩa | Hành vi an toàn của máy chủ |
|--------|---------|----------------------|
| `accept` | Người dùng đã gửi tương tác | Xác thực nội dung và tiếp tục |
| `decline` | Người dùng từ chối rõ ràng | Trả về một kết cục từ chối hoàn chỉnh, không phải lỗi |
| `cancel` | Người dùng đóng form hoặc không hoàn tất được | Dừng an toàn và cho phép thử lại sau |

Không bao giờ coi nội dung thiếu là đồng ý. Không bao giờ biến decline thành một vòng lặp nhắc lại.

## Bảo vệ state MRTR của thao tác phá huỷ

Danh sách ứng viên không thể chỉ nằm trong một prompt hay một giá trị Base64 chưa ký. Client kiểm soát mọi thứ nó gửi về.

Bài học ký một payload state chứa:

- principal đã xác thực;
- method khởi phát;
- digest của `workspaceUri` và `title`;
- các note id được phép hiển thị trong form;
- pha của thao tác;
- thời hạn ngắn.

Trước khi thay đổi dữ liệu, máy chủ cũng kiểm tra bản ghi ghi chú đang sống. Điều này bắt được race xoá và trường hợp mục tiêu bị di chuyển ra ngoài workspace sau khi form đã hiển thị.

Với một thao tác tài chính hoặc không thể đảo ngược, chỉ HMAC không ngăn được một state hợp lệ bị phát lại trong thời hạn của nó. Hãy lưu và tiêu thụ nonce đúng một lần trong một replay store dùng chung cho mọi phiên bản handler. Bài học inject một store có giới hạn, tự cắt theo TTL, và giữ claim nguyên tử trong lúc thực hiện việc xoá trong bộ nhớ. Một cơ sở dữ liệu cho môi trường thật nên gộp việc claim nonce và thay đổi dữ liệu vào một transaction hoặc một ranh giới ghi có điều kiện tương đương.

Hãy xác thực tương tác trước khi claim nonce. Một phản hồi sai định dạng hoặc `cancel` không thực hiện thay đổi nào và để state có thể thử lại cho tới khi hết hạn. Một `decline` rõ ràng là kết thúc, nên bài học tiêu thụ nonce mà không xoá gì.

```figure
t3-roots-boundary
```

## Tự tay làm

`code/main.py` minh hoạ một công cụ `notes_delete` hiện đại:

- `tools/list` trả về một descriptor tất định, cache được, với schema workspace và title bắt buộc.
- Phạm vi là một tham số `workspaceUri` tường minh.
- Cấu hình máy chủ cấp quyền cho workspace đó với principal của bài học.
- Chuẩn hoá URI từ chối sự nhầm lẫn tiền tố và đường dẫn duyệt ngược đã mã hoá.
- Mọi thao tác xoá phá huỷ đều yêu cầu elicitation chế độ form.
- Elicitation đi bên trong `resultType: "input_required"`.
- `requestState` đã ký gắn với đúng danh sách ứng viên và tham số gốc.
- Một replay store được inject từ chối cùng state đã chấp nhận hoặc đã từ chối trên các phiên bản máy chủ khác nhau.
- Lần thử lại dùng id yêu cầu mới và trả về `resultType: "complete"`.

Kho dữ liệu nằm trong bộ nhớ để dễ quan sát hành vi giao thức. Các quy tắc bảo mật vẫn giữ nguyên khi dùng cơ sở dữ liệu.

## Áp dụng

Từ gốc repository:

```bash
cd phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/code
python3 main.py
python3 -m unittest discover tests -v
```

Các mốc kiểm tra mong đợi:

- Discovery khai báo tools mà không có Roots.
- Tool discovery trả về `notes_delete` kèm `resultType`, danh tính máy chủ và các gợi ý cache.
- Yêu cầu id `1` trả về form trong `inputRequests.delete_choice`.
- Yêu cầu id `2` echo lại state đã ký và hoàn tất việc xoá.
- Một đường dẫn tiền tố và một đường dẫn duyệt ngược đã mã hoá đều không qua được bước giữ trong ranh giới.
- Một title đã đổi không thể dùng lại state xác nhận ban đầu.
- Một decline để ghi chú không thay đổi.
- Hai object máy chủ dùng chung state ghi chú và replay không thể cùng thực thi một xác nhận.
- Khai báo form rỗng và tường minh đều hoạt động, còn hỗ trợ chỉ URL trả về đúng yêu cầu form `-32021`.
- Lỗi version không được hỗ trợ dùng đúng dạng dữ liệu `-32022`.
- Một notification không có id không tạo ra phản hồi JSON-RPC nào.

## Đóng gói

`outputs/skill-elicitation-form-designer.md` thiết kế phạm vi tường minh, các bước kiểm tra cấp quyền, form MRTR, các nhánh phản hồi và việc gắn state. Nó từ chối coi root đã ngừng hỗ trợ là sandbox hoặc thu thập bí mật qua chế độ form.

## Bài tập

1. Thay replay store trong bộ nhớ bằng SQLite. Dùng một transaction để claim nonce và xoá ghi chú, rồi chứng minh hai tiến trình không thể cùng commit.
2. Thêm thương lượng capability `url` và một luồng thiết lập ngoài luồng. Giữ thông tin của bên thứ ba ra khỏi `inputResponses`.
3. Thay map ghi chú trong bộ nhớ bằng một cơ sở dữ liệu SQLite tạm thời. Kiểm tra lại việc cấp quyền và việc giữ trong ranh giới bên trong transaction thay đổi dữ liệu.
4. Thêm chính sách symbolic link cho một bản triển khai hệ thống tệp thực. Giải thích vì sao chỉ kiểm tra ranh giới từ vựng trên URI không ngăn được symlink thoát ra.
5. Thiết kế một adapter 2025-11-25 ánh xạ đầu ra của handler MRTR hiện đại sang elicitation do máy chủ khởi tạo kiểu cũ. Giữ nó tách biệt khỏi handler hiện tại.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa trong 2026-07-28 |
|------|------------------------|
| Roots | Gợi ý workspace mang tính thông tin đã ngừng hỗ trợ, không phải cấp quyền hay sandbox |
| Explicit scope | Handle workspace, thư mục hoặc tài nguyên hiển thị trong tham số yêu cầu |
| Containment | Kiểm tra theo thành phần đường dẫn sau chuẩn hoá, giữ mục tiêu bên trong một ranh giới |
| Elicitation | Tính năng client để lấy dữ liệu người dùng nhập trong một thao tác MCP |
| Form mode | Dữ liệu người dùng nhập có cấu trúc, in-band, dùng schema phẳng bị giới hạn |
| URL mode | Tương tác ngoài luồng cho luồng công việc nhạy cảm hoặc bên ngoài |
| MRTR | Kết quả yêu cầu nhập liệu không trạng thái, sau đó thử lại với yêu cầu mới |
| `requestState` | State mờ được echo lại chính xác và máy chủ kiểm tra tính toàn vẹn |
| Decline | Người dùng từ chối rõ ràng |
| Cancel | Đóng form hoặc tương tác chưa hoàn tất, không có phê duyệt |

## Tương thích với bản cũ

Với một peer ghim ở 2025-11-25, `roots/list`, `notifications/roots/list_changed` và `elicitation/create` do máy chủ khởi tạo trực tiếp vẫn có thể tồn tại. Hãy dán nhãn adapter đó là legacy. Không để một danh sách Root kiểu cũ bỏ qua việc cấp quyền của máy chủ, và không mang các giả định về phiên giao thức vào handler hiện đại.

## Đọc thêm

- [Elicitation trong MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/client/elicitation)
- [MRTR nhiều vòng trong MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [Roots bị ngừng hỗ trợ trong MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/client/roots)
- [Khám phá máy chủ MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)

---

[← Bài trước: Đầu vào mô hình MCP: Di trú Sampling và MRTR không trạng thái](../../11-mcp-sampling/docs/vi.md) · [Bài sau: Phần mở rộng MCP Tasks: công việc bền vững trên một lõi không trạng thái →](../../13-mcp-async-tasks/docs/vi.md)
