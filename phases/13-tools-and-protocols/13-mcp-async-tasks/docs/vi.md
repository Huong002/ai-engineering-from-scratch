# Phần mở rộng MCP Tasks: công việc bền vững trên một lõi không trạng thái

> MCP không trạng thái không có nghĩa mọi thao tác phải xong trong một request. Phần mở rộng Tasks chính thức trao cho công việc dài hơi (async task — tác vụ bất đồng bộ) một handle bền vững, tường minh. Server có thể trả handle đó từ `tools/call`, bất kỳ instance nào cũng trả lời được `tasks/get`, và đầu vào của client đến qua `tasks/update` mà không cần hồi sinh session giao thức.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 13, Bài 09 (kênh truyền), Giai đoạn 13, Bài 11 (MRTR không trạng thái), Giai đoạn 13, Bài 12 (gợi hỏi)
**Time:** ~90 phút

## Mục tiêu học tập

- Phân biệt transport giao thức không trạng thái với trạng thái tác vụ bền vững của ứng dụng.
- Thương lượng phần mở rộng `io.modelcontextprotocol/tasks` trong capabilities theo từng request và trong `server/discover`.
- Chỉ trả `CreateTaskResult` do server chỉ định với `resultType: "task"` sau khi đã tạo bền vững.
- Hỏi vòng bằng `tasks/get`, đáp ứng đầu vào cho tác vụ bằng `tasks/update`, và yêu cầu huỷ hợp tác bằng `tasks/cancel`.
- Bỏ các giả định cũ về `tasks/status`, `tasks/result` và `tasks/list`.
- Đăng ký nhận thông báo tác vụ tuỳ chọn qua `subscriptions/listen` trên luồng SSE của response POST.
- Mô hình hoá đúng hạn tác vụ, khôi phục sau restart, chống trùng khoá đầu vào và lỗi thực thi.

## Vì sao Tasks là một phần mở rộng

Tasks xuất hiện lần đầu như một tính năng lõi thử nghiệm vào 2025-11-25. Bản thiết kế lại tháng 7 năm 2026 chuyển chúng vào phần mở rộng chính thức `io.modelcontextprotocol/tasks` để client và server có thể tự chọn thêm vòng đời mở rộng mà không phải nới rộng giao thức lõi cho tất cả mọi người.

Đặc tả phần mở rộng vẫn là bề mặt nháp dù đây là nhà chính thức hiện tại của Tasks. Hãy ghim phiên bản phần mở rộng mà SDK của bạn hỗ trợ, chạy các kịch bản tuân thủ, và tách adapter đường truyền khỏi worker cùng miền lưu trữ của bạn.

Dùng tác vụ khi thao tác có một hoặc nhiều thuộc tính sau:

- Nó có thể sống lâu hơn timeout của một request thông thường.
- Một hàng đợi worker hoặc hệ thống job bên ngoài đã nắm việc thực thi.
- Client cần khôi phục sau khi chính nó khởi động lại.
- Thao tác tạm dừng để chờ người dùng hoặc model nhập liệu giữa lúc thực thi.
- Cancellation (huỷ) và việc lấy kết quả bền vững là yêu cầu của sản phẩm.

Đừng tạo tác vụ cho một phép tra cứu tất định rẻ tiền. Một handle, lưu trữ bền, polling (hỏi vòng), hết hạn và huỷ là độ phức tạp thật sự.

## Lõi không trạng thái, ứng dụng có trạng thái

MCP 2026-07-28 loại bỏ `initialize`, `notifications/initialized`, session giao thức và `Mcp-Session-Id`. Điều đó không cấm các sản phẩm có trạng thái.

Task id (mã tác vụ) là trạng thái ứng dụng tường minh:

- Server lưu bền nó trước khi trả về.
- Client có thể cất nó và hỏi vòng lại sau khi khởi động lại.
- Id có thể định tuyến tới bất kỳ replica nào dùng chung kho bền vững.
- Uỷ quyền được kiểm tra trên mọi phương thức tác vụ.
- Việc hết hạn và xoá do các trường của tác vụ quy định, không do vòng đời transport.

Về mặt vận hành, điều này khác với trạng thái ẩn gắn vào một kết nối.

Hãy tách bốn vòng đời riêng:

| Trạng thái | Vòng đời | Nơi thuộc về |
|---|---|---|
| Protocol metadata | Một request | `params._meta`, được kiểm tra lại ở mọi lần gọi |
| Transport work | Một request stdio hoặc một response HTTP | Coordinator đang xử lý với deadline có giới hạn |
| MRTR continuation | Một chuỗi thử lại | `requestState` được bảo vệ toàn vẹn, kèm kiểm soát replay khi cần |
| Durable task | Xuyên request, replica, restart và kết nối lại | Kho ứng dụng dùng chung, khoá theo `taskId` đã được uỷ quyền |

Đưa bản ghi tác vụ vào bộ nhớ tiến trình không làm MCP có trạng thái. Nó làm ứng dụng không đáng tin. Giao thức vẫn không trạng thái, nhưng một `tasks/get` sau đó định tuyến sang replica khác sẽ không tìm lại được bản ghi. Hãy lưu bền trước khi trả handle, rồi làm cho mọi phương thức tác vụ phân giải cùng một bản ghi chia sẻ dưới kiểm tra tenant và principal.

## Thương lượng capability

Client quảng bá hỗ trợ trên mọi request đủ điều kiện:

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "extensions": {
        "io.modelcontextprotocol/tasks": {}
      }
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "lesson-client",
      "version": "1.0.0"
    }
  }
}
```

Server trả về `supportedVersions`, capabilities, `ttlMs` và `cacheScope` chính xác từ `server/discover`, với cùng phần mở rộng nằm dưới capabilities. Vì nó quảng bá tools, nó cũng triển khai `tools/list` bắt buộc. Kết quả đó trả một descriptor `generate_report` tất định, `inputSchema` dạng object hợp lệ, `resultType: "complete"`, metadata danh tính server và gợi ý cache công khai.

Một phương thức tác vụ đến từ client không khai báo phần mở rộng sẽ trả `-32021`, Missing Required Client Capability, với `data.requiredCapabilities` đặt thành `{"extensions":{"io.modelcontextprotocol/tasks":{}}}`. Một chuỗi giao thức không được hỗ trợ trả `-32022` với dữ liệu `supported` và `requested` chính xác; phiên bản thiếu hoặc không phải chuỗi trả `-32602`.

Một envelope không có `id` của JSON-RPC là một notification. Bên nhận có thể xử lý nó, nhưng không phát ra kết quả hay lỗi JSON-RPC nào. Adapter Streamable HTTP trả `202 Accepted` không body cho một notification được chấp nhận.

Hiện tại, chỉ `tools/call` hỗ trợ thực thi có tác vụ. Hãy thiết kế lớp trừu tượng nội bộ sao cho các loại request sau này không buộc phải viết lại phần lưu trữ.

## Server chỉ định việc tạo tác vụ

Cờ cũ phía client `params._meta.task.required` đã bị bỏ. Client khai báo hỗ trợ phần mở rộng, rồi server quyết định một `tools/call` cụ thể có trở thành tác vụ hay không.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "generate_report",
    "arguments": {"size": "large"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "task",
    "taskId": "tsk_786512e29e0d",
    "status": "working",
    "statusMessage": "Preparing report outline.",
    "createdAt": "2026-08-21T10:30:00Z",
    "lastUpdatedAt": "2026-08-21T10:30:00Z",
    "ttlMs": 900000,
    "pollIntervalMs": 1000
  }
}
```

Server không được trả handle này cho tới khi một `tasks/get` với id đó phân giải được. Với kho nhất quán cuối, hãy chờ tới khi đọc thấy được rồi mới trả lời. Nếu không, client có thể nhận một id trông hợp lệ và ngay lập tức gặp "not found".

Response tác vụ là không được yêu cầu theo nghĩa client không xin chế độ tác vụ. Nhưng nó không phải không thương lượng: request hiện tại vẫn phải quảng bá phần mở rộng.

## Hình dạng của tác vụ

Mỗi tác vụ mang:

- `taskId`: định danh ổn định do server sinh;
- `status`: `working`, `input_required`, `completed`, `cancelled`, hoặc `failed`;
- `createdAt` và `lastUpdatedAt`: mốc thời gian ISO 8601;
- `ttlMs`: thời lượng hết hạn tính từ lúc tạo, hoặc `null` nếu không công bố giới hạn;
- `pollIntervalMs` tuỳ chọn: nhịp hỏi vòng tối thiểu mà server gợi ý hiện tại;
- `statusMessage` tuỳ chọn: ngữ cảnh dành cho người dùng hoặc model.

Các trường riêng theo trạng thái chỉ xuất hiện khi liên quan:

- `input_required` gồm `inputRequests`.
- `completed` gồm hình dạng `result` của request gốc.
- `failed` gồm một object `error` của JSON-RPC.

Client nên tôn trọng `pollIntervalMs`. Server có thể giới hạn tần suất khi bị hỏi vòng quá dày và có thể đổi khoảng nghỉ trong suốt vòng đời tác vụ.

## Hỏi vòng bằng `tasks/get`

Client xin một ảnh chụp trạng thái hiện tại:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/get
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tasks/get",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Bản thân `tasks/get` đã hoàn tất, nên kết quả của nó luôn có `resultType: "complete"`. Tác vụ lồng bên trong vẫn có thể có `status: "working"` hoặc `status: "input_required"`.

Khác biệt này chặn một lỗi parser phổ biến:

```text
result.resultType = complete    means the tasks/get RPC finished
result.status = working        means the represented job is still running
```

Không có lệnh gọi `tasks/result`. Khi tác vụ hoàn tất, response `tasks/get` kế tiếp nhúng luôn `CallToolResult` gốc dưới `result`:

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "completed",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:34:12Z",
  "ttlMs": 900000,
  "result": {
    "resultType": "complete",
    "content": [
      {"type": "text", "text": "Generated large report with approved outline."}
    ],
    "structuredContent": {"size": "large", "approved": true},
    "isError": false,
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "tasks-demo",
        "version": "1.0.0"
      }
    }
  },
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "tasks-demo",
      "version": "1.0.0"
    }
  }
}
```

`resultType` ngoài cùng cho biết RPC `tasks/get` đã hoàn tất. `result.resultType` lồng bên trong cho biết lệnh gọi tool gốc đã hoàn tất. Discriminator lồng đó là bắt buộc. `CallToolResult` lồng SHOULD cũng mang `io.modelcontextprotocol/serverInfo` của riêng nó; bài học này đưa nó vào thay vì lưu một payload không có kiểu.

Không có `tasks/list`. Server không session không thể suy ra an toàn tác vụ nào thuộc một danh sách gắn với kết nối. Ứng dụng cần lịch sử nên phơi ra một tool miền đã được uỷ quyền với bộ lọc và quy tắc sở hữu tường minh.

## Đầu vào trong lúc tác vụ thực thi

Đầu vào cho tác vụ và MRTR lõi trông giống nhau nhưng dùng continuation khác nhau.

### Cần đầu vào trước khi tạo tác vụ

Trả `resultType: "input_required"` của lõi từ `tools/call` gốc. Client đáp ứng rồi thử lại chính lệnh gọi gốc đó. Chỉ tạo tác vụ sau khi các vòng MRTR đồng bộ đó kết thúc.

### Cần đầu vào sau khi tạo tác vụ

Đặt tác vụ thành `input_required`. `tasks/get` phơi ra các `inputRequests` còn treo, và client gửi câu trả lời qua `tasks/update`. Client không thử lại `tools/call` gốc.

Ảnh chụp:

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "input_required",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:31:00Z",
  "ttlMs": 900000,
  "inputRequests": {
    "approve_outline": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Approve the generated report outline?",
        "requestedSchema": {
          "type": "object",
          "properties": {"approved": {"type": "boolean"}},
          "required": ["approved"]
        }
      }
    }
  }
}
```

Cập nhật:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/update
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tasks/update",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "inputResponses": {
      "approve_outline": {
        "action": "accept",
        "content": {"approved": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Response thành công là một chấp nhận rỗng kèm `resultType: "complete"`. Thay đổi trạng thái có thể nhất quán cuối, nên client tiếp tục hỏi vòng hoặc lắng nghe.

Mỗi khoá trong `inputRequests` phải là duy nhất trong toàn bộ vòng đời tác vụ. Các ảnh chụp `tasks/get` lặp lại có thể hiển thị cùng khoá đang treo; client khử trùng trên UI và server bỏ qua câu trả lời cho khoá không rõ, đã bị thay thế hoặc đã được đáp. Một cập nhật một phần có thể để tác vụ ở `input_required` cho tới khi mọi khoá bắt buộc đều được trả lời.

## Huỷ là việc hợp tác

`tasks/cancel` báo ý định và trả một chấp nhận hoàn tất rỗng. Chấp nhận đó không bảo đảm worker đã dừng. Công việc có thể xong trước, bỏ qua việc huỷ, hoặc chuyển trạng thái sau đó.

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/cancel
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tasks/cancel",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Với cả ba phương thức tác vụ, `Mcp-Name` phản chiếu `params.taskId`. Nó không lặp lại tên phương thức JSON-RPC. `code/main.py` tập trung quy tắc này trong `make_http_request`.

Worker của bài tôn trọng việc huỷ ngay lập tức, làm các lần gọi lặp lại mang tính idempotent. Client production vẫn phải coi việc huỷ là hợp tác thay vì suy ra trạng thái cuối của tác vụ từ chấp nhận đó.

Đừng dùng `notifications/cancelled` để huỷ một tác vụ. Notification đó thuộc về huỷ request, không thuộc Tasks bền vững.

Khác biệt này hiện rõ ở ranh giới định tuyến. Huỷ request nhắm vào một thao tác JSON-RPC đang bay hoặc response HTTP gắn phạm vi request của nó. Nếu `tools/call` đã trả `resultType: "task"`, request đó đã xong và việc đóng transport của nó không thể đặt tên hay dừng công việc bền vững. `tasks/cancel` là một RPC mới đã được uỷ quyền. Nó mang `params.taskId`, phản chiếu id đó trong `Mcp-Name`, phân giải backend sở hữu tác vụ, ghi nhận ý định huỷ hợp tác, và trả chấp nhận mà không khẳng định worker đã dừng.

Vì thế một gateway phải giữ coordinator request và route tác vụ ở hai bảng khác nhau. Bảng request có thể biến mất khi response xong. Route tác vụ phải sống tới trạng thái kết thúc và hết hạn lưu giữ. [Bài 29: Độ tin cậy, huỷ và điều tiết luồng của MCP](../../29-mcp-reliability-cancellation-and-flow-control/docs/en.md) xây các quy tắc tranh chấp, timeout, idempotency, backpressure và retry cho cả hai đường.

## Thông báo tuỳ chọn

Polling là mức cơ bản. Client muốn nhận progress notification (thông báo tiến độ) đẩy thì gửi `subscriptions/listen` kèm các task id. Với Streamable HTTP, đây là một POST có response là luồng SSE gắn phạm vi request. Không có luồng sự kiện GET độc lập và không có session giao thức để giữ sống.

Server xác nhận các id đã chấp nhận bằng `notifications/subscriptions/acknowledged` rồi có thể gửi ảnh chụp đầy đủ qua `notifications/tasks`. Chấp nhận và mọi thông báo tác vụ đều mang `io.modelcontextprotocol/subscriptionId` trong `_meta`, bằng id của request `subscriptions/listen`. Ngoài ra, mỗi thông báo tác vụ tương đương với những gì `tasks/get` sẽ trả tại thời điểm đó.

Client vẫn phải khai báo phần mở rộng Tasks. Client nên kết nối lại và resumption (tiếp tục) từ các task id bền vững thay vì dựa vào phát lại sự kiện hay `Last-Event-ID`.

## Ngữ nghĩa lỗi

Dùng đúng hai tầng lỗi.

### Lỗi giao thức

Tham số phương thức không hợp lệ hoặc task id không rõ trả một lỗi JSON-RPC, thường là `-32602`. Thiếu hỗ trợ phần mở rộng trả `-32021` kèm object capability bắt buộc.

### Kết cục thực thi tác vụ

- Một kết quả tool bình thường với `isError: true` vẫn là tác vụ `completed`, vì lệnh gọi tool đã tạo ra kết quả xác định của nó.
- Một lỗi JSON-RPC xảy ra trong lúc thực thi hoãn lại làm tác vụ `failed` và lưu lỗi JSON-RPC đó dưới `error`.
- Việc người dùng từ chối có thể tạo ra `cancelled`, một kết quả từ chối đã hoàn tất, hoặc một kết cục an toàn khác theo miền. Hãy ghi rõ lựa chọn.

## Bền vững, hết hạn và quyền sở hữu

Lưu bền ít nhất: task id, trạng thái, mốc thời gian, ttl, khoảng hỏi vòng, quyền sở hữu thao tác gốc, kết quả hoặc lỗi, các yêu cầu đầu vào còn treo, và mọi khoá đầu vào đã phát ra.

Khoá lưu trữ phải bao gồm hoặc phân giải được tenant và principal có thẩm quyền. Biết task id không được cấp quyền truy cập. Hãy kiểm tra quyền sở hữu trên mọi `tasks/get`, `tasks/update`, `tasks/cancel` và mọi subscription.

`ttlMs` được tính từ lúc tạo và có thể thay đổi. Client có thể coi nó là chốt chặn cuối khi tác vụ đã ngừng tạo cập nhật quan sát được. Server có thể đánh dấu lỗi rồi xoá tác vụ hết hạn sau đó. Đừng mô tả nó như lời hứa giữ kết quả đã hoàn tất trong ngần ấy mili-giây sau khi hoàn tất.

Dùng ghi nguyên tử hoặc transaction. Bài học ghi một file tạm rồi đổi tên nguyên tử. Dịch vụ nhiều replica nên dùng kho bền vững chia sẻ cùng lease cho worker hoặc cơ chế kiểm soát đồng thời tương đương.

```figure
tp-task-lifecycle
```

## Tự tay làm

`code/main.py` triển khai một dịch vụ tác vụ tất định:

- `server/discover` trả `supportedVersions`, gợi ý cache và phần mở rộng Tasks.
- `tools/list` trả một descriptor `generate_report` tất định, cache được, với input schema hợp lệ.
- `tools/call` tạo và lưu bền tác vụ trước khi trả `resultType: "task"`.
- Một instance dịch vụ mới nạp lại đúng tác vụ đó, chứng minh khả năng khôi phục sau restart.
- `tasks/get` trả ảnh chụp tác vụ đầy đủ.
- Worker chuyển từ `working` sang `input_required`.
- `tasks/update` nhận một câu trả lời form và trả chấp nhận hoàn tất rỗng.
- Worker lưu một `CallToolResult` lồng với `resultType` và danh tính server của riêng nó, rồi chuyển sang `completed`.
- `tasks/cancel` mang tính idempotent trong cách triển khai này.
- Bộ dựng HTTP đặt `Mcp-Name` bằng `params.taskId` cho `tasks/get`, `tasks/update` và `tasks/cancel`.
- Các hàm hỗ trợ notification dùng `notifications/subscriptions/acknowledged` và `notifications/tasks`, cả hai đều gắn id của request listen.
- Notification không có id không tạo ra response JSON-RPC nào.

Worker tiến bước tường minh thay vì ngủ trong một thread nền. Điều đó làm mọi chuyển trạng thái trở nên tất định và tách ví dụ giao thức khỏi cơ chế hàng đợi.

## Áp dụng

Từ thư mục gốc của repo:

```bash
cd phases/13-tools-and-protocols/13-mcp-async-tasks/code
python3 main.py
python3 -m unittest discover tests -v
```

Chuỗi kết quả mong đợi:

```text
id=0 resultType=complete status=ack
id=1 resultType=task status=working
id=2 resultType=complete status=working
id=3 resultType=complete status=input_required
id=4 resultType=complete status=ack
id=5 resultType=complete status=completed
```

Cũng hãy kiểm tra rằng `tasks/status`, `tasks/result` và `tasks/list` trả method-not-found trong dịch vụ hiện đại.
Kiểm tra rằng `tools/list` mang tính tất định và mọi phương thức tác vụ HTTP hiện hành đều phản chiếu task id qua `Mcp-Name`.

## Đóng gói

`outputs/skill-task-store-designer.md` giờ tạo ra một thiết kế nhận biết phần mở rộng: thương lượng capability, tạo bền vững trước khi trả, các phương thức hiện hành, luồng cập nhật đầu vào, quyền sở hữu, hết hạn, huỷ, subscription, và di trú khỏi các phương thức thử nghiệm đã bị bỏ.

## Bài tập

1. Thêm khoá đầu vào đang treo thứ hai. Gửi một `tasks/update` một phần và chứng minh tác vụ vẫn ở `input_required` cho tới khi cả hai khoá được trả lời.
2. Thêm quyền sở hữu theo tenant vào kho và từ chối một task id hợp lệ nhưng do principal đã xác thực sai trình ra.
3. Thêm lease cho worker kèm hạn. Chứng minh hai instance dịch vụ không thể hoàn tất cùng một tác vụ đồng thời.
4. Triển khai một adapter SSE cho response POST của `subscriptions/listen`. Không thêm GET, `Last-Event-ID`, hay header session.
5. Thêm dọn dẹp tác vụ hết hạn. Phân biệt tác vụ đã hết hạn với task id sai định dạng mà không để lộ sự tồn tại giữa các tenant.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa trong phần mở rộng hiện tại |
|------|----------------------------------|
| Tasks extension | Capability tuỳ chọn `io.modelcontextprotocol/tasks` cho công việc bất đồng bộ bền vững |
| `CreateTaskResult` | Response `resultType: "task"` do server chỉ định cho một request đủ điều kiện |
| `tasks/get` | Hỏi vòng ảnh chụp tác vụ đầy đủ hiện tại, gồm cả kết quả cuối hoặc đầu vào đang chờ |
| `tasks/update` | Gửi câu trả lời cho các `inputRequests` còn treo của một tác vụ |
| `tasks/cancel` | Chấp nhận ý định huỷ một cách hợp tác |
| `input_required` | Trạng thái tác vụ cho biết đầu vào từ client còn treo |
| `pollIntervalMs` | Độ trễ tối thiểu do server gợi ý trước lần hỏi vòng kế tiếp |
| `ttlMs` | Thời lượng hết hạn tính từ lúc tạo tác vụ |
| Durable-before-return | Quy tắc: task id phải phân giải được trước khi handle của nó được gửi đi |
| `notifications/tasks` | Ảnh chụp tác vụ đầy đủ tuỳ chọn, gửi trên một response SSE đã đăng ký |

## Tương thích kế thừa

Bề mặt thử nghiệm 2025-11-25 dùng việc tăng cường tác vụ do client yêu cầu, `tasks/status`, `tasks/result` và `tasks/list` tuỳ chọn. Chỉ giữ những tên đó bên trong một adapter kế thừa đã ghim. Client hiện tại dùng capability phần mở rộng, chấp nhận handle do server chỉ định, hỏi vòng `tasks/get`, cung cấp đầu vào bằng `tasks/update`, và đọc kết quả cuối từ ảnh chụp tác vụ.

## Đọc thêm

- [Phần mở rộng MCP Tasks chính thức](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)

---

[← Bài trước: Phạm vi tường minh và gợi hỏi không trạng thái](../../12-mcp-roots-and-elicitation/docs/vi.md) · [Bài sau: MCP Apps trên giao thức phi trạng thái →](../../14-mcp-apps/docs/vi.md)
