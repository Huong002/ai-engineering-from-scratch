# Đầu vào mô hình MCP: Di trú Sampling và MRTR không trạng thái

> MCP 2026-07-28 ngừng hỗ trợ Sampling (lấy mẫu) cho các thiết kế mới và loại bỏ kênh yêu cầu từ máy chủ tới máy khách. Nếu một quy trình hiện có vẫn cần mô hình của máy khách, máy chủ trả về kết quả `input_required` và máy khách gửi lại yêu cầu ban đầu kèm đầu ra của mô hình. Vòng lặp suy luận trở nên tường minh, có giới hạn và không trạng thái ở tầng giao thức.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 13, Bài 07 (máy chủ MCP), Giai đoạn 13, Bài 10 (tài nguyên và prompt)
**Time:** ~75 phút

## Mục tiêu học tập

- Giải thích vì sao Sampling bị ngừng hỗ trợ trong MCP 2026-07-28 và chọn tích hợp mô hình trực tiếp làm mặc định cho máy chủ mới.
- Triển khai một quy trình tương thích đưa `sampling/createMessage` đi qua Multi Round-Trip Requests (MRTR).
- Đặt phiên bản giao thức và năng lực máy khách vào mọi đối tượng `_meta` của yêu cầu.
- Trả về `resultType: "input_required"` và gửi lại phương thức gốc với một id JSON-RPC mới.
- Bảo vệ tính toàn vẹn của `requestState` và gắn nó với chủ thể, phương thức, tham số và thời hạn.
- Giới hạn các vòng lặp có mô hình hỗ trợ bằng kiểm tra năng lực, phê duyệt (approval), kiểm tra hợp lệ phản hồi và giới hạn số vòng.

## Quyết định trước giao thức

Một công cụ như `summarize_repo` cần hai loại việc:

1. Việc tất định: liệt kê tệp, đọc các tệp được phép, kiểm tra đường dẫn và lắp ghép nội dung.
2. Việc của mô hình: chọn các tệp tiêu biểu và tổng hợp bản tóm tắt.

Giờ bạn có hai kiến trúc hợp lệ.

### Máy chủ mới: tích hợp trực tiếp với nhà cung cấp mô hình

Đây là mặc định hiện tại. Máy chủ tự sở hữu việc chọn mô hình, thông tin xác thực, ngân sách, thử lại và khả năng quan sát. Nó trả về một kết quả `tools/call` thông thường cho máy khách MCP.

Chọn cách này khi máy chủ đã là một dịch vụ được lưu trữ hoặc khi hành vi mô hình dự đoán được quan trọng hơn việc dùng mô hình của máy chủ nhà.

### Quy trình Sampling hiện có: di trú sang MRTR

Sampling vẫn tồn tại trong giai đoạn ngừng hỗ trợ. Một máy chủ nhắm tới 2026-07-28 không thể gửi một yêu cầu `sampling/createMessage` sống trở lại máy khách. Thay vào đó, nó nhúng yêu cầu đó vào một `InputRequiredResult`.

Chỉ chọn con đường tương thích này khi dùng mô hình và thông tin xác thực của máy khách là yêu cầu sản phẩm thực sự. Hãy ghi lại một kế hoạch loại bỏ vì các triển khai mới không nên áp dụng Sampling đã ngừng hỗ trợ.

## Hợp đồng không trạng thái

Giao thức tháng 7 năm 2026 không có trao đổi `initialize`, không có `notifications/initialized`, và không có `Mcp-Session-Id`. Mỗi yêu cầu mang theo thông tin trước đây nằm trong bước bắt tay:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Máy chủ kiểm tra phiên bản trên mọi yêu cầu. Phiên bản thiếu hoặc không phải chuỗi là tham số không hợp lệ, `-32602`. Một chuỗi không được hỗ trợ trả về `-32022` với dữ liệu chính xác `{"supported":["2026-07-28"],"requested":"<client version>"}`. Thiếu năng lực Sampling trả về `-32021` với `data.requiredCapabilities` được đặt thành `{"sampling":{}}`.

Một phong bì không có `id` JSON-RPC là một thông báo. Bên nhận có thể xử lý nó, nhưng không phát phản hồi thành công lẫn phản hồi lỗi. Một adapter Streamable HTTP trả về `202 Accepted` không có body cho một thông báo được chấp nhận.

Máy chủ cũng triển khai `server/discover` với khoá `supportedVersions` chính xác, các năng lực, `ttlMs` và `cacheScope` để máy khách có thể biết và lưu bộ nhớ đệm hợp đồng máy chủ trước khi gọi một công cụ. Vì khám phá quảng bá `tools`, máy chủ cũng triển khai `tools/list` bắt buộc. Mô tả `summarize_repo` tất định của nó bao gồm một `inputSchema` dạng đối tượng hợp lệ, `resultType: "complete"`, metadata danh tính máy chủ và các gợi ý bộ nhớ đệm công khai.

Mọi kết quả hiện đại thành công đều có một bộ phân biệt:

- `resultType: "complete"` nghĩa là thao tác đã xong.
- `resultType: "input_required"` nghĩa là máy khách phải đáp ứng các yêu cầu được nhúng và thử lại.
- Các phần mở rộng có thể định nghĩa thêm kiểu kết quả. Phần mở rộng Tasks thêm `"task"` trong Bài 13.

## Một vòng MRTR

Máy chủ không thể gọi máy khách trong khi đang xử lý yêu cầu. Thay vào đó nó trả về kết quả này:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "pick_files": {
        "method": "sampling/createMessage",
        "params": {
          "messages": [
            {
              "role": "user",
              "content": {
                "type": "text",
                "text": "Choose three representative files and return a JSON array."
              }
            }
          ],
          "systemPrompt": "Return only the requested value.",
          "modelPreferences": {
            "costPriority": 0.8,
            "intelligencePriority": 0.2
          },
          "maxTokens": 400
        }
      }
    },
    "requestState": "opaque-integrity-protected-value"
  }
}
```

Máy khách kiểm tra rằng nó hỗ trợ Sampling, áp dụng chính sách phê duyệt và chính sách mô hình của mình, rồi lấy một phản hồi từ mô hình. Sau đó nó gửi một yêu cầu mới với một id JSON-RPC khác:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "inputResponses": {
      "pick_files": {
        "role": "assistant",
        "content": {
          "type": "text",
          "text": "[\"README.md\", \"server.py\", \"docs/intro.md\"]"
        },
        "model": "host-model",
        "stopReason": "endTurn"
      }
    },
    "requestState": "opaque-integrity-protected-value",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}}
    }
  }
}
```

Lần thử lại không phải là sự tiếp nối của một phiên giao thức. Đó là một yêu cầu mới lặp lại phương thức và tham số ban đầu, chỉ thêm `inputResponses` của vòng hiện tại, và phản chiếu `requestState` nguyên từng byte.

MRTR chỉ được phép trên `tools/call`, `prompts/get` và `resources/read`. Máy chủ không được trả về `input_required` từ những phương thức không liên quan.

## Trạng thái nhiều vòng

Bài học này cần hai lần gọi mô hình:

1. `pick_files` trả về một mảng JSON.
2. `summary` trả về đoạn văn cuối cùng.

Mỗi lần thử lại chỉ mang phản hồi của vòng đó. Vì vậy máy chủ đặt pha và dữ liệu trung gian đã kiểm tra vào `requestState` tiếp theo.

Coi giá trị đó như do kẻ tấn công kiểm soát. Ký một tên pha thô là không đủ. Gắn trạng thái với:

- chủ thể đã xác thực, không phải `clientInfo` tự khai;
- phương thức khởi phát;
- một bản băm của tham số gốc;
- một thời hạn ngắn;
- pha hiện tại và các giá trị trung gian đã kiểm tra.

Dùng HMAC khi không cần bảo mật. Dùng mã hoá có xác thực khi máy khách không được đọc trạng thái. Từ chối chữ ký sai, giá trị hết hạn, chủ thể thay đổi hoặc tham số thay đổi bằng `-32602`.

Máy khách không được phân tích hay sửa `requestState`. Việc duy nhất của nó là phản chiếu đúng chuỗi đó khi thử lại.

## Model preferences (tuỳ chọn mô hình) chỉ là gợi ý

`costPriority`, `speedPriority` và `intelligencePriority` là các model preferences (tuỳ chọn mô hình) độc lập. Chúng không phải một phân phối xác suất và không cần cộng lại bằng một. Máy khách có thể bỏ qua chúng vì máy khách sở hữu chính sách mô hình.

Giữ `includeContext` ở `"none"` nếu bạn duy trì một luồng Sampling kế thừa. Các chế độ ngữ cảnh khác làm tăng nguy cơ rò rỉ và bản thân chúng cũng đã bị ngừng hỗ trợ. Truyền ngữ cảnh tường minh tối thiểu trong yêu cầu.

## Các bất biến an toàn

Máy khách là ranh giới tin cậy cho các yêu cầu Sampling được nhúng.

- Cho người dùng thấy máy chủ đang yêu cầu mô hình làm gì khi chính sách yêu cầu phê duyệt.
- Giới hạn số vòng MRTR. Nếu không, một máy chủ độc hại có thể tạo vòng lặp đốt tiền mô hình.
- Kiểm tra mọi phản hồi lấy mẫu trước khi dùng nó làm tên tệp, URL hoặc đầu vào công cụ.
- Giới hạn byte và token mỗi vòng.
- Từ chối một yêu cầu đầu vào không được khai báo trong năng lực hiện tại của máy khách.
- Giữ đầu ra của mô hình ngoài các quyết định phân quyền.
- Ghi log phương thức khởi phát và khoá yêu cầu đầu vào mà không ghi nội dung prompt nhạy cảm.

`clientInfo` và `serverInfo` là metadata hiển thị và chẩn đoán. Không bao giờ dùng cái nào làm danh tính đã xác thực.

```figure
t3-sampling-flip
```

## Tự tay làm

`code/main.py` triển khai toàn bộ luồng hai vòng mà không cần gói bên thứ ba:

- `server/discover` trả về `supportedVersions`, quảng bá hỗ trợ công cụ và trả về gợi ý bộ nhớ đệm.
- `tools/list` trả về một mô tả `summarize_repo` tất định, có thể lưu bộ nhớ đệm với schema đầu vào dạng đối tượng.
- `tools/call` kiểm tra metadata theo từng yêu cầu.
- Kết quả đầu tiên nhúng `sampling/createMessage` để chọn tệp.
- Lần thử lại đầu tiên kiểm tra kết quả mô hình và nhúng yêu cầu thứ hai.
- `requestState` được bảo vệ bằng HMAC mang pha giữa các yêu cầu độc lập.
- Kết quả cuối cùng dùng `resultType: "complete"`.

Mô hình máy chủ giả lập giúp ví dụ tất định. Chỉ thay `fake_host_model` khi kết nối một máy chủ thật. Máy trạng thái phía máy chủ nên giữ tất định và có thể kiểm thử.

## Áp dụng

Từ gốc kho mã:

```bash
cd phases/13-tools-and-protocols/11-mcp-sampling/code
python3 main.py
python3 -m unittest discover tests -v
```

Các mốc kiểm tra mong đợi:

- Khám phá trả về kết quả hoàn chỉnh với `ttlMs` và `cacheScope`.
- Khám phá công cụ trả về cùng mô tả đã sắp xếp, kèm `resultType`, danh tính máy chủ và gợi ý bộ nhớ đệm.
- Thiếu năng lực và phiên bản không được hỗ trợ dùng đúng dữ liệu lỗi `-32021` và `-32022`.
- Một thông báo không có id không tạo ra phản hồi JSON-RPC.
- Id yêu cầu là `[1, 2, 3]`, chứng minh mỗi vòng MRTR là độc lập.
- Hai kết quả đầu là `input_required`.
- Kết quả cuối là `complete` và chứa các tệp được chọn cùng bản tóm tắt.
- Đổi tham số gốc khi thử lại sẽ làm kiểm tra trạng thái yêu cầu thất bại.

## Đóng gói

`outputs/skill-sampling-loop-designer.md` giờ là một trình lập kế hoạch di trú. Trước tiên nó quyết định xem có nên loại bỏ Sampling để thay bằng tích hợp mô hình trực tiếp. Nếu cần tương thích, nó tạo ra các vòng MRTR, ràng buộc trạng thái, cổng năng lực, ngân sách, kiểm tra hợp lệ và kế hoạch loại bỏ.

## Bài tập

1. Đổi phản hồi chọn tệp thành JSON không hợp lệ. Xác nhận máy chủ trả về `-32602` thay vì tin đầu ra mô hình.
2. Đổi `audience` giữa lần gọi đầu và lần thử lại. Giải thích vì sao trạng thái được niêm phong chặn việc tái sử dụng qua các yêu cầu.
3. Thêm vòng thứ ba yêu cầu máy chủ nhà phê bình bản tóm tắt. Mang bản tóm tắt trước đó trong trạng thái đã ký và giới hạn cả luồng ở ba vòng.
4. Loại bỏ Sampling bằng cách thay callback máy chủ giả lập bằng một adapter mô hình do máy chủ sở hữu. Liệt kê những trách nhiệm phê duyệt, thanh toán và khả năng quan sát chuyển sang máy chủ.
5. Thêm một test hết hạn dùng giá trị trạng thái trễ một giây so với hạn của nó.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa trong 2026-07-28 |
|------|------------------------|
| Sampling | Tính năng đã ngừng hỗ trợ, yêu cầu mô hình của máy khách tạo một phần hoàn chỉnh |
| MRTR | Mẫu thử lại không trạng thái cho đầu vào của máy khách cần trong lúc xử lý một yêu cầu |
| `InputRequiredResult` | Kết quả có `resultType: "input_required"` |
| `inputRequests` | Bản đồ do máy chủ gán cho các yêu cầu elicitation, sampling hoặc roots được nhúng |
| `inputResponses` | Kết quả của máy khách trong vòng hiện tại, đánh khoá như `inputRequests` |
| `requestState` | Trạng thái mờ của máy chủ, được máy khách phản chiếu chính xác và máy chủ kiểm tra |
| `resultType` | Bộ phân biệt bắt buộc cho các kết quả MCP hiện đại |
| Direct model integration | Thay thế được khuyến nghị cho máy chủ mới cần suy luận mô hình |
| Capability gate | Quy tắc ngăn gửi một yêu cầu nhúng mà máy khách không quảng bá |
| Loop budget | Số vòng, token, byte, thời gian và chi phí tối đa được phép cho thao tác |

## Tương thích kế thừa

Một máy khách ghim vào 2025-11-25 vẫn có thể dùng luồng `sampling/createMessage` do máy chủ khởi tạo (server-initiated request) qua một kết nối trực tiếp. Chỉ giữ hành vi đó trong một adapter riêng cho phiên bản. Đừng biến đường có phiên thành kiến trúc cho máy chủ 2026-07-28.

Các SDK chính thức có thể chuyển các trình xử lý `input_required` hiện đại cho những bên cũ hơn. Lớp chuyển đó là một ranh giới tương thích, không phải sự cho phép thêm logic phụ thuộc phiên mới.

## Đọc thêm

- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [Nhật ký thay đổi MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [Ngừng hỗ trợ MCP Sampling](https://modelcontextprotocol.io/seps/2577-deprecate-roots-sampling-and-logging)
- [Khám phá máy chủ MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)

---

[← Bài trước: Tài nguyên và Prompt MCP: Ngữ cảnh có địa chỉ cho máy chủ phi trạng thái](../../10-mcp-resources-and-prompts/docs/vi.md) · [Bài sau: Phạm vi tường minh và gợi hỏi không trạng thái →](../../12-mcp-roots-and-elicitation/docs/vi.md)
