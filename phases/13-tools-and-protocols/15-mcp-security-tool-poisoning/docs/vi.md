# Bảo mật MCP: Metadata bị đầu độc, định tuyến và trạng thái MRTR

> Phi trạng thái không có nghĩa là không cần tin cậy. Nó có nghĩa là mỗi request đều phơi ra bằng chứng mà máy chủ và gateway cần để tự kiểm chứng lời gọi một cách độc lập.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 13, Bài 07 (máy chủ MCP), Giai đoạn 13, Bài 08 (máy khách MCP)
**Time:** ~60 phút

## Mục tiêu học tập

- Coi tool description (phần mô tả công cụ), annotation, thông tin client và thông tin server là dữ liệu không đáng tin.
- Phát hiện metadata poisoning (đầu độc metadata), thay đổi descriptor và trùng tên giữa các máy chủ.
- Kiểm chứng metadata request 2026-07-28 và các header định tuyến của Streamable HTTP.
- Bảo vệ MRTR `requestState` khỏi bị sửa đổi và gắn xác nhận với đúng tham số cụ thể.
- Áp dụng phân quyền và giới hạn tốc độ theo principal, không theo một phiên giao thức đã bị bỏ.

## Vấn đề

Một model đọc tool description để quyết định gọi gì. Một router đọc tên công cụ để quyết định gửi request đi đâu. Người dùng đọc nhãn để quyết định phê duyệt gì. Một descriptor độc hại có thể nhắm vào cả ba.

Hướng dẫn bảo mật chính thức của MCP nói thẳng: description và annotation nên được coi là không đáng tin trừ khi chúng đến từ một máy chủ đáng tin. Kể cả vậy, độ tin cậy của bản triển khai vẫn có thể thay đổi. Một bản cập nhật máy chủ, một gói bị xâm phạm, một sai sót trong registry hay một lần gộp gateway đều có thể làm thay đổi thứ mà model nhìn thấy.

Giao thức hiện tại cũng dịch chuyển ranh giới bảo mật. Trong 2026-07-28 không có handshake lõi và không có phiên transport. Một thiết kế bảo mật chỉ dựa vào `Mcp-Session-Id` để gắn phê duyệt, giới hạn tốc độ hay lịch sử kiểm toán không phải là thiết kế của thời điểm hiện tại.

## Khái niệm

### Bảy bề mặt tấn công đáng kiểm tra

Hãy dùng một danh sách cụ thể thay vì lời dặn mơ hồ "hãy cẩn thận".

1. **Metadata poisoning.** Một description chứa các chỉ dẫn không liên quan đến hành vi đã khai báo của công cụ.
2. **Descriptor rug pull.** Một tên, description, schema hoặc annotation đã được phê duyệt trước đó bị thay đổi.
3. **Cross-server shadowing (che khuất chéo máy chủ).** Hai backend cùng phơi ra một tên công cụ không đủ định danh và việc định tuyến lặng lẽ chọn một bên.
4. **Header và body lẫn lộn.** `Mcp-Method` hoặc `Mcp-Name` không khớp với request JSON-RPC.
5. **Leo thang quyền.** Một peer khai báo một extension hoặc tính năng client và máy chủ nhầm lời khai đó là sự cho phép.
6. **Sửa đổi trạng thái MRTR.** Client đổi `requestState`, trả lời một câu hỏi khác, hoặc dùng lại xác nhận với tham số khác.
7. **Lẫn lộn danh tính trong supply chain (chuỗi cung ứng).** Một tên hiển thị quen thuộc bị coi là bằng chứng về danh tính nhà phát hành hoặc máy chủ.

Các bề mặt này chồng lấn nhau. Ghim hash giúp ích với thay đổi descriptor nhưng không chứng minh descriptor đầu tiên là an toàn. Quét tĩnh bắt được những cụm từ rõ ràng nhưng không bắt được chỉ dẫn tinh vi. Đặt namespace ngăn được một lớp trùng tên nhưng không ngăn được một máy chủ độc hại có namespace. Hãy xếp chồng các biện pháp kiểm soát.

### Phong bì request hiện tại là bằng chứng, không phải danh tính

Mọi request 2026-07-28 đều chứa:

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "elicitation": {"form": {}}
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "security-lab",
      "version": "1.0.0"
    }
  }
}
```

Hãy kiểm chứng version và hình dạng capability trên mọi request. Dùng capability để chọn hình dạng phản hồi tương thích. Đừng dùng `clientInfo` như một principal đã xác thực. Nó là thông tin tự khai.

Cảnh báo tương tự áp dụng cho `io.modelcontextprotocol/serverInfo` trong metadata kết quả. Nó hữu ích cho log và gỡ lỗi. Nó không phải chứng thư, không phải bằng chứng từ registry, cũng không phải quyết định phân quyền.

### Kiểm chứng định tuyến trước khi áp chính sách

Với `tools/call`, Streamable HTTP bao gồm:

```text
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.export
```

Method trên header phải bằng method trong body. Tên trên header phải bằng `params.name`. Từ chối sự không khớp bằng `-32020` trước khi chọn backend, áp RBAC hay tiêu một token giới hạn tốc độ.

Thứ tự này đóng lại một mơ hồ thường gặp: một thành phần phân quyền theo body trong khi thành phần khác định tuyến theo header.

Kiểm chứng trên đường truyền tuân theo đúng một trình tự. Kiểm tra kiểu JSON-RPC và metadata, so sánh giá trị header với body, rồi xem version khớp có được hỗ trợ không. Header không khớp trả HTTP 400 với `-32020`. Nếu header và body khớp nhau ở một version không được hỗ trợ, trả HTTP 400 với `-32022` và `data` đúng bằng `{"supported":["2026-07-28"],"requested":"<actual>"}`. Method không xác định trả HTTP 404 với `-32601`.

Mọi object lỗi đều có `data` tuỳ chọn khi hợp đồng cần thông tin phục hồi có cấu trúc. Notification không có `id`, nên nó không bao giờ nhận phản hồi thành công hay lỗi JSON-RPC. Một notification HTTP được chấp nhận trả 202 với body rỗng.

### Ghim toàn bộ descriptor

Chỉ hash description sẽ bỏ sót thay đổi schema và annotation. Hãy chuẩn hoá rồi hash các trường descriptor mà người dùng đã phê duyệt:

```python
normalized = json.dumps(tool, sort_keys=True, separators=(",", ":"))
digest = hashlib.sha256(normalized.encode()).hexdigest()
```

Lưu digest dưới một key đủ định danh như `notes.export`, cùng bằng chứng về nhà phát hành và thời điểm phê duyệt, ở ngoài ví dụ đồ chơi này.

Trong mỗi lần làm mới:

- Key lạ: cách ly chờ rà soát.
- Cùng key, digest khác: cách ly như một rug pull cho tới khi được phê duyệt lại.
- Tên không đủ định danh bị trùng: yêu cầu đặt namespace tất định.
- Scanner báo: chặn và rà soát toàn bộ descriptor.

Hash bằng nhau chứng minh tính ổn định, không chứng minh an toàn. Một descriptor bị đầu độc vẫn bị đầu độc dù được ghim hoàn hảo.

### Quét tĩnh là dây bẫy

Các mẫu đơn giản có thể gắn cờ thẻ vai trò, chỉ dẫn ghi đè, che giấu, truy cập bí mật và điểm đến mạng bị làm mờ. Chúng đủ rẻ để chạy lúc cài đặt và trong CI.

Chúng không phải bằng chứng ngữ nghĩa. Một description an toàn có thể chứa cụm từ bị gắn cờ trong một cảnh báo hợp lệ. Một description độc hại có thể né mọi cụm từ. Hãy coi kết quả scanner là bằng chứng để rà soát, không phải điểm số vô tội tự động.

### Đặt namespace trước khi gộp

Giả sử hai máy chủ cùng phơi ra `search`. Đừng bao giờ để thứ tự khám phá quyết định bên nào thắng.

```text
notes.search
issues.search
```

Tên đủ định danh là tên công khai ở gateway. Ghi lại ánh xạ tới backend ở nơi riêng. Tên ổn định giúp phê duyệt, kiểm toán, ghim hash và định tuyến theo `Mcp-Name` cùng trỏ về một đối tượng.

### Capability là lời khai tương thích

`clientCapabilities` theo từng request cho máy chủ biết client xử lý được những tính năng giao thức nào. Nó không cấp cho client quyền truy cập công cụ, dữ liệu hay hành động.

Phân quyền vẫn đến từ principal đã xác thực và chính sách tài nguyên. Trình tự là:

1. Xác thực thông tin đăng nhập ở lớp transport.
2. Kiểm chứng version, header và hình dạng request.
3. Kiểm tra tương thích capability.
4. Phân quyền cho principal, công cụ, tài nguyên và tham số.
5. Thực thi hoặc yêu cầu người dùng nhập.

### Bảo vệ xác nhận MRTR phi trạng thái

Một công cụ có hệ quả có thể cần người dùng xác nhận. MCP hiện tại dùng Multi Round-Trip Requests thay cho callback từ máy chủ tới client.

Phản hồi đầu tiên:

```json
{
  "resultType": "input_required",
  "inputRequests": {
    "confirm": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Export notes to archive?",
        "requestedSchema": {
          "type": "object",
          "properties": {
            "confirm": {"type": "boolean"}
          },
          "required": ["confirm"]
        }
      }
    }
  },
  "requestState": "opaque-integrity-protected-value"
}
```

Client lấy dữ liệu nhập rồi thử lại method gốc với một id JSON-RPC mới:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes.export",
    "arguments": {"query": "private", "destination": "archive"},
    "requestState": "opaque-integrity-protected-value",
    "inputResponses": {
      "confirm": {
        "action": "accept",
        "content": {"confirm": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

Mỗi giá trị trong `inputRequests` là một request nhúng hoàn chỉnh với `method` và `params`. Key của nó phải khớp với mục tương ứng trong `inputResponses`. Một elicitation dạng form dùng `requestedSchema` gốc là object, và client phải khai báo capability elicitation dạng form trước khi máy chủ yêu cầu việc đó.

Capability hiện tại có hai cách khai báo form hợp lệ. `{"elicitation":{}}` ngầm hỗ trợ elicitation dạng form, còn `{"elicitation":{"form":{}}}` nói rõ điều đó. Một khai báo chỉ có URL như `{"elicitation":{"url":{}}}` không hỗ trợ request dạng form. Máy chủ trả HTTP 400 với `-32021` và `data.requiredCapabilities` bằng `{"elicitation":{"form":{}}}`.

Hãy coi `requestState` là đầu vào thù địch. Ký hoặc mã hoá nó, kiểm chứng nó, và gắn nó với method, công cụ, tham số chính xác, mục đích, thời điểm hết hạn, principal, và một nonce dùng một lần khi việc phát lại có ý nghĩa. Code của bài dùng HMAC và so khớp tham số chính xác để làm ranh giới này hiện rõ.

Sổ ghi nonce không được nằm bên trong một object gateway. Mô hình chạy được tiêm vào một replay store có giới hạn, tự cắt theo TTL, và có thể chia sẻ giữa nhiều gateway instance. Thao tác chiếm nguyên tử của nó chính là ranh giới thực thi: chỉ một lần chấp nhận đã kiểm chứng hoặc một lần từ chối dứt khoát tường minh mới tiêu thụ trạng thái. Một phản hồi sai định dạng hoặc `cancel` không thực thi gì và vẫn có thể thử lại cho tới khi hết hạn. Một cụm production cần cùng phép chiếm có điều kiện đó trong kho lưu bền chia sẻ.

Đừng lưu ngữ cảnh xác nhận ẩn trong một phiên giao thức. Bất kỳ instance máy chủ nào cũng phải kiểm chứng được lần thử lại.

### Quy tắc hai cho các lời gọi rủi ro cao

Phân loại một lời gọi theo ba trục:

- Nó tiêu thụ đầu vào không đáng tin.
- Nó có thể truy cập dữ liệu nhạy cảm.
- Nó gây ra một hành động bên ngoài có hệ quả.

Một bước tự động duy nhất không nên kết hợp cả ba. Hãy tách nhỏ, giảm quyền, hoặc yêu cầu người dùng nhập tường minh qua MRTR. Đây là một heuristic thiết kế, không phải một tính năng giao thức.

### Giảm quyền trước khi thực thi

Riêng tính phi trạng thái không làm nên an toàn. Nó loại bỏ lịch sử giao thức ẩn, nhưng một request tự chứa vẫn có thể đòi một handler quá nhiều quyền để rò rỉ dữ liệu hoặc tạo thay đổi không thể đảo ngược. An toàn đến từ việc giảm quyền ở từng ranh giới:

1. **Động từ có kiểu.** Phơi ra một thao tác có biên như `archive_note`, không phải một công cụ tổng quát như `run` hay `request` có thể diễn đạt những quyền không liên quan.
2. **Tham số đã kiểm chứng.** Dùng schema đóng ở nơi thực tế cho phép, từ chối trường không rõ, chuẩn hoá định danh một lần, giới hạn kích thước, và kiểm chứng đích đến, tenant và quyền sở hữu tài nguyên trước khi đánh giá chính sách.
3. **Phân quyền hiện hành.** Gắn principal đã xác thực với đúng động từ, tài nguyên, môi trường và tham số đã chuẩn hoá. Annotation của công cụ và capability của client không cấp quyền này.
4. **Phê duyệt gắn với hành động.** Với một lời gọi có hệ quả, gắn phê duyệt với một digest của động từ có kiểu và tham số đã chuẩn hoá, cộng với principal, thời điểm hết hạn và chính sách dùng một lần. Bất kỳ trường nào thay đổi đều cần một quyết định mới.
5. **Từ chối là kết quả hạng nhất.** Mô hình hoá việc từ chối, phê duyệt hết hạn, người dùng từ chối và đích đến không an toàn như những kết quả bình thường không gây tác dụng phụ. Đừng biến việc từ chối thành một công cụ dự phòng yếu hơn.
6. **Bằng chứng kiểm toán đã che dữ liệu.** Ghi lại ai đã hỏi, descriptor và phiên bản chính sách nào đã được chấp nhận, đích đã chuẩn hoá nào được cho phép, vì sao quyết định cho phép hay từ chối, và việc thực thi đã bắt đầu hay chưa. Lưu digest hoặc giá trị đã che thay vì bí mật.

Mỗi bước thu hẹp những gì thành phần tiếp theo được làm. Handler cuối cùng nên nhận một lệnh miền đã được kiểm chứng, không phải text thô của model cộng với thông tin đăng nhập rộng. Hãy lặp lại toàn bộ chuỗi khi MRTR thử lại, khi task cập nhật, hoặc khi gateway chuyển tiếp lời gọi. Một phê duyệt trước đó không biến các request sau thành lưu lượng phiên đáng tin.

### Các đường tương tác hiện tại và kế thừa

Roots, Sampling và Logging đã bị coi là lỗi thời cho các bản triển khai 2026-07-28 mới. Một gateway có thể giữ code của kênh request cũ chỉ như một đường tương thích được chặn theo version.

Đừng xây một lớp bảo vệ mới quanh bộ giới hạn sampling theo từng phiên. Hãy áp quota theo principal đã xác thực, issuer, tài nguyên, công cụ và cửa sổ thời gian. Với công việc tương tác hiện tại, hãy kiểm tra các input request và response của MRTR.

### Kiểm tra transport phi trạng thái

- Chấp nhận các thông điệp MCP hiện đại ở một endpoint POST duy nhất.
- Trả 405 cho GET và DELETE hiện đại.
- Không tạo ra và không phụ thuộc vào `Mcp-Session-Id`.
- Bỏ qua header phiên và header phát lại kiểu cũ như đầu vào quyền hạn.
- Trả JSON hoặc SSE theo phạm vi request cho POST đó.
- Chỉ dùng `subscriptions/listen` cho các thông báo thay đổi dài hạn đã đăng ký tham gia.

```figure
tp-tool-poisoning
```

## Tự tay làm

`code/main.py` triển khai một mô hình gateway bảo mật nhỏ chạy trong tiến trình. Nó chuẩn hoá và ghim toàn bộ descriptor của công cụ, báo cáo metadata poisoning và shadowing, kiểm chứng phong bì request hiện đại cùng các giá trị định tuyến, và thực hiện một lần export có xác nhận hai vòng với `requestState` được ký và một replay store chia sẻ được tiêm vào.

Mô hình bắt đầu sau khi một adapter HTTP đã phân tích body JSON và các header định tuyến. Nó không kiểm chứng `Content-Type` hay `Accept`. Hãy nối cùng dispatcher đó vào adapter Streamable HTTP đầy đủ của Bài 09, adapter này yêu cầu `Content-Type: application/json` và giá trị `Accept` chứa cả `application/json` lẫn `text/event-stream`.

Chạy nó:

```bash
cd phases/13-tools-and-protocols/15-mcp-security-tool-poisoning
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Mẫu này cố ý thay đổi một descriptor. Scanner và phép so sánh digest đưa ra các phát hiện độc lập. Sau đó phần export trình diễn phản hồi `input_required` và lần thử lại phi trạng thái.

## Áp dụng

Thay `SAFE_TOOLS` bằng một snapshot đã chuẩn hoá từ các máy chủ đã được phê duyệt của bạn. Giữ thông tin đăng nhập và bí mật ngoài snapshot. Rà soát mọi descriptor mới hoặc bị thay đổi trước khi cập nhật digest của nó.

Ở gateway, hãy chạy cùng các kiểm tra đó trong lúc khám phá và một lần nữa trước khi dispatch. Cache có thể giảm công việc khám phá, nhưng một phê duyệt đã cache phải hết hạn hoặc bị vô hiệu hoá khi descriptor thay đổi.

## Đóng gói

Bài này đóng gói `outputs/skill-mcp-threat-model.md`. Nó tạo ra một mô hình mối đe doạ theo giao thức hiện tại, trải qua các ranh giới metadata, định tuyến, capability, phân quyền, MRTR, cache, registry và tương thích.

## Bài tập

1. Gắn principal đã xác thực và quyết định phân quyền hiện hành vào trạng thái MRTR đã niêm phong, rồi từ chối một lần thử lại dưới principal khác.
2. Thay replay store trong bộ nhớ bằng một phép chèn có điều kiện bền vững và chứng minh hai tiến trình không thể cùng chiếm một nonce.
3. Tiêm một lỗi sau khi chiếm nonce nhưng trước khi export mô phỏng. Định nghĩa và kiểm thử quy tắc giao dịch hoặc idempotency giúp việc phục hồi an toàn.
4. Đổi `inputSchema` của một công cụ mà không đổi description. Xác nhận rằng việc ghim toàn bộ descriptor bắt được thay đổi đó.
5. Thêm một chính sách từ chối cache công khai khi `tools/list` khác nhau theo principal.
6. Mô hình hoá một máy chủ cũ phía sau gateway. Đặt toàn bộ hành vi handshake và phiên phía sau một nhánh tương thích `2025-11-25` tường minh.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa thực sự |
|------|---------|
| Metadata poisoning | Các chỉ dẫn hoặc tuyên bố đánh lừa được nhúng trong descriptor của công cụ |
| Rug pull | Thay đổi một descriptor đã được phê duyệt trước đó |
| Tool shadowing | Định tuyến mơ hồ do trùng tên không đủ định danh |
| Header mismatch | Header định tuyến và body JSON-RPC không khớp, lỗi `-32020` |
| Hash pin | Digest của toàn bộ descriptor đã được phê duyệt |
| MRTR | Mẫu phản hồi và thử lại phi trạng thái cho dữ liệu nhập do máy chủ yêu cầu |
| `requestState` | Giá trị khứ hồi mờ phải được coi là đầu vào không đáng tin |
| Capability declaration | Tuyên bố về tương thích giao thức, không phải phân quyền |
| Implicit form support | Một object capability `elicitation` rỗng, tương đương hỗ trợ dạng form |
| Qualified tool name | Tên gateway ổn định như `notes.search` |

## Đọc thêm

- [Hướng dẫn bảo mật và tin cậy của MCP](https://modelcontextprotocol.io/specification/2026-07-28#security-and-trust--safety)
- [Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [Transport Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Các tính năng đã lỗi thời](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)

---

[← Bài trước: MCP Apps trên giao thức phi trạng thái](../../14-mcp-apps/docs/vi.md) · [Bài sau: Uỷ quyền MCP: CIMD, ràng buộc issuer, PKCE và step-up →](../../16-mcp-security-oauth-2-1/docs/vi.md)
