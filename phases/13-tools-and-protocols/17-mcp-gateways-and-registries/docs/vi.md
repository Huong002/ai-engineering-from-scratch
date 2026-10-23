# Gateway MCP phi trạng thái và kiểm soát tiếp nhận tại registry (sổ đăng ký)

> Một gateway (cổng kết nối) nên khiến mọi tuyến trở nên tường minh. Giao thức 2026-07-28 trao cho nó ranh giới về method, name, version, capability, identity, cache và trace mà không cần một phiên transport.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 13, Bài 15 (bảo mật), Giai đoạn 13, Bài 16 (uỷ quyền)
**Time:** ~75 phút

## Mục tiêu học tập

- Gộp nhiều MCP server sau một endpoint 2026-07-28 duy nhất mà không cần session affinity.
- Xác thực metadata theo từng request và header định tuyến trước khi áp policy hay chuyển tiếp.
- Hợp nhất tool với namespace ổn định, thứ tự tất định, descriptor pin, RBAC và cache riêng.
- Coi bản ghi registry là bằng chứng khám phá (discovery) vẫn cần admission control (kiểm soát tiếp nhận).
- Định tuyến đúng SSE theo phạm vi request, `subscriptions/listen`, các lần thử lại MRTR và lời gọi của phần mở rộng Tasks.
- Cô lập hỗ trợ handshake và session kiểu cũ khỏi tuyến hiện đại.

## Vấn đề

Nối một client trực tiếp vào một server thì đơn giản. Một triển khai lớn hơn cần câu trả lời nhất quán cho những câu hỏi khó hơn:

- Server nào được phép?
- Principal nào được thấy và gọi từng tool?
- Điều gì xảy ra khi hai backend cùng công bố một tên?
- Thay đổi descriptor được duyệt thế nào?
- Rate limit (giới hạn tần suất) và sự kiện audit được áp ở đâu?
- Instance nào cũng xử lý được request kế tiếp chứ?

Gateway nằm giữa client và các MCP server backend. Nó trình ra một endpoint MCP duy nhất, áp policy xuyên suốt và chuyển tiếp những request đã được duyệt.

Các thiết kế gateway cũ thường dồn một session của client thành nhiều session backend và ghi lại `Mcp-Session-Id`. Đó là thiết kế tương thích kiểu cũ. Phần lõi 2026-07-28 không có protocol session.

## Khái niệm

### Tuyến gateway hiện đại

Với mỗi request:

1. Xác thực principal từ authorization của transport.
2. Kiểm tra `MCP-Protocol-Version`, `Mcp-Method`, `Mcp-Name` và `params._meta`.
3. Cấp quyền cho principal, resource, method, tool và arguments.
4. Áp policy về descriptor, registry, rate và data.
5. Tạo một request mới, tự chứa, cho backend đã chọn.
6. Kiểm tra kết quả từ backend rồi trả về kết quả của gateway.
7. Ghi một sự kiện audit mà không log bí mật.

Không bước nào cần một protocol session ẩn. Trạng thái ứng dụng vẫn có thể tồn tại trong database, handle tường minh, Tasks hoặc trạng thái MRTR được bảo vệ tính toàn vẹn.

### Policy lúc chạy là quyết định chính của gateway

Việc tiếp nhận quyết định backend version nào được vào gateway. Nó không cấp quyền cho một lời gọi đang chạy. Với mỗi request, gateway tính lại policy từ principal đã xác thực, issuer và resource, tenant (đơn vị thuê), method và name khớp, arguments đã chuẩn hoá, descriptor pin đã tiếp nhận, tình trạng backend hiện tại, phần giao của capability, phân loại dữ liệu, trạng thái rate và mọi phê duyệt gắn với hành động.

Thứ tự này quan trọng. Một bản ghi Registry có thể vẫn active trong khi vai trò của người dùng đã bị thu hồi. Một descriptor có thể vẫn được pin trong khi một argument đích vượt qua ranh giới tenant. Một backend có thể vẫn được duyệt trong khi policy sự cố cách ly các lời gọi làm thay đổi trạng thái. Vì thế policy lúc chạy là quyết định cho phép hay từ chối chính, còn bằng chứng từ Registry và descriptor chỉ là đầu vào.

Đừng cache một quyết định cho phép theo kết nối hay theo session id đã bị xoá. Nếu policy không sẵn sàng, hãy theo một failure policy đã khai báo theo lớp thao tác. Mặc định an toàn là đóng lại (fail closed) với các thay đổi trạng thái và các lượt đọc nhạy cảm, còn những tuyến đọc công khai đã được duyệt tường minh có thể dùng policy gần nhất trong thời gian ngắn, chỉ khi mô hình rủi ro của chúng cho phép. Ghi lại policy version và nhánh thất bại nào đã đưa ra quyết định, rồi kiểm tra kết quả từ backend trước khi trả về.

### Một endpoint POST

Streamable HTTP hiện đại gửi mỗi thông điệp JSON-RPC qua POST:

```text
POST /mcp
Authorization: Bearer <gateway-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.search
Accept: application/json, text/event-stream
```

Gateway có thể trả về JSON hoặc SSE theo phạm vi request cho POST đó. GET và DELETE trả 405 với các request hiện đại. `Mcp-Session-Id` và `Last-Event-ID` không tạo ra thẩm quyền, affinity hay hành vi replay.

Giá trị ở header và body phải khớp nhau. Từ chối chỗ lệch bằng `-32020` trước khi tra backend. Nhờ vậy load balancer, gateway và rate limiter định tuyến được mà không cần phân tích toàn bộ body, vẫn giữ toàn vẹn đầu-cuối.

Kiểm tra theo đúng một thứ tự: kiểu của JSON-RPC và metadata, sự bằng nhau giữa header và body, rồi tới việc hỗ trợ version khớp. Lệch nhau thì trả HTTP 400 kèm `-32020`. Nếu header và body thống nhất về một version không được hỗ trợ, trả HTTP 400 kèm `-32022` và `data` đúng bằng `{"supported":["2026-07-28"],"requested":"<actual>"}`. Method không biết thì trả HTTP 404 kèm `-32601`.

`ProtocolError` mang `data` tuỳ chọn, và gateway serialize nó vào object lỗi JSON-RPC. Một notification không có `id`, nên nó không bao giờ nhận thành công hay lỗi JSON-RPC. Một notification HTTP được chấp nhận trả 202 với body rỗng.

### Triển khai discovery ở mọi tầng

Gateway triển khai `server/discover` cho client. Nó cũng khám phá từng backend để biết protocol version, capability và phần mở rộng.

Ví dụ kết quả của gateway:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": true}
  },
  "ttlMs": 30000,
  "cacheScope": "private",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "enterprise-gateway",
      "version": "2.0.0"
    }
  }
}
```

Chỉ công bố phần giao capability mà gateway giữ được từ đầu tới cuối. Một tính năng của backend không tự động an toàn để phơi ra. Một tính năng của gateway mà không có đường tới backend thì công bố cũng vô ích.

`serverInfo` là dữ liệu tự khai để hiển thị và chẩn đoán. Đừng dùng nó làm bằng chứng registry hay bằng chứng về nhà phát hành.

### Capability của client theo từng request

Mỗi request được chuyển tiếp cần một envelope `_meta` hiện hành:

```json
{
  "io.modelcontextprotocol/protocolVersion": "2026-07-28",
  "io.modelcontextprotocol/clientCapabilities": {},
  "io.modelcontextprotocol/clientInfo": {
    "name": "enterprise-gateway",
    "version": "1.0.0"
  }
}
```

Đừng sao chép mù capability của client bên ngoài cho backend. Gateway chính là client của backend. Chỉ công bố những tính năng mà gateway sẽ trung gian đúng cách.

### Namespace tất định

Hợp nhất tool của backend dưới những tên công khai ổn định:

```text
notes.search
notes.create
issues.list
issues.open
```

Giữ một ánh xạ từ tên công khai sang backend và tên tool gốc. Đừng bao giờ chọn cái đầu hay cái cuối khi trùng tên. Tên công khai là một phần của hợp đồng phê duyệt và audit, nên đổi nó là một cuộc migration.

`tools/list` phải tất định. Khi khả năng hiển thị phụ thuộc principal, trả `cacheScope: private`. Một `ttlMs` có giới hạn giúp giảm tải discovery ở backend mà không cho một danh sách riêng của người dùng rò rỉ sang ngữ cảnh cấp quyền khác.

Mỗi descriptor tool được phơi ra gồm tên ổn định, mô tả và `inputSchema` có gốc là object. Việc đặt namespace không được xoá các trường descriptor bắt buộc. Kết quả danh sách đầy đủ cũng gồm `resultType`, metadata định danh server và các gợi ý cache.

### Pin các descriptor đã duyệt

Lúc tiếp nhận, chuẩn hoá toàn bộ descriptor và lưu digest của nó dưới tên công khai đã định danh đầy đủ. Lúc liệt kê và lúc gọi, so sánh descriptor đang sống với digest đã duyệt.

Nếu nó thay đổi:

- Gỡ nó khỏi `tools/list`.
- Từ chối các lời gọi trực tiếp.
- Phát một sự kiện audit.
- Yêu cầu policy hoặc con người duyệt lại trước khi cập nhật pin.

Gateway là điểm cưỡng chế tập trung hữu ích, nhưng nó không biến một descriptor lần đầu xuất hiện thành descriptor an toàn. Vẫn cần rà soát ban đầu.

### Registry giúp khám phá, không quyết định

Một `server.json` trong Registry cung cấp metadata công bố. Một bản ghi dựa trên package có thể trông như sau:

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
  "name": "com.example/notes",
  "description": "Example notes MCP server.",
  "version": "1.0.0",
  "packages": [
    {
      "registryType": "npm",
      "identifier": "@example/notes-mcp",
      "version": "1.0.0",
      "transport": {"type": "stdio"}
    }
  ]
}
```

Metadata công bố không mang quyết định an ninh của gateway. Hãy giữ bằng chứng về nhà phát hành đã xác minh và provenance trong một trạng thái tiếp nhận riêng:

```json
{
  "registryName": "com.example/notes",
  "registryVersion": "1.0.0",
  "publisher": {"namespace": "com.example", "status": "verified"},
  "provenance": {
    "source": "registry.modelcontextprotocol.io",
    "recordId": "com.example/notes@1.0.0"
  },
  "admission": {"status": "approved", "reviewedBy": "gateway-policy"}
}
```

Gateway kiểm tra hình dạng `server.json` rồi nối nó với trạng thái bên ngoài đó. Gateway vẫn cần một policy tiếp nhận.

Với mỗi backend đã tiếp nhận, ghi lại:

- Định danh chính xác của registry và bản ghi.
- Bằng chứng namespace hoặc domain của nhà phát hành đã xác minh.
- Transport và endpoint được phép.
- Version đã pin hoặc policy nâng cấp đã duyệt.
- Digest của artifact hoặc descriptor.
- Issuer và resource cấp quyền.
- Người duyệt, thời điểm duyệt và thời hạn.

Đừng chấp nhận một server chỉ vì tên hiển thị của nó giống một sản phẩm quen thuộc. Đừng coi sự có mặt trong registry là một lần rà soát an ninh vận hành. Server nội bộ có thể được tiếp nhận qua cùng schema bằng chứng đó, kể cả khi chúng không bao giờ xuất hiện trong một registry công khai.

Bài học này triển khai đường nối của gateway: ghép bằng chứng công bố với tiếp nhận cục bộ trước khi một backend trở nên định tuyến được. [Lesson 30: MCP Registry Supply Chain, Admission, Drift, and Rollback](../../30-mcp-registry-supply-chain-and-drift/docs/en.md) xây dựng toàn bộ control plane cho chứng minh namespace chính xác, provenance của artifact, pin bất biến, descriptor drift đang sống, đối soát trạng thái Registry, một sổ tiếp nhận chống giả mạo và rollback có bằng chứng. Hãy giữ trạng thái chuỗi cung ứng đó tách khỏi quyết định runtime theo từng request ở trên.

### Trung gian hoá credential

Gateway xác thực người gọi nó và xác thực riêng với các backend. Credential của backend không bao giờ tới tay client.

Giữ các ràng buộc này thật tường minh:

```text
outer principal -> gateway role and policy
backend issuer + resource -> backend registration and token
```

Đừng bao giờ chuyển token của gateway bên ngoài cho backend. Đừng bao giờ dùng lại token của backend ở một issuer hay resource khác. Nếu một tool hành động thay mặt người dùng cuối, hãy giữ uỷ quyền đó bằng một mô hình exchange hoặc claims được thiết kế, thay vì mạo danh người dùng bằng một credential dịch vụ dùng chung.

### Rate limit không cần session

Khoá giới hạn theo principal đã xác thực, issuer, resource, tool công khai, hạng chi phí và cửa sổ thời gian. Session id không tồn tại, mà nếu có thì cũng dễ bị xoay vòng.

Hãy kiểm tra những thứ rẻ trước khi tiêu tốn việc đắt. Quyết định xem các lời gọi bị từ chối có tính vào giới hạn lạm dụng, vào quota kinh doanh, hay cả hai.

### Audit chuỗi quyết định

Ghi đủ để dựng lại một lời gọi:

- Định danh request và trace.
- Principal và issuer đã xác thực.
- Tool công khai và tuyến backend.
- Version của descriptor pin.
- Quyết định policy và lý do.
- Độ trễ và lớp kết quả.
- Vòng MRTR hoặc định danh task khi có.

Che đi bearer token, authorization code, refresh token, bí mật thô và những argument nhạy cảm không cần thiết.

### SSE theo phạm vi request

Một POST bình thường có thể trả SSE theo phạm vi request khi công việc được stream trong chính request đó. Đóng luồng phản hồi là huỷ request HTTP hiện đại đang bay đó.

Đừng tạo một luồng GET riêng và đừng hứa hẹn replay theo Last-Event-ID. Đó là những giả định transport cũ.

### Thông báo thay đổi dài hạn

Với thông báo thay đổi danh sách và tài nguyên, một client hiện hành gửi `subscriptions/listen` qua POST và nhận một phản hồi SSE. Bộ lọc thông báo dùng đúng các trường phẳng `toolsListChanged`, `promptsListChanged`, `resourcesListChanged` và `resourceSubscriptions`:

```json
{
  "jsonrpc": "2.0",
  "id": "listen-tools",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

Sự kiện đầu tiên xác nhận tập con được hỗ trợ. Định danh subscription của nó chính là id JSON-RPC của request đã mở luồng:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": "listen-tools"
    },
    "notifications": {
      "toolsListChanged": true
    }
  }
}
```

Sau đó gateway chỉ chuyển tiếp những kiểu thay đổi đã được xác nhận. Mọi thông báo trên luồng đó đều mang cùng `io.modelcontextprotocol/subscriptionId` trong `params._meta`. Không có replay tự động và không có tự động listen lại. Khi nối lại, client mở lại subscription và làm mới những danh sách mà nó dựa vào. Một lần đóng nhẹ nhàng do server khởi xướng trả về kết quả complete cuối cùng gắn cùng subscription id đó.

Tuyến hiện đại thay thế `resources/subscribe`, `resources/unsubscribe` và kiểu stream GET độc lập tự phát. Chỉ giữ những thứ đó trong một tuyến cũ được chốt theo version.

### MRTR qua gateway

Khi một backend trả về `resultType: input_required`, gateway chỉ chuyển tiếp kết quả đó nếu client bên ngoài hỗ trợ request nhập liệu cần thiết. Giữ `requestState` nguyên từng byte, trừ khi gateway cố ý kết thúc và phát lại tương tác.

Client thử lại tool công khai ban đầu với một id JSON-RPC mới và `inputResponses`. Gateway cấp quyền lại cho lần thử đó, kiểm tra đúng tuyến công khai, rồi chuyển tiếp một request backend mới. Nó không được giả định rằng một vòng trước đó đã cấp phê duyệt vô hạn.

### Định tuyến phần mở rộng Tasks

Tasks là một phần mở rộng chính thức, định danh bởi `io.modelcontextprotocol/tasks`. Chúng không phải vật thay thế session ở phần lõi.

Client khai báo phần mở rộng trong capability theo từng request, và gateway chỉ công bố nó khi khám phá nếu giữ được vòng đời từ đầu tới cuối. Với một `tools/call` được hỗ trợ, chỉ backend quyết định trả kết quả thường hay `resultType: task`. Một kết quả task mang `taskId`, `status`, các dấu thời gian, `ttlMs` và `pollIntervalMs` tuỳ chọn ngay trong kết quả. Task phải đọc được bền vững trước khi kết quả đó được gửi đi.

Gateway ghi principal đã xác thực và tuyến backend cho định danh task mờ đó. Các lời gọi `tasks/get`, `tasks/update` và `tasks/cancel` sau đó dùng `params.taskId` làm `Mcp-Name`, nhờ vậy các bên trung gian có khoá định tuyến. `tasks/get` trả `resultType: complete` với trạng thái task hiện tại và nhúng kết quả cuối hoặc lỗi protocol vào trạng thái kết thúc. `tasks/update` gửi `inputResponses` có khoá cho phần nhập liệu còn thiếu của task và trả một xác nhận complete rỗng. `tasks/cancel` là một ý định hợp tác với xác nhận complete rỗng, không phải bảo đảm công việc sẽ dừng.

Đừng triển khai các method mới `tasks/list` hay `tasks/result`. Chúng thuộc mô hình thử nghiệm cũ. Một task cần nhập liệu thì phơi ra các request nhúng đầy đủ qua `tasks/get`; client trả lời chúng qua `tasks/update`, không phải bằng cách thử lại lời gọi tool ban đầu. Client vẫn thăm dò theo khoảng thời gian được gợi ý; việc tạo task vẫn do server điều khiển.

Trạng thái tuyến task bền vững là dữ liệu ứng dụng, khoá theo handle của task, không phải một protocol session.

### Ranh giới tương thích

Nếu gateway buộc phải phục vụ một client hay backend cũ:

- Phát hiện thời đại một cách tường minh.
- Giữ initialization, transport session, luồng GET, subscription tài nguyên và từ vựng task cũ bên trong một adapter kiểu cũ.
- Đừng bao giờ để session id kiểu cũ rò rỉ vào định tuyến hay cấp quyền hiện đại.
- Ưu tiên một phép dò discovery có giới hạn và policy dự phòng tường minh hơn là âm thầm hạ cấp.

```figure
t3-gateway-funnel
```

## Tự tay làm

`code/main.py` triển khai một gateway protocol chạy trong tiến trình và hai server backend. Mỗi backend nhận một request mới theo protocol hiện hành. Gateway cung cấp discovery, `tools/list` tất định có lọc theo người dùng, định tuyến theo namespace, Registry `server.json` cùng trạng thái tiếp nhận bên ngoài, descriptor pin, RBAC, rate limit khoá theo principal, quyết định audit và một xác nhận SSE `subscriptions/listen` được mô hình hoá.

Model nhận body request đã phân tích, header định tuyến và một danh tính bearer đã xác thực. Nó không phải một adapter HTTP đầy đủ và không phân tích `Content-Type` hay toàn bộ hợp đồng `Accept`. Hãy nối nó với adapter Streamable HTTP của Bài 09, vốn yêu cầu `Content-Type: application/json` và một giá trị `Accept` chứa cả `application/json` lẫn `text/event-stream`.

Chạy nó:

```bash
cd phases/13-tools-and-protocols/17-mcp-gateways-and-registries
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Bản demo in ra id request bên ngoài và id request backend mới để thấy rõ bước nhảy phi trạng thái.

## Áp dụng

Thay các object backend trong tiến trình bằng client protocol hiện hành thật. Giữ nguyên các đường nối:

- Bản ghi tiếp nhận trước khi kết nối.
- Discovery backend trước khi phơi capability.
- Tên công khai đủ định danh trước khi cấp quyền.
- Descriptor pin trước khi liệt kê hoặc gọi.
- Metadata mới theo từng request trước khi chuyển tiếp.
- Kiểm tra kết quả trước khi trả về.

## Đóng gói

Bài học này đóng gói `outputs/skill-gateway-bootstrap.md`. Nó tạo ra một thiết kế gateway hiện đại bao gồm ingress, discovery, tiếp nhận, namespace, cấp quyền, cache, streaming, subscription, MRTR, Tasks, quan sát được và cô lập phần cũ.

## Bài tập

1. Thêm trace context vào metadata của request bên ngoài và request được chuyển tiếp, rồi ghi mối tương quan đó vào sự kiện audit.
2. Thêm một backend hỗ trợ Tasks và định tuyến `tasks/get` theo task id trong `Mcp-Name`.
3. Thay đổi một descriptor của backend và chứng minh cả discovery lẫn lời gọi trực tiếp đều bị chặn.
4. Thêm một capability server riêng cho từng principal và giải thích vì sao discovery vẫn phải được cache riêng.
5. Viết một interface adapter kiểu cũ mà không thêm bất kỳ trạng thái cũ nào vào class `Gateway` hiện đại.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa thực sự |
|------|---------|
| MCP gateway | Server áp policy và định tuyến giữa client và các MCP server backend |
| Admission record | Bằng chứng và quyết định policy cho phép một backend vào gateway |
| Qualified tool name | Tuyến công khai ổn định, ví dụ `notes.search` |
| Descriptor pin | Digest đã duyệt được kiểm tra khi khám phá và khi điều phối |
| Private cache scope | Kết quả cache bị giới hạn trong một ngữ cảnh cấp quyền |
| Request-scoped SSE | Phản hồi streaming gắn với một request POST |
| `subscriptions/listen` | Luồng SSE do client mở cho các thông báo thay đổi dài hạn đã chọn |
| Task route | Ánh xạ ứng dụng từ một task id mờ tới backend của nó |
| Legacy adapter | Ranh giới tường minh, chốt theo version, cho handshake và session kiểu cũ |

## Đọc thêm

- [Transport Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Khám phá server](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [Yêu cầu server.json của Official Registry](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [Phần mở rộng MCP Tasks](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)

---

[← Bài trước: Uỷ quyền MCP: CIMD, ràng buộc issuer, PKCE và step-up](../../16-mcp-security-oauth-2-1/docs/vi.md) · [Bài sau: Xác thực MCP trong production: đăng ký gắn issuer và token gắn issuer →](../../18-mcp-auth-production/docs/vi.md)
