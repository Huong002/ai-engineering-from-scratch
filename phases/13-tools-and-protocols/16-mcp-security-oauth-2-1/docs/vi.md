# Uỷ quyền MCP: CIMD, ràng buộc issuer, PKCE và step-up

> Một yêu cầu MCP từ xa là phi trạng thái, nhưng uỷ quyền của nó không phải ẩn danh. Hãy gắn mọi thông tin xác thực với issuer đã tạo ra nó và mọi token với tài nguyên sẽ nhận nó.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 13, Bài 09 (kênh truyền), Giai đoạn 13, Bài 15 (bảo mật)
**Time:** ~90 phút

## Mục tiêu học tập

- Khám phá authorization server (máy chủ uỷ quyền) thông qua metadata tài nguyên được bảo vệ.
- Ưu tiên Client ID Metadata Document thay cho Dynamic Client Registration đã bị ngừng khuyến nghị.
- Khai báo đúng `application_type` khi không thể tránh khỏi đường tương thích DCR.
- Kiểm tra `iss` trong phản hồi uỷ quyền và tách biệt thông tin xác thực theo issuer.
- Dùng PKCE (thử thách S256 bảo vệ mã uỷ quyền), resource indicator, kiểm tra audience và scope (phạm vi quyền) tăng dần.
- Gửi yêu cầu MCP 2026-07-28 đã được uỷ quyền mà không cần phiên giao thức.

## Vấn đề

Một máy chủ MCP từ xa có thể đọc bản ghi riêng tư, ghi vào hệ thống bên ngoài hoặc kích hoạt công việc tốn kém. Xác thực cho biết ai đã xuất trình thông tin xác thực. Authorization (uỷ quyền) cũng phải trả lời:

- Thông tin xác thực đó do máy chủ uỷ quyền nào cấp?
- Token dành cho tài nguyên MCP nào?
- Client nào và redirect URI (URI chuyển hướng) nào đã hoàn tất luồng?
- Người dùng đã chấp thuận những thao tác nào?
- Yêu cầu chính xác này có còn nằm trong phạm vi chấp thuận đó không?

Hồ sơ uỷ quyền 2026-07-28 siết chặt việc đăng ký client và cách xử lý issuer. Hồ sơ này ưu tiên Client ID Metadata Document, ngừng khuyến nghị Dynamic Client Registration, yêu cầu đúng `application_type` trên DCR, kiểm tra phản hồi issuer theo RFC 9207 và cấm dùng lại thông tin xác thực giữa các issuer.

Những quy tắc này bổ trợ cho phần lõi phi trạng thái. Chúng không khôi phục bắt tay ở lõi hay `Mcp-Session-Id`.

## Khái niệm

### Nắm ba vai trò

- **MCP client:** gửi yêu cầu thay mặt chủ sở hữu tài nguyên.
- **MCP resource server (máy chủ tài nguyên):** nhận access token (mã truy cập) và phục vụ endpoint MCP.
- **Authorization server:** xác thực chủ sở hữu tài nguyên, thu thập sự chấp thuận và cấp token.

Máy chủ tài nguyên và máy chủ uỷ quyền có thể do cùng một bên vận hành, nhưng phải tách biệt định danh và trách nhiệm kiểm tra của chúng.

### Uỷ quyền chỉ áp dụng cho HTTP

Đặc tả uỷ quyền MCP áp dụng cho các kênh truyền dựa trên HTTP. Máy chủ stdio cục bộ chạy trong ranh giới tin cậy của tiến trình và hệ điều hành. Đừng thêm luồng OAuth giả lập trình duyệt vào stdio chỉ để cho đối xứng.

Với Streamable HTTP từ xa, hãy gửi bearer token trong header `Authorization` ở mọi yêu cầu. Không bao giờ đặt nó trong URL.

### Bắt đầu từ metadata tài nguyên được bảo vệ

Máy chủ tài nguyên công bố metadata theo RFC 9728:

```json
{
  "resource": "https://notes.example.com/mcp",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:delete", "notes:read", "notes:write"]
}
```

Client bắt đầu từ URL tài nguyên MCP, tải tài liệu này, chọn một máy chủ uỷ quyền được công bố, rồi tải metadata OAuth hoặc OpenID Connect của máy chủ đó.

Giữ nguyên đường dẫn tài nguyên khi dựng URL well-known theo RFC 9728. Với tài nguyên `https://notes.example.com/mcp`, bài này dùng `https://notes.example.com/.well-known/oauth-protected-resource/mcp`. Bỏ hậu tố `/mcp` có thể chọn nhầm metadata của một tài nguyên được bảo vệ khác trên cùng origin.

Đừng đoán máy chủ uỷ quyền từ hostname. Đừng đi theo issuer phát hiện được từ một body lỗi chưa được kiểm tra. Hãy giữ một chính sách về những issuer mà client sẵn sàng tin cậy.

### Kiểm tra metadata máy chủ uỷ quyền

Metadata nên công bố các endpoint và cơ chế kiểm soát được hỗ trợ:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "code_challenge_methods_supported": ["S256"],
  "authorization_response_iss_parameter_supported": true,
  "client_id_metadata_document_supported": true
}
```

Yêu cầu S256 cho PKCE. Ghi lại chính xác chuỗi issuer. Giá trị chính xác đó trở thành khoá cho việc đăng ký và lưu token.

### Tuân theo thứ tự ưu tiên đăng ký

Dùng thông tin client đã đăng ký trước nếu client đã có quan hệ rõ ràng với issuer đã chọn. Nếu không, ưu tiên Client ID Metadata Document khi máy chủ uỷ quyền công bố hỗ trợ. Chỉ dùng DCR như đường tương thích đã bị ngừng khuyến nghị, và nếu không cơ chế nào trong số đó khả dụng thì hỏi người dùng nhập thông tin client.

### Ưu tiên Client ID Metadata Document

Client ID Metadata Document cung cấp cho máy chủ uỷ quyền một URL HTTPS vừa là định danh client vừa là vị trí metadata của nó:

```json
{
  "client_id": "https://client.example.com/oauth/metadata.json",
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

Máy chủ uỷ quyền tải và kiểm tra tài liệu. `client_id` phải là URL HTTPS có đường dẫn, và giá trị bên trong tài liệu phải bằng chính xác URL đó. Các trường bắt buộc của tài liệu là `client_id`, `client_name` và `redirect_uris`. `application_type` xuất hiện trong ví dụ này nhưng không phải yêu cầu của CIMD. Việc bắt buộc mới của nó chỉ dành riêng cho đường DCR.

Coi việc tải tài liệu là thao tác nhạy cảm với SSRF. Phân giải và kiểm tra đích, từ chối địa chỉ loopback, private, link-local và mọi địa chỉ bị cấm khác, kiểm tra lại sau khi chuyển hướng và khi DNS thay đổi, giới hạn số lần chuyển hướng, số byte và thời gian, yêu cầu JSON, và chỉ cache theo các chỉ thị cache HTTP đã được kiểm tra. Coi `client_name` và các trường hiển thị khác là văn bản không đáng tin.

CIMD loại bỏ nhu cầu tạo định danh động mới cho mỗi lần liên hệ đầu tiên. Nó không loại bỏ việc kiểm tra redirect URI, chính sách issuer hay sự chấp thuận của người dùng.

### DCR là đường tương thích

Dynamic Client Registration vẫn khả dụng cho các máy chủ uỷ quyền cũ hơn, nhưng đã bị ngừng khuyến nghị cho các triển khai MCP mới.

Khi dùng DCR, hãy khai báo `application_type`:

```json
{
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

- Client desktop, mobile, dòng lệnh và loopback dùng `native`.
- Ứng dụng trình duyệt được lưu trữ từ xa dùng `web` và redirect HTTPS từ xa.

Bỏ trống trường này có thể mặc định thành `web` trong một triển khai đăng ký OpenID Connect và khiến redirect loopback hợp lệ bị lỗi.

Giữ mã DCR sau một quyết định dự phòng rõ ràng. Đừng âm thầm chuyển sang dự phòng sau một lỗi kiểm tra CIMD bất kỳ. Điều đó có thể biến một lỗi bảo mật thành đường đăng ký yếu hơn.

### Gắn thông tin xác thực với issuer

Lưu vật liệu đăng ký do issuer cấp dưới đúng issuer đó:

```text
issuer_credentials[issuer] = pre_registered_or_dcr_client
tokens[(issuer, resource)] = access_token
```

Nếu khám phá tài nguyên được bảo vệ thay đổi từ `https://auth-one.example` sang `https://auth-two.example`, hãy đánh giá lại mức tin cậy. Không bao giờ gửi client secret, client id DCR, registration access token, refresh token hay access token của issuer thứ nhất sang issuer thứ hai. Client đã đăng ký trước và client DCR phải dùng thông tin xác thực được cấp cho issuer mới.

Client id của CIMD thì khác, vì nó là URL HTTPS tự lưu trữ chứ không phải thông tin xác thực do máy chủ uỷ quyền cấp. Cùng một URL CIMD có thể dùng lại được: một issuer tin cậy mới sẽ tải và kiểm tra tài liệu mà không cần đăng ký lại DCR. Phản hồi uỷ quyền và token vẫn được kiểm tra và lưu dưới issuer mới.

### Authorization code với PKCE

Luồng tương tác là:

1. Sinh một `code_verifier` có entropy cao.
2. Suy ra `code_challenge` S256.
3. Gửi yêu cầu uỷ quyền với đúng `client_id`, `redirect_uri`, `scope`, `code_challenge` và `resource`.
4. Nhận phản hồi uỷ quyền chứa `code` và, nếu có, `iss`.
5. Kiểm tra `iss` với issuer đã ghi lại trước khi dùng bất kỳ trường nào của phản hồi.
6. Đổi mã với `code_verifier`, cùng redirect URI và cùng `resource`.
7. Lưu token nhận được dưới `(issuer, resource)`.

Tham số `resource` từ RFC 8707 xuất hiện trong cả yêu cầu uỷ quyền và yêu cầu token. Nó xác định URI chuẩn tắc của máy chủ MCP.

### Kiểm tra `iss` chính xác

RFC 9207 ngăn phản hồi uỷ quyền từ một issuer bị nhầm lẫn với phản hồi từ issuer khác.

Khi `iss` có mặt, hãy so sánh với issuer đã ghi lại mà không đổi hoa thường, không thêm bớt dấu gạch chéo cuối, không bỏ cổng mặc định, không chuẩn hoá percent-encoding. Khi không khớp, đừng dùng `code` và thậm chí đừng hiển thị chi tiết lỗi do kẻ tấn công kiểm soát từ phản hồi đó.

Máy chủ uỷ quyền có kèm `iss` sẽ công bố `authorization_response_iss_parameter_supported: true`. Client hiện tại vẫn kiểm tra `iss` khi nó có mặt kể cả khi thiếu công bố đó.

### Kiểm tra audience tại máy chủ MCP

Máy chủ tài nguyên chỉ nhận token được cấp cho chính nó:

```text
token.issuer == configured_authorization_server
token.audience == canonical_mcp_resource
```

Token không hợp lệ, hết hạn, sai issuer hoặc sai audience nhận 401. Máy chủ MCP không được nhận hay chuyển tiếp token dành cho dịch vụ khác.

### Yêu cầu scope nhỏ nhất cần cho hiện tại

Bắt đầu với scope cần ngay lúc này. Nếu một công cụ sau đó cần nhiều hơn, máy chủ trả 403 kèm thử thách scope có thẩm quyền:

```text
WWW-Authenticate: Bearer error="insufficient_scope",
  scope="notes:delete",
  resource_metadata="https://notes.example.com/.well-known/oauth-protected-resource/mcp"
```

Client giải thích quyền mới, lấy sự chấp thuận, thực hiện luồng uỷ quyền mới với tập scope kết hợp, rồi thử lại yêu cầu MCP với một JSON-RPC id mới.

Đừng cho rằng scope trong thử thách là tập con của `scopes_supported`. Thử thách có thẩm quyền cho thao tác hiện tại.

### Uỷ quyền và đường truyền MCP phi trạng thái

Một lời gọi công cụ đã được uỷ quyền vẫn mang đầy đủ phong bì yêu cầu hiện tại:

```text
POST /mcp
Authorization: Bearer <access-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.delete
```

```json
{
  "jsonrpc": "2.0",
  "id": 12,
  "method": "tools/call",
  "params": {
    "name": "notes.delete",
    "arguments": {"id": "note-7"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "oauth-lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Token uỷ quyền cho chủ thể. Metadata yêu cầu thoả thuận hành vi giao thức. Không cái nào thay thế cái kia.

Kiểm tra đường truyền theo thứ tự cố định: kiểu JSON-RPC và metadata, sự khớp giữa header và body, rồi mới đến hỗ trợ giao thức. Sai lệch định tuyến hoặc header phiên bản trả HTTP 400 với `-32020`. Nếu header và body khớp nhau về một phiên bản không được hỗ trợ, trả HTTP 400 với `-32022` và `data` đúng bằng `{"supported":["2026-07-28"],"requested":"<actual>"}`. Phương thức không xác định trả HTTP 404 với `-32601`.

Mọi lỗi yêu cầu, kể cả 401 token không hợp lệ và 403 thiếu scope, đều là phong bì lỗi JSON-RPC với `id` gốc của yêu cầu. Thông tin phục hồi có cấu trúc thuộc về `data` tuỳ chọn của lỗi; `WWW-Authenticate` vẫn là header phản hồi HTTP. Notification không có `id`, nên không nhận body JSON-RPC. Notification HTTP được chấp nhận trả 202 với body rỗng.

Máy chủ triển khai `server/discover` và có công bố công cụ, nên nó cũng triển khai phương thức bắt buộc `tools/list`. Các descriptor công cụ của nó có tên, mô tả và `inputSchema` gốc kiểu object ổn định. Danh sách mang tính xác định và trả về `resultType`, metadata định danh máy chủ, `ttlMs` có giới hạn và `cacheScope`. Phần khám phá và danh sách công cụ không phụ thuộc người dùng có thể khả dụng trước khi uỷ quyền. Áp dụng chính sách thông thường và cache riêng nếu một trong hai thay đổi theo chủ thể.

### Không chuyển tiếp token

Máy chủ MCP không được chuyển tiếp access token MCP của client sang API hạ nguồn. Hãy lấy một token hạ nguồn riêng với audience đúng hoặc dùng thiết kế trao đổi token rõ ràng. Kiểm tra audience chỉ hiệu quả khi các dịch vụ từ chối token được cấp cho bên khác.

### Refresh token

Refresh token là tuỳ chọn. Khi được cấp, hãy lưu chúng bí mật và đặt khoá theo issuer và tài nguyên. Đừng cho rằng chúng luôn tồn tại. Hãy xoay vòng khi máy chủ uỷ quyền hỗ trợ và phát hiện việc dùng lại các giá trị đã bị vô hiệu.

```figure
t3-scope-stepup
```

## Tự tay làm

`code/main.py` là bộ mô phỏng giao thức và uỷ quyền chạy trong tiến trình. Nó triển khai khám phá tài nguyên được bảo vệ, metadata máy chủ uỷ quyền, đăng ký CIMD, dự phòng DCR có kiểm soát phiên bản, kiểm tra kiểu ứng dụng, PKCE, kiểm tra issuer, token gắn với tài nguyên, step-up scope, `server/discover`, `tools/list` và một yêu cầu công cụ phi trạng thái.

Mô hình nhận body yêu cầu đã được phân tích và các header định tuyến. Nó không phải adapter HTTP hoàn chỉnh và không phân tích `Content-Type` hay `Accept`. Hãy nối nó với adapter Streamable HTTP của Bài 09, adapter này yêu cầu `Content-Type: application/json` và giá trị `Accept` chứa cả `application/json` và `text/event-stream`.

Chạy nó:

```bash
cd phases/13-tools-and-protocols/16-mcp-security-oauth-2-1
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Kết quả hiển thị phần khám phá trước, đăng ký CIMD, một lần đọc thông thường, hai lần step-up scope riêng biệt và việc lưu thông tin xác thực theo khoá issuer.

## Áp dụng

Ánh xạ các đối tượng mô phỏng sang thành phần production:

- `ResourceServer.protected_resource_metadata` trở thành endpoint RFC 9728.
- `AuthorizationServer.metadata` trở thành RFC 8414 hoặc khám phá OpenID Connect.
- `Client.enroll` trở thành phân giải CIMD cộng với một nhánh tương thích DCR rõ ràng.
- Thông tin xác thực client do issuer cấp và `tokens_by_issuer_resource` trở thành các bản ghi được mã hoá. URL CIMD có thể vẫn dùng lại được trong khi kết quả uỷ quyền của nó vẫn gắn với issuer.
- `ResourceServer.handle` trở thành middleware kiểm tra header MCP hiện tại, token và scope của công cụ trước khi dispatch, đồng thời giữ mọi lỗi yêu cầu trong phong bì JSON-RPC tương ứng.

## Đóng gói

Bài này đóng gói `outputs/skill-oauth-scope-planner.md`. Skill này giờ thiết kế thứ tự ưu tiên đăng ký, lưu thông tin xác thực gắn với issuer, kiểu ứng dụng, PKCE, resource indicator, thử thách scope và ranh giới yêu cầu phi trạng thái hiện tại.

## Bài tập

1. Thêm xoay vòng refresh token và từ chối việc dùng lại refresh token trước đó.
2. Thêm danh sách issuer được phép. Khi issuer thay đổi, chỉ dùng lại URL CIMD vốn dùng lại được; từ chối mọi thông tin xác thực và token do issuer trước cấp.
3. Thêm thời hạn cho authorization code và xác nhận việc đổi mã muộn sẽ thất bại.
4. Xây dựng một biến thể client web với redirect HTTPS từ xa và so sánh metadata DCR của nó với client native.
5. Thêm một tài nguyên thứ hai dưới cùng issuer. Xác nhận access token của nó không dùng được ở tài nguyên thứ nhất.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa thực sự |
|------|---------|
| Protected-resource metadata | Tài liệu RFC 9728 xác định tài nguyên và các máy chủ uỷ quyền |
| CIMD | Tài liệu metadata HTTPS mà URL của nó là định danh client OAuth |
| DCR | Đăng ký client động đã bị ngừng khuyến nghị, được giữ lại để tương thích |
| `application_type` | `native` hoặc `web`, dùng để kiểm tra quy tắc redirect URI |
| PKCE | Verifier và thử thách S256 bảo vệ authorization code bị chặn giữa đường |
| `iss` | Định danh issuer trong phản hồi uỷ quyền theo RFC 9207 |
| Resource indicator | Tham số RFC 8707 gắn yêu cầu token với một tài nguyên MCP |
| Audience | Tài nguyên mà token có hiệu lực |
| Step-up | Chấp thuận mới và cấp token mới cho một scope bổ sung của thao tác hiện tại |
| Issuer-bound credentials | Bản ghi đăng ký và token được tách biệt theo đúng issuer của máy chủ uỷ quyền |

## Đọc thêm

- [Đặc tả uỷ quyền MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)
- [RFC 9728: OAuth 2.0 Protected Resource Metadata](https://www.rfc-editor.org/rfc/rfc9728)
- [RFC 8707: Resource Indicators for OAuth 2.0](https://www.rfc-editor.org/rfc/rfc8707)
- [RFC 9207: OAuth 2.0 Authorization Server Issuer Identification](https://www.rfc-editor.org/rfc/rfc9207)
- [OAuth Client ID Metadata Document draft](https://datatracker.ietf.org/doc/draft-ietf-oauth-client-id-metadata-document/)

---

[← Bài trước: Bảo mật MCP: Metadata bị đầu độc, định tuyến và trạng thái MRTR](../../15-mcp-security-tool-poisoning/docs/vi.md) · [Bài sau: Gateway MCP phi trạng thái và kiểm soát tiếp nhận tại registry (sổ đăng ký) →](../../17-mcp-gateways-and-registries/docs/vi.md)
