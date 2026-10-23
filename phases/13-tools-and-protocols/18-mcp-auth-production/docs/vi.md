# Xác thực MCP trong production: đăng ký gắn issuer và token gắn issuer

> Bài 16 đã dựng máy trạng thái (state machine) OAuth 2.1. Bài này gia cố các ranh giới production của nó cho MCP 2026-07-28: ưu tiên Client ID Metadata Documents, dynamic registration đã bị khai tử và chỉ còn để tương thích, kiểm tra issuer (nhà phát hành token) trong authorization response, thông tin đăng nhập client khoá theo issuer, làm mới JWKS, và token ghim audience trên mọi request không trạng thái.
>
> **Ghi chú spec (2026-07-28):** Dynamic Client Registration đã bị khai tử để nhường chỗ cho Client ID Metadata Documents. DCR vẫn là một cơ chế tương thích. Khi dùng nó, client khai báo đúng `application_type`. Client kiểm tra giá trị `iss` RFC 9207 nếu có và không bao giờ dùng lại thông tin đăng nhập giữa các issuer của authorization server.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 13, Bài 16 (máy trạng thái OAuth 2.1), Giai đoạn 13, Bài 17 (gateway)
**Time:** ~90 phút

## Mục tiêu học tập

- Khám phá một authorization server qua metadata RFC 8414 và kiểm tra hợp đồng.
- Đăng ký qua Client ID Metadata Document và cô lập DCR đã bị khai tử như một đường dự phòng.
- Kiểm tra `iss` RFC 9207, khoá các đăng ký theo issuer của authorization server, và khoá token theo issuer cộng resource.
- Cache và làm mới khoá JWKS theo lịch để việc kiểm tra chữ ký sống sót qua các lần xoay khoá.
- Ghim token vào một MCP resource duy nhất bằng resource indicator RFC 8707 và từ chối kiểu dùng lại của confused deputy.
- Chọn kiểm tra JWT hay token introspection (thẩm vấn token), định nghĩa độ tươi của revocation (thu hồi), và thất bại an toàn khi các phụ thuộc danh tính không sẵn sàng.
- Tách authorization server, resource server và client để mỗi bên chỉ thực thi các kiểm tra của chính nó.
- Kiểm toán một authorization server theo checklist triển khai và từ chối việc đăng ký hay dùng lại token không an toàn.

## Vấn đề

Trình mô phỏng của Bài 16 chạy OAuth 2.1 trong bộ nhớ. Production có ba lỗ hổng vận hành mà một trình mô phỏng chỉ dùng bộ nhớ không nhìn thấy.

Lỗ hổng thứ nhất là đăng ký và cô lập thông tin đăng nhập. Một tổ chức thực tế có thể chạy hàng trăm MCP server và hàng nghìn MCP client. Bản sửa đổi 2026-07-28 ưu tiên **Client ID Metadata Document**: client dùng một URL HTTPS với đường dẫn do nó kiểm soát làm định danh, và authorization server tự kéo metadata về. Đăng ký động RFC 7591 chỉ còn là một đường tương thích đã bị khai tử. Khi buộc phải dùng DCR, request khai báo đúng `application_type`. Client lưu các đăng ký theo issuer của authorization server và lưu access token theo cặp `(issuer, resource)`. Issuer thay đổi nghĩa là phải đăng ký lại, và resource khác nghĩa là một token gắn audience riêng.

Lỗ hổng thứ hai là key rotation (xoay vòng khoá). Việc kiểm tra JWT phụ thuộc vào các khoá ký của authorization server, được công bố dưới dạng JSON Web Key Set (JWKS). Authorization server xoay các khoá này theo lịch (thường mỗi giờ, có khi nhanh hơn khi xử lý sự cố). Một MCP server chỉ tải JWKS một lần lúc khởi động sẽ kiểm tra tốt cho tới khi hết cửa sổ xoay khoá — rồi mọi request đều thất bại cho tới lần khởi động lại. Production nối JWKS như một giá trị được cache với một job làm mới ghi đè cache trước khi các khoá cũ hết hạn, cộng thêm một lần tải dự phòng khi cache trượt, cho trường hợp một token được ký bằng khoá mới hơn cache xuất hiện.

Lỗ hổng thứ ba là audience binding (ràng buộc đối tượng thụ hưởng). Bài 16 đã giới thiệu resource indicator RFC 8707. Trong production, chỉ báo đó trở thành một kiểm tra claim cứng trên mọi request. MCP server so `token.aud` với URL resource chuẩn tắc của chính nó và từ chối khi không khớp bằng HTTP 401. Đây là hàng phòng thủ duy nhất chống lại việc một MCP server thượng nguồn (hoặc một client độc hại đang giữ token dành cho một server) phát lại token đó với một server khác trong cùng lưới tin cậy.

Bài này ánh xạ mỗi lỗ hổng vào một phần cụ thể của bề mặt. Tài liệu metadata là một endpoint HTTP. Việc làm mới cache JWKS là một job theo lịch cộng một cache khoá-giá trị. Kiểm tra JWT là một thủ tục mà resource server chạy trước khi điều phối bất kỳ tool nào. Giữ ba vai trò tách rời và mỗi vai trò chỉ thực thi các kiểm tra thuộc về nó: authorization server phát hành và xoay khoá, resource server cache và kiểm tra, client khám phá và đăng ký.

## Phạm vi: thực thi trong production sau Bài 16

[Bài 16: Bảo mật MCP với OAuth 2.1](../../16-mcp-security-oauth-2-1/docs/en.md) sở hữu máy trạng thái authorization code, PKCE, khám phá protected resource, resource indicator và các quyết định scope. Bài này không định nghĩa một luồng OAuth thứ hai. Nó bắt đầu sau khi các hợp đồng đó đã tồn tại và hỏi cách một resource server đã triển khai tiếp tục thực thi chúng trong lúc xoay khoá, kiểm tra token opaque, thu hồi, phụ thuộc hỏng, rollout và xử lý sự cố.

Ranh giới production hẹp hơn và nặng tính vận hành hơn:

- Đường JWT kiểm tra issuer, thuật toán, khoá chữ ký, audience, các claim thời gian và scope được ghim trên mọi request, đồng thời làm mới JWKS một cách an toàn.
- Đường token opaque gọi endpoint introspection có xác thực của issuer và kiểm tra trạng thái active, audience hay resource, thời hạn, chủ thể và scope trả về.
- Chính sách thu hồi định nghĩa một thông tin đăng nhập phải ngừng hoạt động trong bao lâu và cache nào có thể làm chậm sự thật đó.
- Chính sách thất bại quyết định điều gì xảy ra khi hạ tầng khám phá, JWKS, introspection hoặc thu hồi không sẵn sàng.
- Bằng chứng ghi lại metadata issuer nào, bộ khoá hay phản hồi introspection nào, các claim nào của token, phiên bản chính sách nào và lý do từ chối nào dẫn tới kết quả, mà không lưu token.

Sự phân biệt này giữ cho các bài học ghép được với nhau. Bài 16 chứng minh luồng. Bài 18 chứng minh một token vẫn đáng tin, hoặc bị từ chối, sau khi nó đi vào một đường request MCP thật.

## Khái niệm

### RFC 8414 — Metadata của OAuth Authorization Server

Một tài liệu tại `/.well-known/oauth-authorization-server` mô tả mọi thứ client cần:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "client_id_metadata_document_supported": true,
  "registration_endpoint": "https://auth.example.com/register",
  "authorization_response_iss_parameter_supported": true,
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

Client được cho một URL MCP resource sẽ nối chuỗi khám phá: `oauth-protected-resource` từ RFC 9728 (tài liệu của resource server) nêu tên issuer, rồi `oauth-authorization-server` (RFC này) nêu tên mọi endpoint. Client không bao giờ hard-code một URL authorization.

Với một định danh resource có đường dẫn, chèn đoạn well-known trước đường dẫn đó. Ví dụ, `https://mcp.example.com/team/server` phân giải metadata protected resource tại `https://mcp.example.com/.well-known/oauth-protected-resource/team/server`. Việc nối thêm `/.well-known/...` sau đường dẫn resource là sai.

Hợp đồng bạn phải kiểm tra trước khi tin một IdP cho MCP:

- `code_challenge_methods_supported` có chứa `S256` (PKCE theo RFC 7636). Spec nói rõ: nếu trường này **vắng mặt**, authorization server không hỗ trợ PKCE và client **MUST** từ chối đi tiếp.
- `grant_types_supported` có chứa `authorization_code` và từ chối `password` cùng `implicit`.
- Có ít nhất một đường đăng ký: `client_id_metadata_document_supported: true` (CIMD, ưu tiên), một client đã đăng ký trước, hoặc `registration_endpoint` (tương thích RFC 7591 đã bị khai tử).
- Nếu `authorization_response_iss_parameter_supported` là true, client bắt buộc có `iss` RFC 9207 trả về và so khớp chính xác với issuer đã ghi lại trước khi chuyển hướng.
- `response_types_supported` đúng bằng `["code"]` với OAuth 2.1.

Nếu thiếu `S256`, MCP server từ chối triển khai với IdP này — không có chế độ suy giảm cho PKCE. Nếu *không* đường đăng ký nào được quảng bá và bạn không có `client_id` đăng ký trước, bạn cũng không thể đăng ký; manifest triển khai mới là thứ sai, không phải code.

### RFC 9728 (nhắc lại) — Metadata của Protected Resource

Bài 16 đã đề cập RFC 9728. Khác biệt trong production: tài liệu này là nơi duy nhất client nhìn vào để tìm các authorization server mà *MCP server này* tin. Một MCP server có thể chấp nhận token từ nhiều IdP (một cho nhân viên, một cho đối tác). RFC 9728 khai báo tập đó; RFC 8414 ghi lại mỗi IdP hỗ trợ những gì.

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Client ID Metadata Documents (mặc định được khuyến nghị)

CIMD đảo ngược việc đăng ký từ *đẩy* sang *kéo*. Thay vì xin authorization server cấp một `client_id`, client dùng một URL HTTPS do mình kiểm soát **làm** `client_id` của nó. URL đó phân giải thành một tài liệu metadata JSON; authorization server tự tải nó trong luồng OAuth. Niềm tin bắt rễ từ DNS: nếu người vận hành server tin `app.example.com`, họ tin client phục vụ từ `https://app.example.com/client.json`. Không có vòng đăng ký, không có không gian tên `client_id` để cạn kiệt, không có trạng thái theo từng server phải đồng bộ.

Tài liệu metadata mà client tự lưu trữ:

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

Giá trị `client_id` trong tài liệu **MUST** bằng đúng URL phục vụ nó (authorization server kiểm tra điều này; lệch là bị từ chối). Authorization server quảng bá hỗ trợ bằng `client_id_metadata_document_supported: true` trong metadata RFC 8414 của nó.

Với hợp đồng CIMD hiện tại, `client_id`, `client_name` và một mảng `redirect_uris` không rỗng là bắt buộc. Định danh client là một URL HTTPS tuyệt đối có đường dẫn. `application_type` có thể được đưa vào, nhưng nó không phải trường bắt buộc của CIMD. Đừng bê yêu cầu `application_type` của DCR vào đường CIMD ưu tiên.

Hai sự thật bảo mật mà spec nói thẳng:

- **SSRF.** Authorization server tải một URL do kẻ tấn công cung cấp. Nó phải phòng thủ chống giả mạo request phía server (không tải tới các endpoint nội bộ hay quản trị).
- **Mạo danh localhost.** Chỉ CIMD không thể ngăn một kẻ tấn công cục bộ chiếm URL metadata của một client hợp pháp và gắn bất kỳ redirect `localhost` nào. Authorization server **MUST** hiển thị rõ hostname của redirect URI trong bước đồng ý và **SHOULD** cảnh báo với redirect chỉ dùng `localhost`.

Vì CIMD không cần trạng thái phía server, không có cơ quan đăng ký nào phải dựng lên như DCR đòi hỏi. Phía client chỉ đọc: phục vụ tài liệu metadata từ một endpoint HTTPS tĩnh và để authorization server tự kéo về.

Nếu người vận hành authorization server đã cấp sẵn một định danh client, hãy dùng đăng ký theo phạm vi issuer đó trước khi thử đăng ký tự động. Nếu không, ưu tiên CIMD. Chỉ dùng DCR đã bị khai tử khi issuer không thể dùng cả đăng ký trước lẫn CIMD.

### RFC 7591: đăng ký tương thích đã bị khai tử

DCR đã bị khai tử trong bản sửa đổi 2026-07-28. Chỉ giữ nó cho các authorization server không dùng được CIMD và khi đăng ký trước là không thực tế. Một client tương thích gửi:

```json
POST /register
Content-Type: application/json

{
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

Server trả về `client_id` và một `registration_access_token` cho các lần cập nhật sau:

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`application_type` không phải để trang trí. Một client desktop loopback khai `native`; một client do server lưu trữ khai `web` và dùng redirect URI HTTPS. `token_endpoint_auth_method: none` là mặc định đúng cho một public native client. Nó chỉ nhận `client_id`, với PKCE đóng vai trò bằng chứng sở hữu.

Ba cái bẫy trong production:

- Endpoint đăng ký phải giới hạn tốc độ theo IP nguồn. Không có nó, một kẻ đối địch viết script tạo hàng triệu đăng ký giả và làm cạn không gian tên `client_id`. Chạy kiểm tra giới hạn tốc độ trước khi cơ quan đăng ký xử lý request.
- `software_statement` (một JWT đã ký bảo lãnh cho client) là bắt buộc với một số IdP doanh nghiệp. Bản mock của bài bỏ qua nó; production nối một bước kiểm tra để từ chối các đăng ký không ký từ bất kỳ nguồn nào ngoài redirect URI localhost.
- `registration_access_token` phải được lưu dưới dạng hash, không phải văn bản thuần. Đánh cắp token này nghĩa là kẻ tấn công viết lại được redirect URI của client.

### RFC 8707 (nhắc lại) — Resource Indicator

Bài 16 đã dựng hình dạng. Quy tắc production: mọi request token đều kèm `resource=<canonical-mcp-url>`, và MCP server kiểm tra `token.aud` khớp URL resource của chính nó trên mọi lệnh gọi. URI chuẩn tắc là định danh *cụ thể nhất* cho server: dùng scheme và host chữ thường, không fragment, và theo quy ước không có dấu gạch chéo cuối. Phần path **không** bị cắt theo quy tắc — spec giữ nó khi cần để phân biệt một MCP server riêng lẻ. `https://mcp.example.com`, `https://mcp.example.com/mcp`, `https://mcp.example.com:8443`, và `https://mcp.example.com/server/mcp` đều là URI chuẩn tắc hợp lệ. Chọn một cho mỗi server và ghim `aud` đúng vào đó. (Bản mock của bài dùng audience chỉ có host như `https://notes.example.com` cho gọn; một triển khai đặt nhiều MCP server dưới cùng một origin sẽ phân biệt chúng bằng path.)

### RFC 7636 (nhắc lại) — PKCE

PKCE là bắt buộc trong OAuth 2.1. Luồng authorization code của bài luôn mang `code_challenge` và `code_verifier`. Server từ chối mọi request token không có verifier hoặc có verifier không hash ra đúng challenge đã lưu.

### Hồ sơ authorization của MCP 2026-07-28

Bản MCP hiện tại giữ ranh giới OAuth resource server trong khi làm transport MCP trở nên không trạng thái. Không có phiên giao thức nào để cache một quyết định danh tính. Vì vậy lớp authorization kiểm tra từng request một cách độc lập:

- Triển khai metadata protected resource RFC 9728, và cung cấp vị trí của nó hoặc qua header `WWW-Authenticate: Bearer resource_metadata="..."` trên một 401 **hoặc** qua URI well-known `/.well-known/oauth-protected-resource` (SEP-985 làm header thành tuỳ chọn với dự phòng well-known). Trường `authorization_servers` trong metadata **MUST** nêu tên ít nhất một server.
- Chỉ chấp nhận token qua `Authorization: Bearer ...` trên **mọi** request — không bao giờ trong query string, không bao giờ chỉ kiểm tra lúc bắt đầu phiên.
- Kiểm tra `aud`, `iss`, `exp` và các scope bắt buộc theo từng request. Server **MUST** kiểm tra rằng token được phát hành riêng cho nó (audience); `aud` thiếu hoặc không khớp bị từ chối, không bao giờ được coi là wildcard.
- Khi 401/403, trả `WWW-Authenticate: Bearer` mang `error=...`, tham số `resource_metadata="<PRM-URL>"` (URL của tài liệu metadata, *không* phải resource trần), và `scope="..."` khi `insufficient_scope` (403). Lưu ý: tham số là `resource_metadata`, một con trỏ khám phá — không có tham số `resource` trong challenge.
- Khám phá authorization server chấp nhận **hoặc** metadata OAuth RFC 8414 **hoặc** OpenID Connect Discovery 1.0; client phải thử cả hai hậu tố well-known theo thứ tự ưu tiên.
- Client (không phải server) phòng thủ chống **tấn công mix-up**: nó ghi lại `issuer` mong đợi trước khi chuyển hướng và kiểm tra giá trị `iss` trả về trong authorization response thực tế (RFC 9207) trước khi đổi code. Chỉ PKCE không ngăn được mix-up, vì client trao `code_verifier` của mình cho bất kỳ token endpoint nào mà nó bị lái tới.
- Thông tin đăng nhập của client thuộc về một issuer của authorization server. Nếu khám phá phân giải ra một issuer khác, client đăng ký lại thay vì trình `client_id`, registration token hay access token cũ.
- CIMD là cơ chế đăng ký ưu tiên. DCR đã bị khai tử; một request DCR tương thích vẫn phải khai báo đúng `application_type`.

Bản nháp OAuth 2.1 là nền; RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD là bề mặt; spec MCP là hồ sơ.

### Checklist năng lực triển khai

Bảng tính năng của nhà cung cấp nhanh chóng trở nên lỗi thời. Thay vào đó hãy kiểm tra metadata do chính authorization server mà bạn sẽ triển khai trả về. Cổng kiểm tra này mang tính máy móc:

| Kiểm tra | Quyết định bắt buộc |
|---|---|
| Issuer khám phá được | Đúng issuer HTTPS mà chính sách yêu cầu |
| PKCE | Quảng bá `S256`; nếu không thì dừng |
| Đăng ký | Ưu tiên CIMD, chấp nhận đăng ký trước, DCR chỉ là tương thích đã bị khai tử |
| Authorization response | Kiểm tra `iss` RFC 9207 khi có hoặc khi được quảng bá |
| Ràng buộc resource | Request token mang `resource`; resource server đòi `aud` khớp |
| Lưu thông tin đăng nhập | Khoá client ID và thông tin đăng ký theo issuer; khoá access token theo issuer cộng resource |
| Tương thích DCR | Khai `native` hoặc `web`; từ chối redirect URI không khớp loại ứng dụng đã khai |

Đừng suy ra hỗ trợ từ tên sản phẩm hay hạng giá. Hãy chụp lại tài liệu khám phá được vào bằng chứng triển khai và thất bại đóng khi một trường bắt buộc vắng mặt.

### Mẫu làm mới JWKS (xoay ở AS, làm mới ở resource server)

Giữ hai động từ tách rời, vì gộp chúng lại là một lỗi production thật:

- **Xoay (rotate)** là việc *authorization server* làm: tạo một khoá ký mới, công bố nó trong JWKS, rồi gỡ khoá cũ sau đó. Resource server không tham gia và không làm được — nó không giữ khoá riêng của IdP.
- **Làm mới (refresh)** là việc *resource server* làm: `GET` lại JWKS đã công bố vào cache của nó. Đó là hành động JWKS duy nhất mà một resource server từng làm.

Chế độ hỏng trong production là cache cũ. Giải quyết bằng một job làm mới theo lịch cộng một cache khoá-giá trị. Resource server chạy một job (cron, timer, bất cứ thứ gì runtime cung cấp) mà theo chu kỳ cố định tải `<issuer>/.well-known/jwks.json` và ghi đè `cache[issuer] = {keys, fetched_at}`. Validator đọc từ cache đó. Một token có `kid` không có trong cache sẽ kích hoạt **một** lần làm mới đồng bộ như đường dự phòng, rồi kiểm tra lại. Cách này xử lý hai trường hợp cùng lúc: lần làm mới theo lịch, và các cửa sổ chồng khoá khi một token ký bằng khoá hoàn toàn mới tới trước lần làm mới theo lịch kế tiếp.

Đường dự phòng **phải là tải lại, không bao giờ là xoay**. Nếu bạn nối đường cache-miss với xoay-và-tạo, hai thứ sẽ hỏng: (1) tạo một khoá mới sinh ra `kid` vẫn *không* khớp token, nên tra cứu vẫn thất bại; và (2) kẻ tấn công phun token với `kid` ngẫu nhiên sẽ ép ra một chuỗi tạo khoá không giới hạn — một cuộc DoS tự gây. Tải lại là idempotent, nên một `kid` rác chỉ tốn tối đa một lần tải vô ích.

Hình dạng cache:

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

Hai khoá cùng lúc là trạng thái ổn định. Authorization server xoay bằng cách đưa khoá kế tiếp (`k_2026_04`) vào trước khi gỡ khoá trước đó (`k_2026_03`), nên token phát hành dưới khoá cũ vẫn hợp lệ cho tới khi hết hạn. Cache giữ hợp của cả hai; validator chọn theo `kid`.

### Thủ tục kiểm tra

MCP server chạy kiểm tra trước khi điều phối bất kỳ tool nào. Hình dạng mà `code/main.py` dùng:

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` giải mã JWT, tìm khoá ký từ cache JWKS (làm mới một lần khi trượt), kiểm tra chữ ký, rồi so `iss` với allow-list, so `aud` với resource chuẩn tắc của server này, kiểm tra `exp` và scope bắt buộc — trả về một challenge `WWW-Authenticate` ngay ở lần thất bại đầu tiên. Giữ nó là một thủ tục duy nhất trên resource server nghĩa là mọi điểm vào (mọi lệnh gọi tool, mọi transport) đều đi qua cùng các kiểm tra; không có đường nào chạm tới tool mà không kiểm tra trước.

### Token opaque dùng introspection, không đoán mò

Không phải mọi access token đều là JWT. Nếu issuer ghi rõ token là opaque, resource server không thể giải mã nó thành các claim đáng tin. Nó gửi token tới endpoint introspection RFC 7662 của issuer qua một backchannel có xác thực và yêu cầu `active: true`, đúng ngữ cảnh issuer mong đợi, đúng audience hoặc resource MCP, các claim thời gian còn hạn, và các scope mà tool cụ thể cần.

Cache introspection theo issuer, theo một digest một chiều của token và theo resource MCP. Không bao giờ dùng token rõ làm nhãn log hay cache. Giới hạn một mục cache dương bởi giá trị sớm nhất trong ba thứ: hạn của token, hướng dẫn cache của issuer, và mục tiêu độ tươi thu hồi của triển khai. Giữ cache âm đủ ngắn để một token vừa phát hành không bị coi là inactive một cách sai lệch. Kết quả cho một resource không thể cho phép một resource khác, kể cả khi chuỗi token opaque giống hệt.

Đừng chọn chế độ kiểm tra dựa trên nội dung token do kẻ tấn công kiểm soát. Ghim hành vi JWT hay introspection vào metadata issuer đã kiểm tra và cấu hình triển khai. Trên đường JWT, ghim thuật toán được chấp nhận và `jwks_uri` đáng tin; không bao giờ đi theo URL khoá hay thuật toán chỉ do header token chọn.

### Thu hồi là một hợp đồng về độ tươi

RFC 7009 cho phép client yêu cầu authorization server thu hồi một token. Request đó không xoá các bản sao đã được cache ở mọi resource server. Hãy định nghĩa độ trễ thu hồi tối đa chấp nhận được và buộc mọi cache tôn trọng nó.

Các triển khai token opaque có thể đạt mức thu hồi chặt hơn bằng cách introspection ở mỗi lệnh gọi rủi ro cao hoặc dùng cache dương ngắn. Các triển khai JWT tự chứa thường kết hợp vòng đời access token ngắn với thu hồi refresh token, gỡ khoá cho sự cố toàn issuer, và tuỳ chọn một denylist theo chủ thể, phiên hoặc token-id để từ chối cục bộ trong tình huống khẩn cấp. Một JWT đã ký vẫn hợp lệ về mặt mật mã cho tới khi hết hạn, trừ khi resource server có bằng chứng thu hồi bên ngoài mới nhất.

Đăng xuất, vô hiệu hoá tài khoản, rút lại đồng ý và xử lý sự cố là những tác nhân khác nhau nhưng phải hội tụ về một phát biểu đo đếm được: sau tối đa cửa sổ thu hồi đã khai báo, mọi bản sao đều từ chối thông tin đăng nhập đó. Hãy kiểm thử phát biểu đó qua load balancer, không chỉ với một tiến trình đã ấm.

### Hỏng phụ thuộc cần một quyết định đã khai báo

Đừng bao giờ ứng biến chính sách sẵn sàng bên trong một exception handler.

| Sự cố | Hành vi an toàn trong production |
|---|---|
| Làm mới JWKS theo lịch thất bại, `kid` đã biết vẫn còn trong cache có giới hạn và còn hợp lệ | Chỉ tiếp tục trong cửa sổ stale-on-error đã khai báo và phát bằng chứng health suy giảm |
| Token có `kid` lạ và lần làm mới duy nhất được phép thất bại | Từ chối; không bao giờ chấp nhận một chữ ký không kiểm tra được |
| Introspection không sẵn sàng | Thất bại đóng với các lệnh gọi được bảo vệ; không biến lỗi mạng thành `active: true` |
| Metadata protected resource hoặc issuer thay đổi bất ngờ | Dừng đăng ký mới và lấy token; chỉ giữ cấu hình đã ghim rõ ràng và còn hạn theo một chính sách sự cố có giới hạn |
| Endpoint thu hồi không sẵn sàng | Báo đăng xuất hoặc thu hồi là chưa hoàn tất, giữ thông tin đăng nhập cục bộ ở trạng thái không dùng được nếu có thể, và không tuyên bố thu hồi toàn cục đã thành công |
| Nguồn đồng hồ hoặc loại claim không hợp lệ | Từ chối thay vì nới độ lệch cho tới khi token qua được |

Phân loại lỗi tách khỏi thông tin đăng nhập không hợp lệ. Sự cố phụ thuộc là lỗi vận hành với chính sách health và retry. Chữ ký, issuer, audience, thời hạn hay scope sai là một lần từ chối uỷ quyền. Cả hai đều không chạm tới handler của tool, và cả hai đều không được rò nội dung token vào bằng chứng kiểm toán.

### Dạo một vòng phát lại audience (giới hạn đặc quyền của access token)

Server A (`notes.example.com`) và Server B (`tasks.example.com`) cùng đăng ký với một authorization server. Server A bị xâm phạm. Kẻ tấn công lấy token notes của một người dùng và phát lại nó với Server B.

Validator của Server B:

1. Giải mã JWT, lấy JWKS theo `kid`, kiểm tra chữ ký.
2. So `iss` với `authorization_servers` trong metadata protected resource của nó. (Qua — cùng IdP.)
3. Kiểm tra `aud == "https://tasks.example.com"`. (Trượt — `aud` của token là `https://notes.example.com`.)
4. Trả 401 với `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`.

Claim audience là hàng phòng thủ duy nhất chống tấn công này ở tầng giao thức. Bỏ qua nó vì hiệu năng là lỗi production phổ biến nhất; validator phải chạy trên mọi request, không chỉ lúc bắt đầu phiên. Spec gọi đây là **giới hạn đặc quyền của access token**: một MCP server `MUST` từ chối mọi token không nêu tên nó trong audience.

> **Ghi chú về tên gọi.** Spec dành thuật ngữ *confused deputy* cho một vấn đề liên quan nhưng khác: một MCP server đóng vai **proxy** OAuth tới API bên thứ ba, dùng một client ID tĩnh, chuyển tiếp token mà không lấy đồng ý của người dùng theo từng client. Audience binding sửa lỗi phát lại ở trên; cách sửa confused deputy là đồng ý theo từng client **cộng** việc không bao giờ chuyển token đầu vào tới API thượng nguồn (MCP server `MUST` lấy token thượng nguồn riêng của chính nó).

### Tấn công mix-up (phòng thủ phía client mà server không thể cung cấp)

Một client nói chuyện với nhiều authorization server trong vòng đời của nó. Một AS độc hại có thể tìm cách khiến client đổi authorization code của một AS lương thiện tại token endpoint của kẻ tấn công. Audience binding không giúp gì ở đây — tấn công xảy ra trước khi có bất kỳ token nào. Phòng thủ nằm ở client (RFC 9207):

1. Trước khi chuyển hướng, client ghi lại `issuer` mong đợi từ metadata AS đã kiểm tra.
2. Khi nhận authorization response, client so tham số `iss` trả về với issuer đã ghi (so chuỗi đơn giản, không chuẩn hoá) trước khi gửi code đi bất cứ đâu.
3. Lệch (hoặc thiếu `iss` khi AS đã quảng bá `authorization_response_iss_parameter_supported`) → từ chối, và thậm chí không hiển thị các trường `error`.

Chỉ PKCE không ngăn được mix-up, vì client trao `code_verifier` của mình cho bất kỳ token endpoint nào mà nó bị lái tới. Đây là lý do spec ghi lại issuer theo từng request cùng với verifier PKCE và `state`.

### Các chế độ hỏng

- **JWKS cũ.** Validator từ chối các token hợp lệ sau khi AS xoay một khoá. Cách sửa là mẫu làm mới bằng cron + tải lại khi cache trượt ở trên. Không bao giờ cache JWKS mà không có job làm mới.
- **Xoay làm đường dự phòng.** Nối đường cache-miss với xoay-và-tạo thay vì tải lại là một lỗi thật: nó không bao giờ tạo ra `kid` còn thiếu, và nó biến các giá trị `kid` do kẻ tấn công kiểm soát thành một cuộc DoS tạo khoá. Đường dự phòng phải là `refresh-jwks` idempotent.
- **Thiếu claim `aud`.** Một số IdP mặc định bỏ `aud` trừ khi có `resource` trong request token. Validator phải từ chối token thiếu `aud`, không coi việc vắng mặt là wildcard.
- **Mix-up do thiếu kiểm tra `iss`.** Một client không kiểm tra tham số `iss` của authorization response theo RFC 9207 với issuer đã ghi trước khi chuyển hướng có thể bị lái vào việc đổi code của một AS lương thiện tại token endpoint của kẻ tấn công. Đây là lỗi phía client; resource server không thể bù đắp.
- **Đua nâng scope.** Hai luồng step-up đồng thời cho cùng một người dùng có thể cùng thành công và tạo ra hai access token với scope khác nhau. Validator phải dùng token được trình trên request, không tra "scope hiện tại của người dùng" — điều đó tạo một cửa sổ TOCTOU.
- **Đánh cắp registration token.** Một `registration_access_token` bị lộ cho phép kẻ tấn công viết lại redirect URI. Hãy băm chúng khi lưu; yêu cầu client trình bản rõ ở mỗi lần cập nhật; thực hiện token rotation (xoay vòng token) khi có nghi ngờ.
- **`iss` không được ghim.** Một validator chấp nhận mọi `iss` cho phép kẻ tấn công dựng authorization server của riêng chúng, đăng ký một client cho audience mục tiêu, và phát hành token. Danh sách `authorization_servers` trong metadata protected resource là allow-list; hãy thực thi nó.
- **Xung đột cache thông tin đăng nhập hoặc token.** Một client chỉ khoá đăng ký theo resource có thể trình danh tính của authorization server này với server khác. Một client chỉ khoá access token theo issuer có thể phát lại token tại audience sai. Khoá đăng ký theo issuer đã kiểm tra, khoá access token theo `(issuer, resource)`, và đăng ký lại mỗi khi issuer thay đổi.

```figure
t3-jwks-rotate
```

## Áp dụng

`code/main.py` đi qua toàn bộ luồng production bằng Python stdlib với ba vai trò: `AuthorizationServer`, `ResourceServer`, và `Client`. Luồng:

Từ thư mục gốc của repo, chạy:

```bash
cd phases/13-tools-and-protocols/18-mcp-auth-production
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Lệnh đầu tiên in bản ghi đăng ký gắn issuer và kiểm tra
token. Lệnh thứ hai báo mười tám kiểm tra đều qua. Không lệnh nào mở
listener mạng hay ghi thông tin đăng nhập.

1. Authorization server công bố metadata RFC 8414 tại `/.well-known/oauth-authorization-server`.
2. MCP client gọi endpoint metadata và kiểm tra các lựa chọn đăng ký (`client_id_metadata_document_supported` cho CIMD, `registration_endpoint` cho DCR) cùng hỗ trợ PKCE `S256`.
3. Client kiểm tra xem có đăng ký trước theo phạm vi issuer không, nếu không thì đăng ký bằng Client ID Metadata Document HTTPS của mình. DCR đã bị khai tử vẫn là một phương thức tương thích kiểm thử riêng được.
4. Client ghi lại issuer đã kiểm tra, tạo một challenge S256, nhận một authorization code dùng một lần kèm `iss`, kiểm tra issuer trả về đó, và đổi code với verifier gốc cùng resource indicator RFC 8707.
5. MCP client gọi một tool trên MCP server với `Authorization: Bearer ...`.
6. MCP server chạy `validate`, tìm khoá ký từ cache JWKS.
7. IdP xoay một khoá; lần làm mới theo lịch kéo lại JWKS vào cache.
8. Lệnh gọi kế tiếp kiểm tra với các khoá đã làm mới mà không cần khởi động lại, và token trước đó vẫn hợp lệ trong cửa sổ chồng lấn.
9. Một lần thử phát lại audience với một MCP resource khác nhận 401 với `audience mismatch` và một con trỏ `resource_metadata`.

JWT ở đây dùng HS256 với một shared secret (bí mật chia sẻ) (để bài chỉ cần stdlib). Production dùng RS256 hoặc EdDSA với mẫu JWKS ở trên; logic kiểm tra ngoài ra giống hệt. Vì IdP và resource server sống trong cùng một tiến trình, `refresh_jwks` đọc trực tiếp danh sách khoá của authorization server; khi qua mạng, đó là một `GET` HTTP tới `jwks_uri`.

## Đóng gói

Bài này tạo ra `outputs/skill-mcp-auth.md`. Với một cấu hình MCP server và một tập năng lực IdP, skill phát ra bề mặt xác thực cần dựng — metadata protected resource, đường đăng ký cần dùng (CIMD, đăng ký trước, hoặc DCR dự phòng), lịch làm mới JWKS, ánh xạ scope, và các quy tắc từ chối áp dụng khi IdP không hỗ trợ đủ hồ sơ RFC.

## Bài tập

1. Chạy `code/main.py`. Lần theo luồng. Để ý cách IdP xoay một khoá ở bước 6, `refresh_jwks` theo lịch kéo lại tập đã công bố, và cả token cũ (cửa sổ chồng lấn) lẫn token mới đều kiểm tra được mà không cần khởi động lại.

2. Thêm một IdP mới vào danh sách `authorization_servers` của metadata protected resource. Phát hành một token được ký bởi IdP mới và xác nhận validator chấp nhận. Phát hành một token được ký bởi một IdP không có trong danh sách và xác nhận validator từ chối với `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`.

3. Thêm một kiểm tra giới hạn tốc độ vào `register_client` chạy trước khi cơ quan đăng ký chấp nhận request. Dùng token-bucket theo IP nguồn giữ trong một dict nhỏ khoá theo IP.

4. Đọc RFC 7591 và chỉ ra hai trường mà handler `/register` của bài không kiểm tra. Thêm phần kiểm tra đó. (Gợi ý: `software_statement` và scheme URI của `redirect_uris`.)

5. Thêm một authorization server thứ hai. Xác nhận client lưu một đăng ký riêng khoá theo issuer và từ chối dùng lại token hay `client_id` của issuer thứ nhất.

6. Chứng minh cách sửa DoS. Gửi validator một token với `kid` ngẫu nhiên và xác nhận `refresh_jwks` chạy tối đa một lần và số khoá của authorization server không tăng. Rồi cố ý nối lại đường dự phòng với xoay-và-tạo và xem số khoá leo lên theo từng token rác — sau đó khôi phục lại việc tải lại.

7. Thử DCR đã bị khai tử với cả client `native` và `web`. Xác nhận một client web với redirect URI HTTP và một client native không có redirect loopback chính xác đều bị từ chối.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| ASM | "tài liệu metadata OAuth" | JSON `/.well-known/oauth-authorization-server` của RFC 8414 |
| CIMD | "URL metadata của client" | Client ID Metadata Document: một URL HTTPS dùng làm `client_id`; AS tự kéo JSON về. Cơ chế đăng ký ưu tiên trong MCP 2026-07-28 |
| DCR | "tự đăng ký client" | RFC 7591 `POST /register`; đã bị khai tử với MCP hiện tại và chỉ giữ để tương thích |
| JWKS | "khoá công khai để kiểm tra JWT" | JSON Web Key Set, tải từ `jwks_uri`, đánh chỉ mục theo `kid` |
| Rotate vs refresh | "cập nhật khoá" | *Rotate* = AS tạo/gỡ khoá ký; *refresh* = resource server tải lại tập đã công bố. Resource server chỉ làm mới |
| Resource indicator | "tham số audience" | Tham số `resource` của RFC 8707 ghim token vào một server |
| `aud` claim | "audience" | Claim JWT mà validator so với URL resource chuẩn tắc |
| Audience replay | "phát lại token" | Token phát cho Server A bị trình cho Server B; chặn bằng kiểm tra audience (spec: giới hạn đặc quyền của access token) |
| Confused deputy | "lạm dụng token proxy" | Một MCP proxy với client ID tĩnh chuyển tiếp token mà không có đồng ý theo từng client; khác với audience replay |
| Mix-up attack | "sai token endpoint" | Client bị lái đổi code của một AS lương thiện tại endpoint của kẻ tấn công; chặn phía client nhờ `iss` RFC 9207 |
| `iss` allow-list | "các authorization server đáng tin" | Tập được nêu trong `authorization_servers` của metadata protected resource |
| `resource_metadata` | "nơi tìm tài liệu PRM" | Tham số `WWW-Authenticate` nêu URL metadata RFC 9728 trên 401/403 |
| Public client | "client native hoặc trình duyệt" | Client OAuth không có `client_secret`; PKCE bù vào |
| `WWW-Authenticate` | "header phản hồi 401/403" | Mang các chỉ thị `Bearer error=...` giúp client phục hồi |

## Đọc thêm

- [Đặc tả authorization của MCP (2026-07-28)](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization) - hồ sơ authorization MCP hiện tại
- [Changelog MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/changelog) - CIMD, kiểm tra issuer, khai tử DCR, và các thay đổi thông tin đăng nhập khoá theo issuer
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — hợp đồng khám phá
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR (đường dự phòng)
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — bằng chứng sở hữu của public client
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — ghim audience
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — khám phá resource server
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — tham số `iss` chặn tấn công mix-up
- [RFC 7662: OAuth 2.0 Token Introspection](https://datatracker.ietf.org/doc/html/rfc7662)
- [RFC 7009: OAuth 2.0 Token Revocation](https://datatracker.ietf.org/doc/html/rfc7009)

---

[← Bài trước: Gateway MCP phi trạng thái và kiểm soát tiếp nhận tại registry (sổ đăng ký)](../../17-mcp-gateways-and-registries/docs/vi.md) · [Bài sau: A2A →](../../19-a2a-protocol/docs/vi.md)
