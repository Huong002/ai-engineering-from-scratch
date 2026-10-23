# Kỹ nghệ tuân thủ đặc tả MCP: quản lý phiên bản (versioning), bằng chứng và vận hành (operations)

> Một server không trở nên tuân thủ đặc tả (conformance) chỉ vì luồng thuận lợi của nó chạy được qua một SDK. Tuân thủ đặc tả nằm ở tầng wire, ở ranh giới phiên bản, xuyên qua các bên trung gian, và trong lúc rollback.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 13, Bài 09 (kênh truyền), Giai đoạn 13, Bài 17 (gateway), Giai đoạn 13, Bài 30 (tiếp nhận vào sổ đăng ký)
**Time:** ~100 phút

## Mục tiêu học tập

- Biến các quy tắc quy phạm của MCP thành transcript wire vàng và phủ định.
- Giữ hành vi nghiêm ngặt `2026-07-28` tách khỏi dự phòng kiểu cũ có giới hạn.
- Phân biệt trường bổ sung không xác định với một `resultType` không xác định không hợp lệ.
- So sánh bằng chứng JSON-RPC thô với góc nhìn đã được SDK chuẩn hoá.
- Chứng minh tính toàn vẹn của header và body qua một ranh giới proxy thật.
- Chặn phát hành bằng bằng chứng transcript đã che, sức khoẻ và rollback.

## Vấn đề

Client của bạn gọi `tools/list` qua một SDK và nhận được danh sách công cụ. Bài kiểm thử tích hợp pass.

Kết quả đó để lại những câu hỏi quan trọng chưa có lời đáp:

- Request có mang metadata giao thức hiện đại theo từng request không?
- `MCP-Protocol-Version`, `Mcp-Method` và `Mcp-Name` có khớp với body JSON-RPC không?
- Response có chứa `resultType` hợp lệ trên wire không, hay SDK đã tự tạo ra một cái?
- Client có giữ lại một trường bổ sung trong tương lai không?
- Một lỗi hiện đại được nhận diện có vô tình kích hoạt handshake kiểu cũ không?
- Proxy có giữ nguyên status của origin và lỗi JSON-RPC không?
- Bộ tuần tự hoá thông báo (notification serializer) có phát ra một response bị cấm không?
- Bộ phận vận hành có chứng minh được vì sao một bản phát hành được promote hay rollback mà không phải lưu secret không?

Tuân thủ đặc tả là một tập các bất biến quan sát được. Hãy xây một harness ghi lại những bất biến đó trước khi lưu lượng production phải tự phát hiện ra chúng.

```figure
mcp-conformance-operations
```

## Bắt đầu từ các kỷ nguyên phiên bản

MCP `2026-07-28` dùng metadata tự chứa theo từng request. Một request hiện đại mang `params._meta.io.modelcontextprotocol/protocolVersion` và `params._meta.io.modelcontextprotocol/clientCapabilities`. Các khoá có namespace chính xác rất quan trọng; những alias trần như `protocolVersion` hay `clientCapabilities` là sai định dạng. Khi có các header định tuyến được phản chiếu ở ranh giới HTTP, giá trị của chúng phải khớp với body JSON-RPC. Các result thành công kiểu hiện đại mang `resultType`.

Các phiên bản tới `2025-11-25` dùng kỷ nguyên khởi tạo cũ hơn. Một result kiểu cũ không có `resultType` chỉ được hiểu là complete sau khi client đã chọn kỷ nguyên cũ đó.

Đừng tạo một validator dễ dãi chấp nhận cả hai dạng cùng lúc. Hãy dùng hai nhánh:

| Nhánh | Bằng chứng đầu vào | Thiếu `resultType` | Khởi tạo |
|---|---|---|---|
| Hiện đại | `server/discover` thành công hoặc response hiện đại được nhận diện | Không hợp lệ | Không phải đường mặc định |
| Kiểu cũ | Allowlist đã cấu hình cộng một result `initialize` kiểu cũ hợp lệ sau lần dò hiện đại không kết luận | Được hiểu là complete | Kỷ nguyên đó bắt buộc phải có |

Việc tách nhánh ngăn một peer hiện đại sai định dạng được hưởng lợi từ kiểm tra yếu hơn.

### Chế độ nghiêm ngặt

Chế độ nghiêm ngặt đòi hỏi bằng chứng về hành vi hiện đại. Một `server/discover` thành công chứng minh nhánh hiện đại. Một lỗi JSON-RPC hiện đại được nhận diện cũng chứng minh điều đó. Hãy sửa request hoặc dừng lại. Không bao giờ hạ cấp chỉ vì server trả về `-32020`, `-32021` hay `-32022`.

### Chế độ dự phòng

Chế độ dự phòng thực hiện đúng một lần dò hiện đại có giới hạn. Timeout, trả lời rỗng, kết nối bị đóng, hay response không nhận diện được đều là không kết luận. Điều đó không chứng minh peer là kiểu cũ. Chỉ endpoint được cấu hình rõ ràng hoặc nằm trong allowlist cho mục đích tương thích mới có thể nhận một lần dò kiểu cũ có giới hạn, và client chỉ chọn nhánh kiểu cũ sau khi đã kiểm tra result `initialize` của lần dò đó cùng revision kiểu cũ đã thương lượng (negotiation).

Dự phòng không phải là “thử kiểu cũ sau bất kỳ lỗi nào”. Một lỗi hiện đại được nhận diện chứa thông tin sửa lỗi hữu ích. Hạ cấp sau đó có thể che giấu việc header không khớp, thiếu khai báo capability, hay phiên bản không được hỗ trợ.

Điều này ngăn kẻ tấn công, sự cố, hay proxy lọc buộc phải hạ cấp bằng cách bỏ response hiện đại. Hãy ghi lại cùng nhau: chính sách endpoint, quan sát hiện đại không kết luận, bằng chứng kiểu cũ dương tính chính xác, và kỷ nguyên được chọn.

Ghi kỷ nguyên đã chọn bên cạnh mỗi transcript. Thiếu dữ kiện đó, một trường bị thiếu có thể trông chấp nhận được ở lần chạy này và không hợp lệ ở lần chạy khác.

## Xây dựng kho transcript

Một fixture transcript ghi lại những gì đã đi qua ranh giới, không chỉ lời gọi SDK:

```json
{
  "name": "golden-modern-list",
  "era": "modern",
  "headers": {
    "MCP-Protocol-Version": "2026-07-28",
    "Mcp-Method": "tools/list"
  },
  "request": {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {
      "_meta": {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {}
      }
    }
  },
  "responseStatus": 200,
  "responseBody": {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
      "resultType": "complete",
      "tools": []
    }
  }
}
```

Giữ hai lớp fixture.

### Transcript vàng

Transcript vàng chứng minh hành vi được chấp nhận:

- request khám phá (discover) hoặc request method kiểu hiện đại với metadata và header khớp
- result complete có đủ các trường bắt buộc
- result `input_required` khi method có thể yêu cầu thêm đầu vào
- result phần mở rộng chỉ sau khi capability tương ứng đã được công bố
- result kiểu cũ không có `resultType`, nhưng chỉ trong kỷ nguyên kiểu cũ đã chọn
- xử lý notification không có response JSON-RPC

Một transcript vàng cần chính xác chứ không cần lớn. Hãy giữ các ID và timestamp dễ thay đổi ở dạng xác định, hoặc chuẩn hoá chúng trước khi so sánh.

### Transcript phủ định

Transcript phủ định chứng minh hành vi từ chối:

- header và body không khớp
- thiếu capability theo từng request
- phiên bản giao thức khớp nhưng không được hỗ trợ
- thiếu `resultType` kiểu hiện đại
- `resultType` không xác định hoặc chưa được công bố
- `jsonrpc` của response khác `2.0`, hoặc ID khác về giá trị hay kiểu JSON
- response chứa cả `result` và `error`, hoặc không có cái nào
- lỗi không có `code` là số nguyên và `message` là chuỗi
- lỗi giao thức đã biết bị ánh xạ sang HTTP status sai
- có response được phát ra cho một notification
- envelope JSON-RPC sai định dạng
- proxy làm sập một lỗi giao thức

Với mỗi ca phủ định, hãy khẳng định ranh giới từ chối và mã lỗi ổn định. “Lời gọi thất bại” là quá yếu. Một 500 do proxy tạo ra và một `-32020` từ origin đều có thể trông như thất bại nhưng kể cho người vận hành hai câu chuyện hoàn toàn khác nhau.

Fixture header không khớp phải bao gồm response JSON-RPC HTTP 400 thực tế của server với request ID khớp và mã lỗi `-32020`. Hãy bắt buộc điều đó một cách tự động mỗi khi validator cục bộ thấy `HeaderMismatch`; đừng biến việc kiểm tra response thành một cờ tuỳ chọn của fixture. Một ca có HTTP 500 và không có body vẫn thất bại dù mã từ chối cục bộ đúng. Một harness dừng lại ngay sau khi validator request của chính nó ném lỗi thì mới chỉ kiểm tra chính nó, chứ không kiểm tra hành vi wire của server.

Dự án tuân thủ đặc tả MCP chính thức hữu ích như một bộ kiểm thử bên ngoài và một tham chiếu có phiên bản. Nhưng vẫn giữ transcript cục bộ của bạn. Chúng ghi lại proxy, SDK, xác thực, phần mở rộng và đường phát hành của bạn — những thứ một bộ kiểm thử chung không thể biết.

## Giá trị header phải khớp với body RPC

Trong Streamable HTTP kiểu hiện đại, các bên trung gian có thể định tuyến hoặc áp chính sách dựa trên header phản chiếu. Body JSON-RPC vẫn là nguồn sự thật của giao thức. Không khớp là lỗi toàn vẹn, không phải gợi ý để chọn một trong hai giá trị.

Kiểm tra theo thứ tự này:

1. Phân tích và kiểm tra envelope JSON-RPC cùng kiểu của metadata.
2. So sánh `MCP-Protocol-Version` với `params._meta.io.modelcontextprotocol/protocolVersion`.
3. So sánh `Mcp-Method` với `method`.
4. Khi method có tên định tuyến, so sánh `Mcp-Name` với giá trị tương ứng trong body.
5. Sau khi đã xác lập bằng nhau, mới quyết định phiên bản và tập capability khớp đó có được hỗ trợ không.

Thứ tự này phân biệt không khớp `-32020` với phiên bản không hỗ trợ `-32022`. Nó cũng ngăn gateway cho phép tên trong header trong khi origin thực thi một tên khác trong body.

Tên trường HTTP không phân biệt hoa thường, còn giá trị của chúng vẫn phân biệt hoa thường. Hãy chuẩn hoá tên header trước khi tra cứu và từ chối các bản trùng xung đột. Với `Mcp-Name` không an toàn, không phải ASCII, hoặc có khoảng trắng đầu/cuối, hãy giải mã đúng sentinel UTF-8 `=?base64?{Base64EncodedValue}?=` trước khi so với body. Từ chối sentinel không đầy đủ, Base64 không hợp lệ, UTF-8 không hợp lệ, hoặc giá trị thô không an toàn bằng `-32020`. Khoảng trắng thô xung quanh là không hợp lệ kể cả khi body chứa đúng các ký tự đó, vì giá trị ấy cần được mã hoá sentinel trước khi truyền.

Một bên trung gian có thể từ chối HTTP sai định dạng trước khi request tới server MCP, nên lỗi của nó có thể là lỗi HTTP không kèm JSON-RPC. Hãy ghi lại việc từ chối đến từ bên trung gian hay từ origin. Server MCP origin nên dùng hợp đồng lỗi giao thức khi nó đã xử lý một request JSON-RPC hợp lệ.

## Trường không xác định không phải result không xác định

Tương thích tiến (forward compatibility) cần hai quy tắc khác nhau.

### Trường bổ sung không xác định

Các đối tượng result và map `_meta` có thể có thêm trường. Validator nên giữ hoặc bỏ qua một trường bổ sung tuỳ theo vai trò của nó, trừ khi trường đó vi phạm một hợp đồng dành riêng. Ví dụ mẫu giữ nguyên result thô đầy đủ trong bằng chứng và chấp nhận `futureHint` bên cạnh một result đã biết.

Nếu bạn là proxy trong suốt, giữ lại trường không xác định thường an toàn hơn là cắt bỏ. Nếu bạn là client ứng dụng, bỏ qua nó có thể hợp lệ. Nhưng bài kiểm thử so sánh (differential) vẫn nên phát hiện ra rằng SDK đã bỏ nó, để hành vi là có chủ đích.

### `resultType` không xác định

`resultType` là một bộ phân biệt (discriminator). Result lõi kiểu hiện đại dùng `complete` hoặc `input_required`. Một phần mở rộng chỉ có thể thêm giá trị khác khi capability của nó đã được công bố. Ví dụ phần mở rộng Tasks có thể thêm `task` trong ngữ cảnh capability đã thương lượng đó.

Một discriminator không xác định hoặc chưa được công bố không thể được coi là complete một cách an toàn. Client không biết mình đang bỏ đi vòng đời nào. Hãy từ chối nó.

Vì vậy cùng một response thô có thể chứa một trường không xác định chấp nhận được và một kiểu result không xác định không chấp nhận được. Hãy kiểm thử cả hai.

Discriminator chỉ là lớp đầu. Hãy kiểm tra payload riêng theo method sau đó. Một result `tools/list` complete cần mảng `tools` mà các descriptor có tên duy nhất không rỗng, mô tả hữu ích, và `inputSchema` có gốc là object. Một result `task` chỉ hợp lệ cho `tools/call` đủ điều kiện với capability Tasks và yêu cầu `taskId`, status đã biết, timestamp tạo và cập nhật, cùng `ttlMs`, cộng thêm một polling interval tuỳ chọn hợp lệ. Một result `completion/complete` complete yêu cầu đối tượng `completion` với không quá 100 giá trị chuỗi, `total` tuỳ chọn là số nguyên không âm và không nhỏ hơn các giá trị trả về, và `hasMore` tuỳ chọn kiểu Boolean. Một `resultType` viết đúng chính tả không thể làm cho payload sai định dạng trở nên tuân thủ.

## Bất biến của notification

Một notification JSON-RPC không có `id`. Bên nhận không được gửi response thành công hay lỗi JSON-RPC.

Với một dạng notification HTTP được chấp nhận, harness mong đợi HTTP `202` với body rỗng. MCP `2026-07-28` không định nghĩa notification lõi nào từ client tới server qua Streamable HTTP. Ví dụ mẫu dùng một notification phần mở rộng của khoá học có namespace chỉ để kiểm tra bất biến một chiều của serializer. Đừng trình bày nó như một method lõi mới.

Hãy kiểm thử serializer, không chỉ handler. Handler có thể trả về `None` trong khi middleware bọc nó thành một đối tượng thành công JSON. Hãy ghi lại các byte ra cuối cùng (egress).

## Thêm phép so sánh SDK

SDK thường biến đối tượng trên wire thành kiểu dữ liệu tiện dụng của ngôn ngữ. Điều đó hữu ích, nhưng một đối tượng đã chuẩn hoá không thể chứng minh đã nhận được gì.

Với mỗi fixture rủi ro cao, hãy ghi lại:

1. Status, header và body response thô trước khi SDK giải mã.
2. Giá trị trả về hoặc exception đã được SDK chuẩn hoá.
3. Phép chiếu ngữ nghĩa kỳ vọng cho kỷ nguyên đã chọn.
4. Các trường được SDK nâng lên, tự tạo, cắt bỏ hoặc thay đổi.

Ví dụ mẫu cho phép SDK loại bỏ các thông tin sổ sách trên wire đã biết như `resultType`, `_meta`, `ttlMs` và `cacheScope` khi so sánh payload ứng dụng. Nó báo cáo `futureHint` bị bỏ vì trường ngữ nghĩa không xác định đó đã biến mất.

Đừng cho rằng mọi khác biệt đều là lỗi SDK. Mục đích là làm cho phép biến đổi trở nên nhìn thấy được. Hãy quyết định thành phần của bạn là endpoint ứng dụng, thứ có thể bỏ qua một trường bổ sung, hay một trung gian trong suốt, thứ nên giữ lại nó.

Chạy phép so sánh với mọi SDK và phiên bản bạn phát hành. Nếu hai SDK chuẩn hoá cùng một transcript theo cách khác nhau, chính sách phát hành nên nói rõ hành vi nào được chấp nhận, thay vì chọn kết quả tiện nhất sau khi sự việc xảy ra.

## Ghi lại bằng chứng từ proxy

Phần lớn lỗi MCP trên production xảy ra xuyên qua nhiều hơn một tiến trình. Hãy ghi ba góc nhìn:

| Góc nhìn | Bằng chứng tối thiểu |
|---|---|
| Ingress | header request, body JSON-RPC, content type, route đã xác thực, thời điểm nhận |
| Origin | header được chuyển tiếp và digest body, status origin, header và body response |
| Egress | status, header, body mà client thấy, và thời điểm gửi |

Ví dụ mẫu phát hiện hai phép biến đổi phổ biến:

- lỗi JSON-RPC HTTP 400 hoặc 404 của origin biến thành 500 chung chung của proxy
- body JSON-RPC ở egress khác với body ở origin

Thêm các khẳng định riêng cho triển khai về content type, `Accept`, nén, SSE theo từng request, header cache và tương quan trace. Ghi cả hai phía của điểm kết thúc TLS khi chính sách cho phép. Không bao giờ ghi log credential chỉ để chứng minh đường đi.

## Che thông tin trước khi bằng chứng rời khỏi bộ nhớ

Việc che thông tin (redaction) là một phần của vận hành tuân thủ đặc tả, không phải công việc dọn dẹp về sau. Hãy áp dụng nó trước khi tuần tự hoá, băm, ghi log, tạo artifact kiểm thử, hay tải lên báo cáo lỗi.

Ví dụ mẫu chuyển tên khoá về cùng dạng chữ và bỏ các dấu phân cách trước khi so khớp, rồi thay thế đệ quy các giá trị dưới những khoá như `Authorization`, `Cookie`, `Set-Cookie`, `X-Api-Key`, `accessToken`, `clientSecret`, `registrationAccessToken`, `token`, `password`, `secret` và `api_key`. Việc chuẩn hoá và danh sách chặn phải dùng cùng một dạng, để các biến thể camelCase, có gạch nối, có gạch dưới và có dấu chấm không thể vượt qua chính sách của nhau. Bộ thu thập trên production nên thêm chính sách theo từng đối số của method, vì một khoá vô hại như `query` vẫn có thể chứa dữ liệu cá nhân hoặc dữ liệu bị quản lý.

Hãy băm gói bằng chứng đã che. Chỉ giữ bản ghi thô trong một hệ thống được phê duyệt và tồn tại ngắn hạn khi một cuộc điều tra cụ thể cần đến chúng. Digest chứng minh gói đã che nào đã dẫn tới quyết định; nó không làm lộ giá trị đã bị loại.

## Đưa sức khoẻ và rollback vào cổng phát hành

Tuân thủ giao thức là cần nhưng chưa đủ để phát hành. Một bản ứng viên tuân thủ vẫn có thể timeout, rò rỉ bộ nhớ, hay làm quá tải một phụ thuộc.

Định nghĩa cửa sổ sức khoẻ trước khi triển khai:

- số mẫu tối thiểu
- tỉ lệ lỗi tối đa
- phân vị độ trễ tối đa
- giới hạn bão hoà hoặc tài nguyên
- thời lượng quan sát
- so sánh với baseline đã được tiếp nhận

Cũng định nghĩa bằng chứng rollback trước khi triển khai:

- phiên bản trước chính xác
- digest bằng chứng tiếp nhận
- các pin SHA-256 cho artifact và descriptor
- status Registry hiện tại
- kết quả sức khoẻ hiện tại
- quy trình khôi phục route
- một chứng thực (attestation) trên đúng các trường đó từ một danh tính release-controller đáng tin cậy

Yêu cầu mục tiêu rollback đó phải được kiểm chứng và khoẻ mạnh trước khi promote, không chỉ sau khi ứng viên thất bại. Một bản phát hành thành công mà không có đường phục hồi dùng được thì chưa sẵn sàng cho production.

Nếu ứng viên thất bại và mục tiêu rollback thiếu bằng chứng đó, hãy giữ lưu lượng lại thay vì đoán. “Rollback về bất cứ thứ gì từng ở đó” không phải một biện pháp kiểm soát vận hành.

Đừng rút gọn mức sẵn sàng thành các phép kiểm tra truthiness như một phiên bản khác rỗng, `healthy: "yes"`, hay một chuỗi bằng chứng tuỳ ý. Ví dụ mẫu yêu cầu kiểu chính xác, status active, ba digest SHA-256, một bên ký đáng tin cậy, và một chứng thực HMAC-SHA-256 hợp lệ trên toàn bộ payload rollback. Khoá demo xác định của nó là một fixture không bí mật. Trên production, hãy tiêm một khoá được bảo vệ, kết quả xác minh KMS, hoặc bộ xác minh chứng thực bằng khoá công khai tại ranh giới phát hành.

Cổng phát hành cũng từ chối bằng chứng transcript, so sánh SDK, hay proxy rỗng. Mỗi nguồn phải mang digest bằng chứng hợp lệ. Một cửa sổ sức khoẻ xanh không thể lấp vào ranh giới chưa từng được quan sát.

## Tự tay làm

Chạy harness chỉ dùng thư viện chuẩn:

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations
python3 code/main.py
```

Bản demo chạy đúng mười lăm transcript vàng và phủ định, gồm cả result completion hợp lệ và sai định dạng, so sánh một result thô với góc nhìn SDK, kiểm tra một proxy đã làm sập lỗi của origin, đánh giá sức khoẻ, xác thực bằng chứng rollback và chọn mục tiêu đó.

Dạng kỳ vọng:

```json
{
  "transcriptsPassed": 15,
  "transcriptsTotal": 15,
  "sdkDroppedFields": ["futureHint"],
  "proxyIssues": [
    "proxy collapsed a protocol error into HTTP 500",
    "proxy changed the origin JSON-RPC body"
  ],
  "releaseAction": "rollback",
  "evidenceDigest": "..."
}
```

Đọc `code/main.py` theo thứ tự này:

1. `validate_request()` thực thi các quy tắc request và header riêng cho từng kỷ nguyên.
2. `validate_result()` phân biệt discriminator kiểu cũ bị thiếu, giá trị hiện đại hợp lệ, phần mở rộng, và giá trị không xác định.
3. `select_era()` hiện thực chính sách nghiêm ngặt và dự phòng có giới hạn.
4. `run_transcript()` đánh giá các fixture vàng và phủ định.
5. `compare_sdk_view()` phơi ra các khác biệt do chuẩn hoá.
6. `inspect_proxy()` so sánh bằng chứng ingress, origin và egress.
7. `redact()` loại bỏ secret rõ ràng trước khi băm bằng chứng.
8. `rollback_evidence_ready()` kiểm tra các trường pin chính xác và chứng thực phát hành đáng tin cậy.
9. `ReleaseGate.evaluate()` ghép các bằng chứng tuân thủ, SDK, proxy, sức khoẻ và rollback không rỗng.

## Áp dụng

Chạy harness ở bốn thời điểm:

1. Mỗi khi thay đổi hiện thực, với một test adapter trong cùng tiến trình.
2. Với binary client và server đã build, qua transport thật.
3. Qua proxy hoặc gateway đã triển khai trong môi trường staging.
4. Trong lúc rollout canary với bằng chứng sức khoẻ và rollback trực tiếp.

Giữ nguyên tên ca ổn định qua các lớp. `negative-header-body-mismatch` phải mang cùng một bất biến trong báo cáo unit, end-to-end, proxy và canary. Digest bằng chứng sẽ khác vì ranh giới đã thay đổi; còn yêu cầu thì không được thay đổi.

Lưu schema của fixture trong version control. Lưu bằng chứng lần chạy đã che trong hệ thống phát hành của bạn. Chỉ lưu bản ghi thô ngắn hạn dưới các kiểm soát truy cập sự cố.

## Phòng thí nghiệm tương tác

### Lab A: chứng minh ranh giới kỷ nguyên

Từ thư mục `code`, mở Python:

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/code
python3 -q
```

Chạy:

```python
from main import *
validate_result({"tools": []}, "legacy")
validate_result({"tools": []}, "modern")
```

Lời gọi kiểu cũ suy ra `complete`. Lời gọi kiểu hiện đại ném `ProtocolViolation`. Giờ hãy thử dự phòng:

```python
select_era({"kind": "timeout"}, "fallback")
select_era(
    {"kind": "timeout"},
    "fallback",
    legacy_allowed=True,
    legacy_evidence={"kind": "initialize_success", "protocolVersion": LEGACY_VERSION},
)
select_era({"kind": "jsonrpc_error", "code": -32021}, "fallback")
```

Lần timeout đầu tiên đóng an toàn vì im lặng không phải bằng chứng kiểu cũ. Lần gọi thứ hai chọn kiểu cũ chỉ vì cấu hình cho phép và đã quan sát được một result khởi tạo kiểu cũ hợp lệ. Lỗi thiếu capability được nhận diện chứng minh nhánh hiện đại.

### Lab B: trường bổ sung so với discriminator

```python
validate_result({"resultType": "complete", "tools": [], "futureHint": True}, "modern")
validate_result({"resultType": "future_mode", "tools": []}, "modern")
```

Result đầu giữ `futureHint`. Result thứ hai bị từ chối vì discriminator vòng đời không xác định.

### Lab C: kiểm tra một phép biến đổi của SDK

```python
compare_sdk_view(
    {"resultType": "complete", "tools": [], "futureHint": {"mode": "new"}},
    {"tools": []},
)
```

Hãy quyết định thành phần của bạn có thể bỏ qua `futureHint` hay phải chuyển tiếp nó. Ghi lựa chọn đó vào chính sách phát hành. Đừng âm thầm xoá bỏ phép so sánh.

### Lab D: sửa proxy

Sửa trao đổi trong demo để egress giữ nguyên status và body của origin. Chạy lại `python3 main.py`. Các vấn đề proxy sẽ biến mất, nhưng phép so sánh SDK vẫn chặn promote. Sau đó đưa `futureHint` vào góc nhìn SDK và quan sát hành động đổi thành `promote` khi mọi nguồn bằng chứng đều qua.

## Phòng thí nghiệm thực hành

Thêm transcript SSE theo từng request vào harness.

Yêu cầu:

- Ghi lại status response, content type, thứ tự các sự kiện SSE, và việc kết thúc stream.
- Chứng minh mỗi sự kiện JSON-RPC có result hoặc error hợp lệ theo kỷ nguyên.
- Thêm ca phủ định cho proxy đệm toàn bộ stream trước khi chuyển tiếp.
- Thêm ca phủ định cho sự kiện SSE có id JSON-RPC khác với request.
- Che dữ liệu sự kiện trước khi ghi bằng chứng.
- Đưa thời lượng stream, độ trễ sự kiện đầu tiên và số sự kiện vào cửa sổ sức khoẻ.
- Buộc cổng phát hành chỉ chọn mục tiêu rollback có bằng chứng khi stream thất bại.

Thành công nghĩa là cùng một ca chạy trực tiếp và qua proxy, với báo cáo chỉ ra đúng ranh giới đã làm thay đổi hành vi.

## Sản phẩm bàn giao

Bài học này bàn giao `outputs/skill-mcp-conformance-release-gate.md`. Dùng nó để biến một thay đổi server, client, gateway hay SDK thành ma trận tuân thủ có phiên bản và một quyết định phát hành. Artifact yêu cầu bằng chứng wire thô, các ca phủ định, lựa chọn kỷ nguyên tường minh, phép so sánh SDK, chứng minh qua proxy, che thông tin, ngưỡng sức khoẻ và bằng chứng rollback.

## Kiểm chứng

Chạy demo và bộ kiểm thử xác định:

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Việc kiểm chứng phải chứng minh:

- mọi transcript vàng và phủ định được đưa vào đều đạt kết quả kỳ vọng
- request hiện đại yêu cầu đúng các khoá metadata có namespace
- tên header HTTP được so khớp không phân biệt hoa thường và giá trị `Mcp-Name` đã mã hoá được giải mã chính xác
- header và body không khớp trả về mã không khớp kiểu hiện đại
- phiên bản, ID của response, tính loại trừ giữa result và error, hình dạng lỗi, và ánh xạ HTTP đều được kiểm tra
- các yêu cầu payload tool-list, task và completion riêng theo method được thực thi
- mọi `HeaderMismatch` quan sát được đều yêu cầu một response JSON-RPC `-32020` HTTP 400 thực tế
- khoảng trắng thô trong `Mcp-Name` bị từ chối, còn khoảng trắng được mã hoá sentinel chính xác thì đi và về nguyên vẹn
- `resultType` bị thiếu chỉ hợp lệ trong kỷ nguyên kiểu cũ đã chọn
- trường bổ sung sống sót qua kiểm tra thô, còn kiểu result không xác định thì thất bại
- kiểu result của phần mở rộng yêu cầu capability đã công bố của chúng
- lỗi hiện đại được nhận diện không bao giờ gây dự phòng kiểu cũ
- notification không tạo ra response JSON-RPC nào
- việc SDK loại bỏ thông tin sổ sách và việc mất trường ngữ nghĩa được phân biệt
- việc proxy làm sập lỗi được phát hiện và credential được che đệ quy qua các biến thể camelCase và dấu phân cách
- promote yêu cầu bằng chứng transcript, SDK, proxy và vận hành khoẻ mạnh đều không rỗng
- cả promote và rollback đều yêu cầu một mục tiêu rollback đã xác thực, đã pin, active và khoẻ mạnh

## Các kiểu lỗi trên production

| Kiểu lỗi | Bài kiểm thử yếu báo cáo gì | Harness phải chứng minh gì |
|---|---|---|
| SDK tự tạo ra discriminator bị thiếu | “tools/list đã pass” | Result hiện đại thô thiếu `resultType` và không hợp lệ |
| Client hạ cấp sau `-32021` | “thử lại kiểu cũ đã chạy” | Lỗi hiện đại được nhận diện cấm dự phòng |
| Kiểu result không xác định bị coi là complete | “response đã phân tích” | Discriminator vòng đời chưa công bố bị từ chối |
| Proxy cho phép một tool còn origin thực thi tool khác | “request đã tới server” | `Mcp-Name` bằng tên định tuyến trong body ở mọi chặng |
| Harness ném lỗi trước khi đọc response của server | “bài kiểm thử header không khớp đã pass” | Response HTTP 400 và JSON-RPC `-32020` được ghi lại và kiểm tra |
| Proxy biến 400 của origin thành 500 chung chung | “lỗi upstream” | Status và body JSON-RPC của origin và egress được giữ nguyên |
| Middleware notification phát ra `{result: null}` | “handler trả về none” | Body egress cuối cùng rỗng và không có response JSON-RPC nào |
| SDK cắt bỏ một trường bổ sung | “các đối tượng có kiểu khớp nhau” | Góc nhìn thô và chuẩn hoá cho thấy đúng trường bị bỏ |
| Artifact lỗi làm lộ bearer token | “gói debug đã được tải lên” | Việc che thông tin xảy ra trước khi băm, ghi log hoặc tải lên |
| Kiểu khoá credential vượt qua được bước che | “denylist có chứa api_key” | Các biến thể camelCase và dấu phân cách dùng chung một dạng denylist chuẩn |
| Canary không có mẫu nào nhưng trông vẫn khoẻ | “không có lỗi” | Số mẫu tối thiểu được thực thi |
| Rollback chọn một build không xác định | “bản triển khai trước đã được khôi phục” | Phiên bản mục tiêu, digest tiếp nhận, các pin, status và sức khoẻ đều có mặt |

## Quy tắc vận hành

Hãy kiểm thử những byte bạn gửi, những byte mỗi bên trung gian chuyển tiếp, ngữ nghĩa mỗi SDK phơi ra, và bằng chứng mà bộ phận vận hành sẽ dùng khi bị áp lực. Tương thích là một nhánh tường minh. Rollback là một hành động phát hành có bằng chứng hậu thuẫn. Cả hai đều không nên là tác dụng phụ ngẫu nhiên của một parser dễ dãi.

## Đọc thêm

- [Giao thức cơ sở MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [Thương lượng phiên bản MCP](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Dự án tuân thủ đặc tả MCP chính thức](https://github.com/modelcontextprotocol/conformance)

---

[← Bài trước: Chuỗi cung ứng MCP Registry: tiếp nhận, lệch trạng thái và rollback](../../30-mcp-registry-supply-chain-and-drift/docs/vi.md)
