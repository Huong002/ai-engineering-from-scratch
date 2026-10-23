# Model Context Protocol (MCP)

> MCP cho một host AI một giao thức duy nhất để khám phá và gọi công cụ (tool), tài nguyên (resource) và prompt. Bản sửa đổi 2026-07-28 làm giao thức đó trở nên phi trạng thái (stateless): ngữ cảnh về năng lực và phiên bản đi theo mỗi yêu cầu, thay vì nằm trong một bắt tay gắn với kết nối.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 11, Bài 09 (Function calling), Giai đoạn 11, Bài 03 (Đầu ra có cấu trúc)
**Time:** ~75 phút

## Mục tiêu học tập

- Phân biệt host, máy khách (client), máy chủ (server), kênh truyền (transport) và primitive của máy chủ trong MCP.
- Xây một yêu cầu JSON-RPC (giao thức gọi từ xa qua JSON) kèm metadata mà MCP 2026-07-28 yêu cầu.
- Dùng `server/discover` để xem phiên bản, danh tính và năng lực.
- Trả về kết quả có kiểu và biết cache từ công cụ, tài nguyên và prompt.
- Giải thích cách MCP phi trạng thái hiện đại tương tác với các máy chủ thời bắt tay.
- Chọn ranh giới an toàn cho trạng thái, kênh truyền và phê duyệt của một máy chủ.

## Vấn đề

Ứng dụng của bạn cần một truy vấn cơ sở dữ liệu, một thao tác lịch và một trình đọc tệp. Không có giao thức chung, mỗi host AI đều phải tự viết phần keo khám phá, gọi, lỗi, kênh truyền và phân quyền cho đúng những năng lực đó.

MCP thu gọn ma trận tích hợp đó. Một máy chủ công bố bề mặt JSON-RPC chuẩn. Một máy khách tuân thủ có thể khám phá bề mặt đó, trình bày nó cho model hoặc người dùng, gọi nó và diễn giải kết quả mà không cần adapter riêng cho từng máy chủ.

Ranh giới quan trọng rất dễ bị bỏ qua. MCP chuẩn hoá việc giao tiếp. Nó không quyết định model nên gọi công cụ nào, không làm nội dung không đáng tin trở nên an toàn, và không biến một yêu cầu phi trạng thái thành trạng thái ứng dụng bền vững. Host và máy chủ của bạn vẫn sở hữu những quyết định đó.

## Khái niệm

![Host MCP, yêu cầu phi trạng thái và các primitive của máy chủ](../assets/mcp-architecture.svg)

### Ba primitive của máy chủ

1. **Công cụ** là hành động có thể gọi. Mỗi công cụ có tên, mô tả, đầu vào JSON Schema và handler.
2. **Tài nguyên** là nội dung có tên, được định địa chỉ bằng URI, mà máy khách có thể đọc.
3. **Prompt** là mẫu tái sử dụng mà host có thể đưa ra cho người dùng.

Host là ứng dụng AI. Một máy khách MCP bên trong host đó nói chuyện với một máy chủ. Kênh truyền chở các thông điệp JSON-RPC giữa hai bên.

### Yêu cầu phi trạng thái thay thế bắt tay

MCP 2026-07-28 loại bỏ `initialize` và `notifications/initialized`. Nó cũng loại bỏ session ở tầng giao thức. Mỗi yêu cầu mang theo ngữ cảnh cần thiết để diễn giải nó trong `params._meta`:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Phiên bản giao thức và năng lực của máy khách là bắt buộc. Danh tính máy khách là khuyến nghị. Thiếu `_meta`, thiếu một trường bắt buộc, hoặc trường bắt buộc sai kiểu đều là sai định dạng và trả về Invalid Params (`-32602`). Một chuỗi phiên bản đúng định dạng nhưng máy chủ không hỗ trợ sẽ trả về `UnsupportedProtocolVersionError` (`-32022`). Máy chủ có thể xử lý một yêu cầu hợp lệ mà không cần dựng lại bản ghi thương lượng năng lực (capability negotiation) trước đó.

Phi trạng thái không có nghĩa là ứng dụng không bao giờ được giữ trạng thái. Nó có nghĩa là trạng thái đó không bị giấu sau một kết nối MCP hay `Mcp-Session-Id`. Nếu một luồng công việc cần tính liên tục, máy chủ phát ra một handle mờ và máy khách truyền handle đó như một tham số công cụ bình thường ở các lần gọi sau. Phân quyền vẫn phải được kiểm tra trên mọi yêu cầu.

### Khám phá và chọn phiên bản

Mọi máy chủ hiện đại đều hiện thực `server/discover`. Kết quả quảng bá các phiên bản được hỗ trợ, năng lực và danh tính máy chủ:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "complete",
    "supportedVersions": ["2026-07-28"],
    "capabilities": {
      "tools": {},
      "resources": {},
      "prompts": {}
    },
    "ttlMs": 3600000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "demo-server",
        "version": "1.0.0"
      }
    }
  }
}
```

Máy khách có thể gọi thẳng một phương thức khác rồi xử lý lỗi phiên bản, nhưng khám phá giúp việc hiển thị năng lực và chọn phiên bản trở nên tường minh. Một phiên bản không được hỗ trợ trả về `UnsupportedProtocolVersionError` với mã `-32022`. Dữ liệu của nó chứa `supported`, một mảng các bản sửa đổi của máy chủ, và `requested`, bản sửa đổi bị từ chối.

Trên stdio, một máy khách hai thời kỳ thăm dò bằng `server/discover`. Một kết quả khám phá hoặc một lỗi hiện đại được nhận diện như `UnsupportedProtocolVersionError` cho biết đó là máy chủ hiện đại. Bất kỳ lỗi hay timeout nào không được nhận diện là hiện đại đều cho phép quay về luồng `initialize` của 2025-11-25. Hành vi cũ là mã tương thích, không phải mặc định hiện đại.

### Kết quả là tường minh

Mọi kết quả lõi của 2026-07-28 đều có `resultType`:

- `complete` nghĩa là thao tác đã xong.
- `input_required` nghĩa là máy chủ cần thêm một vòng đi-về theo mẫu Multi Round-Trip Requests. Máy chủ lõi chỉ có thể trả nó từ `tools/call`, `resources/read` hoặc `prompts/get`.

Máy khách phải coi một kết quả cũ thiếu `resultType` là complete.

Máy chủ nên đưa `io.modelcontextprotocol/serverInfo` vào `_meta` của mọi kết quả. Danh tính này do chính máy chủ tự khai và chỉ dùng để hiển thị, ghi log và gỡ lỗi, không dùng cho quyết định bảo mật.

Kết quả dạng list và read cũng mang `ttlMs` và `cacheScope`. Thứ tự `tools/list` xác định cộng với gợi ý độ tươi cho phép máy khách cache phần khám phá một cách an toàn và giúp prompt-cache ổn định hơn. `cacheScope: public` cho phép cache dùng chung; `private` giới hạn việc tái dùng trong ngữ cảnh gọi.

### Định dạng truyền và kênh truyền

MCP dùng JSON-RPC 2.0 qua stdio hoặc Streamable HTTP.

- Một yêu cầu có `jsonrpc`, `id`, `method` và `params`.
- Một phản hồi có `id` tương ứng và `result` hoặc `error`.
- Một notification không có `id` và không chờ phản hồi.

Streamable HTTP hiện đại mở một endpoint duy nhất nhận POST. Mỗi thông điệp JSON-RPC có POST riêng. Một POST yêu cầu nhận về hoặc một đối tượng JSON, hoặc một luồng Server-Sent Events gắn với yêu cầu và kết thúc bằng phản hồi cuối. Một POST notification được chấp nhận sẽ nhận HTTP 202 không có phần thân; bản sửa đổi lõi này không định nghĩa notification từ máy khách sang máy chủ qua Streamable HTTP.

Trong 2026-07-28 không có luồng GET MCP độc lập, endpoint DELETE session, `Mcp-Session-Id`, hay phát lại `Last-Event-ID`. Notification thay đổi dài hạn dùng một POST `subscriptions/listen` mà phản hồi vẫn mở như một luồng SSE.

### Đầu vào phía máy khách không cần yêu cầu do máy chủ khởi tạo

Các bản sửa đổi cũ cho phép máy chủ gửi những yêu cầu như `sampling/createMessage`, `roots/list` hay `elicitation/create` qua một luồng. Giao thức hiện tại dùng Multi Round-Trip Requests thay thế. Một lần gọi công cụ, đọc tài nguyên hoặc lấy prompt đủ điều kiện sẽ trả về `resultType: input_required` kèm ít nhất một trong `inputRequests` hoặc `requestState`. Máy khách thu thập mọi đầu vào được yêu cầu, gọi lại phương thức gốc với ID JSON-RPC mới và `inputResponses` tương ứng, rồi phản chiếu đúng `requestState` khi có được cung cấp. Nếu không có `inputRequests` nào, lần gọi lại bỏ `inputResponses`.

Roots, Sampling và Logging vẫn hoạt động nhưng đã bị đánh dấu lỗi thời, nên các hiện thực mới không nên dùng. Các yêu cầu Roots hay Sampling hiện có sẽ đi bên trong `inputRequests` của MRTR, không bao giờ là yêu cầu JSON-RPC độc lập từ máy chủ sang máy khách. Hãy ưu tiên tham số tệp hoặc thư mục tường minh, URI tài nguyên, cấu hình máy chủ và tích hợp trực tiếp với nhà cung cấp model. Dùng stderr cho chẩn đoán trên stdio và OpenTelemetry cho telemetry production.

```figure
mcp-nxm-collapse
```

## Tự tay làm

### Bước 1: đăng ký bề mặt máy chủ

Việc đăng ký vẫn đơn giản dù hợp đồng yêu cầu đã thay đổi:

```python
server = MCPServer("demo-server")

@server.tool(
    "add",
    "Add two integers.",
    {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"}
        },
        "required": ["a", "b"]
    }
)
def add(a: int, b: int) -> dict:
    return {"sum": a + b}
```

Hiện thực đi kèm trong `code/main.py` cũng đăng ký một tài nguyên và một prompt. Nó cố ý dùng thư viện chuẩn để bạn thấy từng envelope thay vì giao phó giao thức cho SDK.

### Bước 2: gắn metadata vào mọi yêu cầu

```python
def request(method, params=None):
    body_params = dict(params or {})
    body_params["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {
            "name": "demo-client",
            "version": "1.0.0"
        }
    }
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": body_params
    }
```

Đừng chỉ cache metadata này trong một đối tượng kết nối. Máy chủ kiểm tra nó trên từng yêu cầu.

### Bước 3: tuỳ chọn khám phá trước khi liệt kê

Gọi `server/discover`, chọn một phiên bản được hỗ trợ, rồi gọi `tools/list`. Gọi thẳng `tools/list` cũng hợp lệ nếu bạn đã biết phiên bản và xử lý được `-32022`.

Bản demo trả danh sách công cụ theo thứ tự tên và gắn `ttlMs`, `cacheScope`, `resultType` cùng danh tính máy chủ. Một lần gọi công cụ trả kết quả complete và không cache được vì đầu ra của nó có thể phụ thuộc trạng thái hiện tại.

### Bước 4: ánh xạ cùng yêu cầu đó sang HTTP

Một POST `tools/call` từ xa có các header phản chiếu phần thân JSON-RPC:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
Accept: application/json, text/event-stream
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: add
```

Header `MCP-Protocol-Version` phải khớp phiên bản trong `_meta`. `Mcp-Method` là bắt buộc trên mọi yêu cầu JSON-RPC và phải khớp `method`. `Mcp-Name` chỉ bắt buộc với `tools/call`, `resources/read` và `prompts/get`, ở đó nó phải khớp tên công cụ, URI tài nguyên hoặc tên prompt. Thiếu header bắt buộc hoặc không khớp sẽ trả HTTP 400 với mã `HeaderMismatch` là `-32020`.

### Bước 5: siết an toàn bên ngoài trạng thái giao thức

- Kiểm tra phân quyền và audience trên mọi yêu cầu HTTP.
- Gắn máy chủ cục bộ vào localhost và kiểm tra `Origin` trên Streamable HTTP.
- Đánh dấu công cụ có thay đổi dữ liệu bằng `destructiveHint: true` và yêu cầu host phê duyệt.
- Truyền phạm vi thư mục và tệp một cách tường minh thay vì phụ thuộc vào Roots đã lỗi thời.
- Coi tài nguyên và đầu ra công cụ là dữ liệu không đáng tin.
- Giữ stdout dành riêng cho JSON-RPC khi dùng stdio; ghi chẩn đoán ra stderr.

## Áp dụng

Chạy bài học từ thư mục của nó:

```bash
python3 code/main.py
cd code
python3 -m unittest discover tests -v
```

Dòng đầu tiên phải báo đã khám phá `demo-server` ở giao thức `2026-07-28`. Sau đó xem `MCPClient.request`: nó dựng lại `_meta` cho mỗi lần gọi. Hãy xoá metadata khỏi một yêu cầu và quan sát máy chủ từ chối nó.

## Đóng gói

`outputs/skill-mcp-server-designer.md` biến một miền nghiệp vụ thành thiết kế MCP phi trạng thái. Cổng nghiệm thu của nó yêu cầu một kết quả khám phá, chính sách metadata theo từng yêu cầu, danh sách xác định và biết cache, handle trạng thái tường minh, header kênh truyền, phân quyền và quy tắc phê duyệt.

## Đi sâu hơn vào MCP

Bài học này cho bạn mô hình giao thức. Giai đoạn 13 biến bốn ranh giới production thành những bài học tự xây rồi tự kiểm riêng:

1. [Hợp đồng công cụ MCP và nội dung](../../../13-tools-and-protocols/28-mcp-tool-contracts-and-content/docs/en.md) đề cập schema đầu vào đóng, nội dung có cấu trúc, metadata định tuyến, phân trang mờ, phân quyền completion, và khác biệt giữa lỗi giao thức và lỗi miền công cụ.
2. [Độ tin cậy, huỷ bỏ và điều khiển luồng của MCP](../../../13-tools-and-protocols/29-mcp-reliability-cancellation-and-flow-control/docs/en.md) đề cập huỷ yêu cầu, huỷ tác vụ bền vững, deadline, idempotency, backpressure, đệm qua proxy và hành vi kết nối lại.
3. [Chuỗi cung ứng, kết nạp, trôi lệch và rollback của MCP Registry](../../../13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/docs/en.md) đề cập chứng minh namespace, nguồn gốc artifact, ghim bất biến, trôi lệch trực tiếp, trạng thái Registry, bằng chứng kết nạp và rollback.
4. [Kỹ nghệ tuân thủ MCP](../../../13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/docs/en.md) đề cập bản ghi truyền vàng và âm, các thời kỳ phiên bản nghiêm ngặt, so sánh SDK, bằng chứng proxy, che dữ liệu, cổng sức khoẻ và rollback phát hành.

Hãy theo đúng thứ tự khi máy chủ sẽ đi qua ranh giới đội nhóm hoặc ranh giới tin cậy. Cùng lại, chúng đưa bạn từ “phương thức chạy được” tới “hợp đồng vẫn an toàn và chẩn đoán được qua mọi lần triển khai.”

## Bài tập

1. Thêm công cụ `subtract` và xác nhận `tools/list` vẫn giữ thứ tự alphabet.
2. Xoá khoá phiên bản giao thức và kiểm tra Invalid Params (`-32602`). Sau đó gửi phiên bản đúng định dạng nhưng không được hỗ trợ `2025-11-25`, kiểm tra `-32022`, xác nhận `requested` phản chiếu bản sửa đổi đó, và chọn từ `supported`.
3. Thêm một `draftId` do máy chủ phát vào thao tác create, rồi bắt buộc nó như một tham số của update. Giải thích vì sao đó là trạng thái ứng dụng chứ không phải session giao thức.
4. Trả `input_required` từ một công cụ cần người dùng xác nhận. Gọi lại lần gọi gốc với ID mới, một mục `inputResponses` và đúng `requestState` thay vì tự bịa một yêu cầu JSON-RPC từ máy chủ sang máy khách.
5. Phác thảo một máy khách stdio hai thời kỳ. Coi một kết quả hoặc một lỗi hiện đại được nhận diện là hiện đại, và chỉ cho phép quay về `initialize` khi gặp lỗi không nhận diện được hoặc timeout.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| MCP | "Giao thức công cụ cho LLM" | Giao thức JSON-RPC để khám phá máy chủ, công cụ, tài nguyên, prompt và phần mở rộng |
| Host | "Ứng dụng AI" | Sở hữu model và UI, gắn một hoặc nhiều máy khách MCP |
| Client | "Cái kết nối" | Nói MCP với một máy chủ, thay mặt host |
| Stateless MCP | "Không session" | Mỗi yêu cầu mang phiên bản và năng lực; không trạng thái giao thức nào gắn với kết nối |
| `server/discover` | "Đầu dò năng lực" | Phương thức bắt buộc của máy chủ, quảng bá phiên bản, năng lực và danh tính |
| `resultType` | "Trạng thái kết quả" | Đánh dấu kết quả là `complete` hoặc `input_required` |
| State handle | "Id luồng công việc" | Định danh ứng dụng do máy chủ phát, truyền như một tham số bình thường |
| Streamable HTTP | "Kênh truyền từ xa" | Một endpoint POST với phản hồi JSON hoặc SSE gắn với yêu cầu |
| MRTR | "Hỏi rồi thử lại" | Yêu cầu đầu vào nhúng trong kết quả, sau đó thử lại thao tác gốc |

## Đọc thêm

- [Các thay đổi chính của MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [Khám phá máy chủ MCP](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Yêu cầu nhiều vòng của MCP](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [Các tính năng đã lỗi thời của MCP](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)

---

[← Bài trước: Xây dựng một Ứng dụng LLM cho Production](../../13-production-app/docs/vi.md) · [Bài sau: Prompt caching và cache ngữ cảnh →](../../15-prompt-caching/docs/vi.md)
