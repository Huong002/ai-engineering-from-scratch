# Hợp đồng công cụ và nội dung trong MCP

> Một công cụ chỉ an toàn để tự động hoá khi khám phá, tham số, kết quả, phân trang và metadata vận chuyển cùng thống nhất trên một hợp đồng công cụ (tool contract).

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 13, Bài 07, 09 và 10
**Time:** ~120 phút

## Mục tiêu học tập

- Định nghĩa đầu vào và đầu ra của công cụ bằng JSON Schema 2020-12.
- Kiểm tra kết quả có cấu trúc mà không mặc định rằng chúng là đối tượng JSON.
- Chọn giữa text, image, audio, resource link và resource nhúng.
- Từ chối các định nghĩa `x-mcp-header` không an toàn trước khi công cụ tới được model.
- Mã hoá giá trị tham số-thành-header và kiểm tra khớp chính xác giữa header và body.
- Duyệt phân trang theo cursor mà không diễn giải giá trị cursor.
- Giới hạn và cấp quyền cho các gợi ý `completion/complete`.

## Vấn đề

Gọi một hàm Python thì dễ. Gọi một năng lực từ xa qua một AI host lại là bài toán hợp đồng.

Server công bố một descriptor. Client biến descriptor đó thành ngữ cảnh cho model và giao diện người dùng. Model tạo ra các tham số. Một gateway có thể định tuyến yêu cầu dựa trên các header được phản chiếu. Server thực thi công cụ. Sau đó client quyết định kết quả có đủ an toàn và hợp lệ để trả về cho model hay không.

Chỉ một ranh giới yếu là hỏng cả chuỗi.

Xét năm kiểu thất bại:

- Descriptor nói kết quả là một object, nhưng server trả về một array.
- Client dừng phân trang khi `nextCursor` là chuỗi rỗng.
- Một tham số token bị phản chiếu vào HTTP header và trở nên lộ với các bên trung gian.
- Một giá trị định tuyến Unicode được gửi dưới dạng header thô, rồi gateway và origin hiểu các byte khác nhau.
- Một endpoint completion gợi ý môi trường production cho người gọi không có quyền truy cập nó.

Không kiểu thất bại nào ở trên được sửa bằng cách viết prompt tốt hơn. Chúng đòi hỏi hợp đồng giao thức và hợp đồng ứng dụng tường minh.

## Pipeline hợp đồng

Hãy coi mỗi lần gọi công cụ là năm cổng:

1. **Khám phá.** Đọc một danh sách công cụ có phân trang và xác định.
2. **Tiếp nhận.** Kiểm tra từng descriptor và áp dụng chính sách bảo mật cục bộ.
3. **Gọi.** Kiểm tra tham số và dựng metadata vận chuyển.
4. **Thực thi.** Chạy handler và phân loại thất bại cho đúng.
5. **Tiêu thụ.** Kiểm tra các khối nội dung (content block) và đầu ra có cấu trúc trước khi model dùng.

```figure
mcp-contract-pipeline
```

Host sở hữu hai cổng tiếp nhận và tiêu thụ. Server không thể ép client tin vào annotation, schema hay đầu ra của nó.

## JSON Schema là một ranh giới lúc chạy

Trong MCP `2026-07-28`, `inputSchema` (schema đầu vào) và `outputSchema` (schema đầu ra) dùng JSON Schema. Khi thiếu `$schema`, dialect mặc định là 2020-12.

Input schema phải là một schema object. Một công cụ không có tham số vẫn nên nói rõ chính xác nó chấp nhận gì:

```json
{
  "type": "object",
  "additionalProperties": false
}
```

Cách này chặt hơn `{ "type": "object" }`, vốn chấp nhận mọi thuộc tính tuỳ ý.

Output schema là tuỳ chọn. Một khi server đã công bố nó, mọi kết quả công cụ hoàn chỉnh
cam kết trả về `structuredContent` tuân thủ schema, kể cả các kết quả
có `isError: true`. Cờ lỗi phân loại kết cục thực thi; nó không
miễn trừ hợp đồng đầu ra đã công bố. Client nên kiểm tra kết quả thay vì
tin vào descriptor.

### Nội dung có cấu trúc (structured content) là một giá trị JSON bất kỳ

Đừng hard-code `structuredContent` là một dictionary. Nó có thể là:

- một object;
- một array;
- một chuỗi;
- một số;
- một boolean;
- `null`.

Công cụ này trả về một array:

```json
{
  "name": "tag_catalog",
  "inputSchema": {
    "type": "object",
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "array",
    "items": {"type": "string"}
  }
}
```

Kết quả thành công của nó là hợp lệ:

```json
{
  "resultType": "complete",
  "content": [
    {
      "type": "text",
      "text": "[\"contracts\", \"mcp\", \"stateless\"]"
    }
  ],
  "structuredContent": ["contracts", "mcp", "stateless"],
  "isError": false
}
```

Để tương thích, kết quả có cấu trúc cũng nên kèm JSON đã tuần tự hoá trong một text block. Phần text không phải nguồn để kiểm tra. `structuredContent` mới là.

### Một validator nhỏ vẫn dạy được ranh giới

Bài học dùng một tập con JSON Schema có chủ đích, vì nó chỉ nằm trong thư viện chuẩn Python. Nó kiểm tra các cơ chế mà những công cụ mẫu dùng:

- các kiểu object, array, string, integer, number, boolean và null;
- thuộc tính bắt buộc;
- `additionalProperties: false`;
- phần tử của array;
- giá trị enum;
- độ dài tối thiểu của chuỗi.

Đây không phải thứ thay thế một validator hoàn chỉnh cho production. Bài học tái sử dụng được nằm ở chỗ việc kiểm tra diễn ra khi nào: sau khi khám phá với descriptor, trước khi thực thi với tham số, và trước khi tiêu thụ với kết quả có cấu trúc.

## Các khối nội dung có chi phí khác nhau

Mảng `content` có thể kết hợp nhiều loại nội dung.

| Loại | Dùng cho | Ranh giới chính |
|------|------------|---------------|
| `text` | Tóm tắt cho người và model đọc được | Coi text là đầu ra không đáng tin |
| `image` | Bằng chứng hình ảnh mã hoá base64 | Kiểm tra media type và kích thước |
| `audio` | Đầu ra nói hoặc ghi âm mã hoá base64 | Kiểm tra media type và giới hạn thời lượng |
| `resource_link` | Một URI mà client có thể tải sau đó | Cấp lại quyền cho lần đọc tài nguyên sau |
| `resource` | Dữ liệu nhúng trực tiếp trong kết quả | Thực thi giới hạn payload và nội dung ngay lúc này |

Một resource link không chứng minh rằng tài nguyên đó xuất hiện trong `resources/list`. Nó chỉ là một tham chiếu do lần gọi công cụ này trả về. Client vẫn áp dụng chính sách tài nguyên của mình khi đi theo URI.

Tài nguyên nhúng tránh thêm một vòng gọi nhưng làm tăng kích thước phản hồi hiện tại. Dùng link cho artifact lớn hoặc thay đổi độc lập. Dùng tài nguyên nhúng cho bằng chứng nhỏ phải đi kèm nguyên khối với kết quả.

Kết quả `evidence_bundle` của bài học bao gồm cả năm loại. Client kiểm tra từng khối trước khi chấp nhận kết quả.

## `x-mcp-header` là metadata định tuyến

Một thuộc tính bên trong `inputSchema` có thể khai báo `x-mcp-header`. Qua Streamable HTTP, client phản chiếu tham số đó vào `Mcp-Param-{name}`.

```json
{
  "region": {
    "type": "string",
    "x-mcp-header": "Region"
  }
}
```

Với `region: "eu-west"`, lớp vận chuyển có thể phát ra:

```http
Mcp-Param-Region: eu-west
```

Annotation tồn tại để bộ cân bằng tải, gateway hoặc policy engine định tuyến mà không cần phân tích JSON body. Đây không phải chỗ để đặt thông tin xác thực.

Giao thức ràng buộc annotation:

- tên header không rỗng và theo cú pháp token của HTTP field-name;
- tên header là duy nhất, không phân biệt hoa thường;
- kiểu thuộc tính là string, integer hoặc boolean;
- không cho phép `number`;
- annotation chỉ xuất hiện trên một thành viên trực tiếp của `inputSchema.properties`;
- giá trị integer nằm trong khoảng `-9007199254740991` đến `9007199254740991`.

Quy tắc vị trí là quy tắc cú pháp và đóng lại khi sai. Hãy duyệt toàn bộ cây schema,
không chỉ những thuộc tính mà validator của bạn tình cờ hiểu. Từ chối một
annotation nằm dưới `properties` của object lồng nhau, dưới nhánh `oneOf`, `items`,
một định nghĩa được `$ref` tới, hoặc bất kỳ output schema nào. Việc phân giải tham chiếu không
biến node được tham chiếu thành một thuộc tính trực tiếp ở cấp cao nhất.

Bài học này thêm một chính sách triển khai: từ chối các descriptor phản chiếu những tên như `password`, `secret`, `token`, `api_key` hay `authorization`. Đặc tả chính thức khuyên tác giả server không phản chiếu các tham số nhạy cảm. Client có thể biến lời khuyên đó thành một quy tắc tiếp nhận cứng.

Hãy audit tên header, không audit giá trị. Mã mẫu ghi lại `Mcp-Param-Region` nhưng giữ `eu-west` ra ngoài sự kiện audit.

### Mã hoá giá trị trước khi dựng HTTP header

Một giá trị tham số chỉ được đi dưới dạng plain text khi nó là chuỗi không rỗng
gồm các ký tự ASCII hiển thị từ `!` đến `~` và không giống sentinel mã hoá.
Mọi trường hợp khác dùng đúng dạng sau:

```text
=?base64?{Base64UTF8}?=
```

`Base64UTF8` là base64 chuẩn trên đúng các byte UTF-8. Đừng trim,
normalize hay thay thế giá trị trước. Hãy mã hoá Unicode, chuỗi rỗng, dấu cách,
tab, ký tự điều khiển, CR hoặc LF, khoảng trắng ở đầu hay cuối, và mọi
giá trị bắt đầu bằng `=?base64?`. Mã hoá lại một giá trị trông giống sentinel chính là
cách để bên nhận khôi phục nguyên văn text gốc thay vì giải mã
nó như cú pháp vận chuyển.

Boolean được ghi là `true` hoặc `false` chữ thường. Integer được ghi theo cơ số 10 và
phải nằm trong khoảng integer an toàn của JavaScript. Giá trị ngoài khoảng đó
bị từ chối thay vì bị một bên trung gian làm tròn.

### Server kiểm tra bản sao được phản chiếu

Việc sinh header chỉ là nửa phần của client. Tại ranh giới Streamable HTTP,
server phải:

1. tìm các tên `Mcp-Param-*` đã nhận diện mà không phân biệt hoa thường của tên header;
2. giải mã đúng dạng sentinel base64 khi có;
3. so sánh chính xác text đã giải mã với tham số JSON body tương ứng;
4. từ chối header đã nhận diện mà bị thiếu, trùng lặp, không mong đợi, sai
   định dạng hoặc không khớp trước khi dispatch.

Việc từ chối là HTTP `400` với mã lỗi JSON-RPC `-32020`. Cả giá trị body
lẫn dạng header đã mã hoá của nó đều không thuộc về bản ghi audit. Chỉ ghi
tên header đã nhận diện và loại từ chối.

`code/main.py` mô hình hoá trực tiếp ranh giới này. [Bài 09](../../09-mcp-transports/)
bao quát thứ tự kiểm tra Streamable HTTP rộng hơn, gồm cả kiểm tra khớp
method và protocol-version.

## Cursor phân trang là bất minh

Các thao tác list của MCP dùng phân trang theo cursor. Server chọn kích thước trang và định dạng cursor. Client chỉ có một quyết định:

```python
if result.get("nextCursor") is None:
    break
cursor = result["nextCursor"]
```

Đừng viết thế này:

```python
if not result.get("nextCursor"):
    break
```

Chuỗi rỗng là một cursor hợp lệ. Kiểm tra theo truthiness sẽ dừng quá sớm.

Client không được giải mã cursor, tăng giá trị nó, so sánh nó với cursor trước để xếp thứ tự, hay suy ra số trang. Server có thể ký cursor, gắn nó với một version catalog, hoặc ánh xạ nó tới trạng thái riêng. Đó là chi tiết cài đặt của server.

Server mẫu cố tình trả về `""` sau trang đầu. Client phải gửi đúng giá trị đó ở yêu cầu thứ hai. Dấu vết của nó là:

```text
<first request with no cursor>
<second request with cursor "">
```

Cursor không hợp lệ tạo ra JSON-RPC invalid params, mã `-32602`.

## Completion là một bề mặt cấp quyền

`completion/complete` cung cấp gợi ý cho tham số của prompt và tham số của resource-template. Nó hữu ích cho form tương tác, nhưng có thể làm lộ những tên mà các phương thức list thông thường vẫn bảo vệ.

Một yêu cầu completion nêu ra một tham chiếu và tham số đang được hoàn thành:

```json
{
  "method": "completion/complete",
  "params": {
    "ref": {
      "type": "ref/prompt",
      "name": "deployment_review"
    },
    "argument": {
      "name": "environment",
      "value": "st"
    }
  }
}
```

Kết quả trả về tối đa 100 giá trị và có thể báo `total` cùng `hasMore`.

Hãy áp dụng đúng ranh giới cấp quyền mà prompt hoặc tài nguyên được tham chiếu đang dùng. Một analyst trong ví dụ mẫu nhận được `development` và `staging`. Chỉ operator mới có thể nhận `production`.

Completion cho production cũng cần:

- kiểm tra đầu vào;
- lọc theo người gọi;
- debounce yêu cầu ở phía client;
- giới hạn tần suất ở phía server;
- số lượng kết quả có giới hạn;
- log không để lộ các giá trị gợi ý nhạy cảm.

Completion là trợ giúp, không phải cách vòng qua bước khám phá.

## Hai tầng lỗi

Hãy tách lỗi giao thức khỏi lỗi thực thi công cụ trong hợp đồng lỗi (error contract).

Dùng lỗi JSON-RPC khi yêu cầu MCP không thể được dispatch cho đúng:

- tên công cụ không xác định;
- dạng yêu cầu sai;
- thiếu metadata của yêu cầu;
- cursor không hợp lệ.

Dùng một kết quả công cụ hoàn chỉnh với `isError: true` khi lời gọi đã tới được công cụ và công cụ báo một thất bại có thể xử lý:

- một nguồn báo cáo không sẵn sàng;
- một ngày nằm ngoài khoảng được hỗ trợ;
- một quy tắc nghiệp vụ từ chối thao tác được yêu cầu.

Model thường có thể sửa một lỗi thực thi công cụ. Model không thể sửa một server đã vi phạm chính output schema của nó.

Nếu công cụ khai báo output schema, hãy mô hình hoá thất bại có thể xử lý ngay trong
schema đó. Thất bại của `route_report` trong ví dụ mẫu trả về đúng region được yêu cầu với
`accepted: false`, kèm text lỗi cho người đọc và `isError: true`.

## Tự tay làm

`code/main.py` dựng cả hai phía của ranh giới bằng thư viện chuẩn Python.

Server cài đặt:

- kiểm tra metadata MCP cho từng yêu cầu;
- `server/discover` với capability tools và completions;
- phân trang `tools/list` có tính xác định;
- bốn descriptor công cụ, trong đó một cái phải bị từ chối;
- đầu ra có cấu trúc dạng array;
- mọi loại khối nội dung công cụ hiện hành;
- một cổng kiểm tra khớp của Streamable HTTP, giải mã các header tham số đã nhận diện và
  trả về HTTP `400` cộng JSON-RPC `-32020` khi không khớp;
- completion có cấp quyền và giới hạn tần suất.

Client cài đặt:

- tiếp nhận descriptor;
- kiểm tra vị trí `x-mcp-header` trên toàn cây và chính sách trường nhạy cảm;
- mã hoá giá trị chính xác theo plain-visible-ASCII hoặc base64 UTF-8;
- vòng lặp cursor bất minh, vẫn đi tiếp với chuỗi rỗng;
- kiểm tra tham số và kết quả;
- kiểm tra khối nội dung;
- sự kiện audit header chứa tên nhưng không chứa giá trị.

Descriptor cố tình không an toàn là dữ liệu dạy học. Nó chứng minh rằng một công cụ bị từ chối không ngăn các công cụ hợp lệ được nạp.

## Áp dụng

Từ thư mục gốc của repo:

```bash
cd phases/13-tools-and-protocols/28-mcp-tool-contracts-and-content/code
python3 main.py
python3 -m unittest discover tests -v
```

Bản demo in ra các công cụ được tiếp nhận, descriptor bị từ chối, cả hai yêu cầu
phân trang, nội dung array có cấu trúc, các loại khối nội dung, tên header được
phản chiếu, việc giá trị có cần mã hoá hay không, trạng thái khớp HTTP, và
các giá trị completion đã lọc theo người gọi.

## Lab tương tác

Mở `code/main.py` và tìm `TOOLS`.

1. Đổi `tag_catalog.outputSchema.type` từ `array` sang `object`.
2. Chạy demo. Client sẽ phải từ chối array được trả về.
3. Khôi phục schema.
4. Giữ `nextCursor` của trang đầu là `""`, rồi cho trang cuối trả về
   `nextCursor: None` thay vì bỏ hẳn trường đó.
5. Chạy test và so sánh dấu vết cursor.
6. Thêm `x-mcp-header: "Authorization"` vào một thuộc tính string.
7. Xác nhận bước tiếp nhận descriptor từ chối nó trước khi gọi.
8. Thử các giá trị `region` chứa Unicode, một newline, dấu cách hai bên, và
   text đúng nguyên `=?base64?SGVsbG8=?=`. Giải mã từng header được phát ra và chứng minh
   giá trị gốc sống sót chính xác.
9. Chuyển annotation xuống dưới `oneOf`, `items`, hoặc một định nghĩa `$ref`. Xác nhận
   từng descriptor bị từ chối kể cả khi nhánh đó không bao giờ được demo dùng.
10. Xoá header đã nhận diện hoặc đổi giá trị đã giải mã của nó. Xác nhận ranh giới HTTP
    trả về status `400` và mã JSON-RPC `-32020`.

Điểm cốt lõi không phải là học thuộc một dạng JSON. Mà là quan sát từng cổng thất bại ngay tại ranh giới sở hữu nó.

## Lab thực hành

Mở rộng lab hợp đồng với một công cụ `search_evidence`.

Yêu cầu:

1. Input schema của nó chấp nhận `query`, `limit`, và một trường định tuyến `region` an toàn.
2. Output schema của nó là một array các object với `uri`, `title` và `score`.
3. Kết quả có text tương thích và một resource link cho mỗi phần tử.
4. Tham số từ chối các thuộc tính không xác định.
5. `limit` bị giới hạn bởi kiểm tra của ứng dụng.
6. Người gọi không có quyền với một URI sẽ không bao giờ thấy URI đó qua completion hay đầu ra công cụ.
7. Test gồm một score không tuân thủ, một annotation header không hợp lệ, và một danh sách hai trang.
8. Test giá trị header bao phủ ASCII hiển thị, Unicode, ký tự điều khiển,
   khoảng trắng, text trông giống sentinel, và cả hai biên integer an toàn của JavaScript.
9. Fixture HTTP chấp nhận tên header không phân biệt hoa thường nhưng từ chối giá trị
   đã nhận diện bị thiếu hoặc không khớp với status `400` và mã `-32020`.

## Sản phẩm bàn giao

`outputs/skill-mcp-contract-reviewer.md` là một skill review phẳng, tái sử dụng được. Đưa cho nó một descriptor công cụ, các kết quả mẫu, hành vi phân trang và chính sách completion. Nó trả về quyết định tiếp nhận, kế hoạch kiểm tra kết quả, chính sách header và các test thất bại cụ thể.

## Kiểm chứng

Bài học hoàn tất khi những phát biểu sau là đúng:

- `tools/list` trả về cùng một thứ tự logic qua những lần gọi lặp lại.
- Client thực hiện yêu cầu thứ hai khi `nextCursor` là `""`.
- Descriptor có header nhạy cảm không an toàn bị loại trong khi các công cụ khác vẫn sẵn sàng.
- Một array đi qua output schema dạng array của nó.
- Một object thất bại với chính schema array đó.
- Kết quả lỗi không được bỏ qua hay vi phạm một output schema đã công bố.
- Các khối text, image, audio, resource link và resource nhúng đều kiểm tra được.
- Sự kiện audit header có tên và không có giá trị.
- ASCII hiển thị thường vẫn giữ nguyên; giá trị Unicode, điều khiển, có đệm, rỗng và
  trông giống sentinel đi vòng qua mã hoá base64 UTF-8 chính xác.
- Các integer được phản chiếu nằm ngoài khoảng an toàn của JavaScript bị từ chối.
- Annotation dưới `oneOf`, `items`, object lồng nhau, định nghĩa `$ref`, hoặc
  output schema đều bị từ chối trong bước tiếp nhận.
- Tên header đã nhận diện không phân biệt hoa thường chỉ đi qua khi giá trị đã giải mã
  khớp chính xác với body; bản sao bị thiếu hoặc không khớp tạo ra HTTP `400`
  và JSON-RPC `-32020`.
- Completion cho analyst không bao giờ trả về `production`.
- Một thất bại của công cụ dùng `isError: true`; một lời gọi giao thức sai dùng `error` của JSON-RPC.

## Các kiểu thất bại trong production

| Thất bại | Người học thấy gì | Phản ứng đúng |
|---------|-----------------------|------------------|
| Client mặc định đầu ra là object | Các array hợp lệ thất bại hoặc bị bọc một cách âm thầm | Kiểm tra theo schema đã công bố mà không chỉ dùng kiểu object |
| Cursor rỗng bị coi là false | Những trang cuối biến mất | Tiếp tục mỗi khi `nextCursor` có mặt và khác null |
| Giá trị nhạy cảm bị phản chiếu | Bí mật xuất hiện trong proxy, WAF hoặc dữ liệu trace | Từ chối descriptor và giữ bí mật trong dữ liệu yêu cầu được bảo vệ |
| Unicode thô hoặc khoảng trắng bị phản chiếu | Gateway và origin bất đồng hoặc giá trị bị normalize | Dùng mã hoá sentinel base64 UTF-8 chính xác và so sánh sau khi giải mã |
| Annotation ẩn trong một nhánh schema | Client bỏ sót metadata định tuyến khi tiếp nhận | Duyệt toàn bộ cây schema và chỉ cho phép thuộc tính trực tiếp ở cấp cao nhất |
| Integer lớn bị phản chiếu | Bên trung gian JavaScript làm tròn giá trị định tuyến | Từ chối giá trị ngoài khoảng integer an toàn của JavaScript |
| Header và body không khớp | Gateway định tuyến một đích trong khi origin thực thi đích khác | Từ chối trước khi dispatch với HTTP `400` và JSON-RPC `-32020` |
| Output schema bị bỏ qua | Code phía sau tiêu thụ cấu trúc hỏng | Kiểm tra trước khi model hoặc ứng dụng dùng |
| Resource link được tin tự động | Người gọi đi theo một URI không có quyền | Cấp lại quyền cho mọi lần đọc tài nguyên |
| Completion chia sẻ gợi ý toàn cục | Tên tenant bị ẩn lộ ra | Lọc theo người gọi, tham chiếu và quyền |
| Annotation công cụ bị coi là chính sách | Thao tác phá huỷ vượt qua bước xác nhận | Thực thi cấp quyền và phê duyệt bên ngoài annotation |
| Một công cụ sai làm hỏng khám phá | Toàn bộ server trở nên không khả dụng | Từ chối descriptor xấu và tiếp nhận các công cụ hợp lệ một cách độc lập |

## Kết nối Capstone

Capstone của Giai đoạn 13 cần một gateway có thể gộp công cụ từ nhiều server. Bài học này cung cấp phần lõi tiếp nhận cho nó.

Hãy dùng artifact để chấm bốn mảnh bằng chứng của capstone:

- khám phá có phân trang đầy đủ và xác định;
- kiểm tra descriptor trước khi phơi bày cho model;
- đầu ra có cấu trúc đã kiểm tra cùng các khối nội dung có giới hạn;
- metadata completion và định tuyến giữ nguyên các ranh giới cấp quyền.

Đừng tuyên bố tương thích gateway chỉ từ một lần `tools/call` thành công. Hãy lưu lại descriptor, dấu vết phân trang, tập công cụ được tiếp nhận, tập công cụ bị từ chối, và một kết quả đã kiểm tra.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa thực sự |
|------|---------|
| `inputSchema` | Đối tượng JSON Schema định nghĩa các tham số mà công cụ chấp nhận |
| `outputSchema` | JSON Schema tuỳ chọn định nghĩa `structuredContent` |
| `structuredContent` | Bất kỳ giá trị JSON nào do một kết quả công cụ tạo ra |
| Content block | Text, image, audio, resource link hoặc resource nhúng có kiểu |
| `x-mcp-header` | Annotation schema phản chiếu một tham số nguyên thuỷ vào metadata Streamable HTTP |
| Opaque cursor | Token phân trang do server phát, client không diễn giải giá trị |
| Completion reference | Tên prompt hoặc URI/template tài nguyên đang được hoàn thành tham số |
| Admission | Quyết định của client về việc phơi bày hay từ chối một descriptor đã khám phá |

## Đọc thêm

- [Công cụ MCP](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [Completion trong MCP](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/completion)
- [Phân trang trong MCP](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/pagination)
- [Header tham số trong Streamable HTTP của MCP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http#custom-headers-from-tool-parameters)

---

[← Bài trước: Eval kỹ năng, đóng gói và tính di động](../../27-skill-evals-packaging-and-portability/docs/vi.md) · [Bài sau: MCP: độ tin cậy (reliability), huỷ (cancellation) và điều tiết luồng (flow control) →](../../29-mcp-reliability-cancellation-and-flow-control/docs/vi.md)
