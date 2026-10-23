# Chuỗi cung ứng MCP Registry: tiếp nhận, lệch trạng thái và rollback

> Một mục trong registry cho bạn biết nhà phát hành đã khai báo gì. Bước tiếp nhận (admission) trong môi trường chạy thật chứng minh bạn đã tải về cái gì, bạn đã quan sát thấy gì, bạn đã phê duyệt cái gì, và bạn có thể khôi phục an toàn cái gì.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 13, Bài 17 (gateway và sổ đăng ký), Giai đoạn 13, Bài 18 (xác thực trong môi trường chạy thật)
**Time:** ~90 phút

## Mục tiêu học tập

- Tách bạch việc công bố trên Registry, nguồn gốc xuất xứ (provenance) của gói, việc khám phá lúc chạy và việc phê duyệt cục bộ.
- Xác minh namespace của một MCP server mà không tin vào cái tên nằm trong chính bản ghi của nó.
- Ghim (pinning) bằng chứng bất biến về bản công bố, nguồn thực thi, nguồn gốc xuất xứ và descriptor đang chạy.
- Phát hiện thay đổi trạng thái registry và lệch trạng thái (drift) lúc chạy sau khi đã tiếp nhận.
- Rollback định tuyến về một phiên bản đã được tiếp nhận trước đó mà không viết lại lịch sử.
- Duy trì một sổ tiếp nhận có khả năng phát hiện sửa đổi, giải thích được từng quyết định.

## Vấn đề

Bạn tìm thấy `com.example/inventory` trong một registry. Mô tả của nó có vẻ đúng. Gói của nó tồn tại. Server trả lời `server/discover`.

Đó không phải là một sự thật duy nhất. Đó là một chuỗi sự thật đến từ những cơ quan có thẩm quyền khác nhau:

1. Một nhà phát hành đã xác thực cho một namespace nộp một bản ghi.
2. Một package registry phục vụ một artifact với danh tính và digest xác định.
3. Một endpoint đang chạy báo cáo phiên bản giao thức, các khả năng, các tool và thông tin server dùng cho chẩn đoán.
4. Tổ chức của bạn đã quyết định rằng đúng tổ hợp này được phép.

Gộp những sự thật đó thành "nó nằm trong registry, nên cứ tin nó" sẽ tạo ra một điểm mù trong chuỗi cung ứng (supply chain). Một bản công bố hợp lệ vẫn có thể bị deprecated. Một tag của gói có thể trỏ tới một artifact bất ngờ nếu bạn không ghim digest của nó. Một server có thể thêm một tool phá hoại sau khi đã review. Một rollback có thể lặng lẽ chọn một phiên bản chưa từng được tiếp nhận.

Cách khắc phục là một admission controller có bằng chứng ở mọi ranh giới.

## Registry là một chỉ mục, không phải hệ thống phê duyệt của bạn

Official MCP Registry lưu metadata của server. Bản ghi `server.json` của nó nêu tên một phiên bản server và khai báo một hoặc nhiều gói hay endpoint từ xa. Các quy tắc công bố bổ sung xác thực namespace, kiểm tra quyền sở hữu gói, quy tắc registry hạn chế, và một chỗ hẹp để nhà phát hành đặt metadata.

Những kiểm soát đó trả lời các câu hỏi về công bố. Chính sách môi trường chạy thật của bạn vẫn phải trả lời các câu hỏi về triển khai:

| Ranh giới | Câu hỏi | Chủ sở hữu bằng chứng |
|---|---|---|
| Namespace | Nhà phát hành có được phép dùng cái tên này không? | Xác thực của Registry cộng với đầu vào namespace đã xác minh của bạn |
| Bản ghi | Nhà phát hành đã khai báo gì cho phiên bản này? | Digest `server.json` bất biến |
| Nguồn thực thi | Gói hay endpoint từ xa nào sẽ thực thi? | Các trường nguồn đã khai báo, kết quả xác minh quyền sở hữu, transport và digest đáng tin |
| Lúc chạy | Endpoint đang phơi ra cái gì? | `server/discover` và các descriptor của tool |
| Tiếp nhận | Chính sách của bạn có phê duyệt đúng tập này không? | Pin cục bộ và mục trong sổ |
| Vận hành | Nó còn an toàn không, và cái gì có thể thay thế nó? | Kiểm tra lệch trạng thái, đồng bộ trạng thái, health và tuyến rollback |

Phiên bản schema của Registry và phiên bản giao thức MCP độc lập với nhau. Một bản ghi có thể dùng schema server `2025-12-11` đã công bố trong khi server đang chạy hỗ trợ MCP `2026-07-28`. Đừng bao giờ suy ra cái này từ cái kia.

```figure
mcp-registry-admission
```

## Bảy kiểm soát trong một quyết định tiếp nhận

### 1. Xác minh namespace

Tên trên Official Registry dùng namespace đã xác thực. Một domain đã xác minh có thể ánh xạ thành tiền tố domain đảo ngược. Ví dụ, quyền kiểm soát `example.com` có thể thiết lập `com.example/*`.

Đừng chấp nhận một phép kiểm tra tiền tố chuỗi:

```python
server_name.startswith("com.example")
```

Phép kiểm tra đó cũng chấp nhận `com.exampleevil/tool`. Hãy tách tên tại `/`, yêu cầu slug không rỗng, và so sánh đoạn namespace một cách chính xác. Quan trọng hơn, hãy truyền namespace đã xác minh vào bước tiếp nhận từ kết quả xác thực. Đừng suy ra niềm tin từ bản ghi không đáng tin.

Namespace dựa trên GitHub và namespace dựa trên domain dùng những đường xác thực khác nhau. Hãy chuẩn hoá cả hai đường thành một đầu vào tiếp nhận duy nhất: chuỗi namespace đã xác minh chính xác.

### 2. Kết nối nguồn gốc xuất xứ

Với một bản ghi gói, phần khai báo và artifact đã tải về phải khớp nhau trên những trường tường minh:

- loại package registry
- định danh gói
- phiên bản gói
- kết quả xác minh quyền sở hữu
- digest của artifact đã tải về

Cũng phải kiểm tra transport của gói đã khai báo. Một bản ghi chỉ có endpoint từ xa là hợp lệ và không được bị từ chối chỉ vì thiếu gói. Với nguồn từ xa, hãy kết nối URL và loại transport đã khai báo với quyền sở hữu endpoint được xác minh độc lập và một digest của bằng chứng kết nối hay triển khai đáng tin.

Mã của bài hỗ trợ cả hai loại nguồn và băm nguồn được chọn cùng với nguồn Registry, tên server, phiên bản Registry, digest bản ghi và digest bằng chứng. Digest nguồn gốc xuất xứ thu được là một con trỏ gọn tới toàn bộ tập bằng chứng. Nó không thay thế cho việc lưu giữ bằng chứng.

Đừng bao giờ chấp nhận một digest chỉ do chính artifact bạn đang muốn xác minh cung cấp. Hãy tính nó tại một ranh giới tải về đáng tin, hoặc nhận nó từ một dịch vụ gói mà bạn đã kiểm tra kết quả xác minh của dịch vụ đó.

### 3. Ghim quyết định, không chỉ ghim phiên bản

Phiên bản Registry là định danh công bố duy nhất. Metadata đã công bố là bất biến. Một bản ghi thay đổi đòi hỏi một phiên bản mới. Semantic versioning được khuyến nghị, nhưng Registry không bắt buộc nó và không chấp nhận khoảng phiên bản.

Điều này có nghĩa `^1.4` không phải một pin tiếp nhận. "latest" cũng vậy. Một pin hữu ích chứa:

```json
{
  "server": "com.example/inventory",
  "version": "1.0.0",
  "recordDigest": "...",
  "source": {"kind": "package", "registryType": "pypi"},
  "sourceDigest": "...",
  "toolsetDigest": "...",
  "provenanceDigest": "...",
  "registryStatus": "active"
}
```

Ghim nhiều lớp giúp bạn xác định ranh giới nào đã thay đổi. Digest bản ghi thay đổi trong cùng một phiên bản Registry là lỗi toàn vẹn của Registry. Digest nguồn thay đổi trong cùng một toạ độ gói hay cùng một triển khai từ xa là lỗi toàn vẹn của nguồn thực thi. Digest toolset thay đổi là lệch trạng thái lúc chạy.

### 4. Phát hiện lệch trạng thái trực tiếp

Bước tiếp nhận nên quan sát chính server sẽ nhận lưu lượng. Gọi `server/discover`, liệt kê hoặc lấy các descriptor tool được phơi ra qua đường đáng tin của bạn, rồi xác minh:

- `2026-07-28` nằm trong `supportedVersions`
- mọi khả năng mà cục bộ yêu cầu đều có mặt
- mọi descriptor tool có danh tính và bề mặt schema cần thiết
- digest descriptor đã chuẩn hoá khớp với pin đã tiếp nhận ở các lần kiểm tra sau

Giá trị tuỳ chọn `_meta["io.modelcontextprotocol/serverInfo"]` trong kết quả là ngữ cảnh hiển thị, log và gỡ lỗi do chính server tự khai. Hãy ghi nó như bằng chứng chẩn đoán, nhưng đừng bao giờ dùng nó để thiết lập namespace, quyền sở hữu gói, quyền sở hữu endpoint, việc tiếp nhận hay bất kỳ quyết định bảo mật nào khác. Một alias `serverInfo` trực tiếp nằm ngoài `_meta` không phải trường của hợp đồng và không nên được nâng thành bằng chứng chẩn đoán.

Chỉ chuẩn hoá những trường mà thứ tự không mang ý nghĩa. Ví dụ mẫu sắp xếp danh sách tool theo tên ổn định trước khi băm, nên một thay đổi thứ tự vô hại trong danh sách không gây ra lệch trạng thái. Nó không loại bỏ các trường của descriptor. Một tool mới, schema thay đổi, mô tả thay đổi hay annotation mới đều làm thay đổi pin.

Ví dụ mẫu coi descriptor dị dạng và mọi thay đổi digest descriptor là lệch trạng thái, cách ly pin, gỡ tuyến đang hoạt động của nó và chặn phiên bản đó làm đích rollback. Một chính sách môi trường chạy thật có thể chỉ cho phép thay đổi câu chữ thông qua một lần review mới, vì phần mô tả ảnh hưởng tới việc model chọn tool. Metadata "chỉ là hình thức" vẫn có thể làm thay đổi hành vi của agent.

### 5. Trạng thái Registry là trạng thái sống

Registry API gắn một đối tượng `_meta` ở mức phản hồi bên cạnh mỗi bản ghi server. Các trường do Registry quản lý nằm dưới `_meta["io.modelcontextprotocol.registry/official"]`. Hãy truyền đối tượng `_meta` của phản hồi vào bước tiếp nhận và đọc `_meta["io.modelcontextprotocol.registry/official"].status`. Một giá trị `_meta.status` trực tiếp không phải hình dạng chính thức trên đường truyền. Đừng nhầm metadata của phản hồi với `_meta` của chính bản ghi công bố. Trạng thái có thể là:

- `active`: được trả về mặc định và đủ điều kiện để tiếp nhận cục bộ
- `deprecated`: vẫn có thể khám phá kèm cảnh báo, nhưng không còn là lựa chọn tự động an toàn
- `deleted`: bị ẩn mặc định trong khi bản ghi lịch sử của nó vẫn truy cập được qua các view deleted hoặc incremental

Đồng bộ trạng thái sau khi tiếp nhận. Nếu một phiên bản đang active trở thành deprecated hoặc deleted, hãy cách ly pin của nó và ngừng đưa công việc mới tới đó. Giữ lại bằng chứng. Việc bị xoá khỏi danh sách mặc định không phải là quyền xoá dấu vết kiểm toán của bạn.

Metadata tuỳ chỉnh do nhà phát hành cung cấp chỉ thuộc về `_meta.io.modelcontextprotocol.registry/publisher-provided` trong một bản ghi công bố. Metadata phản hồi do Registry quản lý là tách biệt. Đừng để nhà phát hành tự đặt trạng thái chính thức cho mình.

### 6. Rollback nghĩa là khôi phục định tuyến

Một bản công bố bất biến không bị sửa trong lúc rollback. Rollback chọn một pin đã được tiếp nhận trước đó và hiện còn đủ điều kiện, rồi đổi tuyến đang hoạt động.

Một đích an toàn phải:

1. Có một bản ghi tiếp nhận đã hoàn tất.
2. Vẫn có trạng thái Registry đang active theo chính sách của bạn.
3. Không bị cách ly bởi bằng chứng lúc chạy hay bằng chứng bảo mật.
4. Vẫn phân giải tới đúng gói đã ghim và tập descriptor đang chạy.
5. Vượt qua các kiểm tra health hiện tại.

Ví dụ mẫu tập trung vào ba điều kiện đầu. Một reconciler thực thụ nên tải lại gói và kiểm tra lại endpoint đang chạy trước khi kích hoạt.

### 7. Ghi thêm vào sổ tiếp nhận

Một cơ sở dữ liệu tiếp nhận cho biết cái gì đang active. Một sổ giải thích vì sao.

Mỗi mục trong ví dụ mẫu chứa số thứ tự, thời gian, sự kiện, server, phiên bản, kết quả, các lý do, bằng chứng, hash của mục trước và hash của chính nó. Sửa một kết quả cũ sẽ làm hỏng việc kiểm chứng mục đó và mọi mắt xích về sau.

Cơ chế này phát hiện được sửa đổi, chứ không phải chống sửa đổi một cách thần kỳ. Hãy neo các đầu sổ định kỳ vào một miền tin cậy riêng, như metadata phát hành đã ký hoặc vùng lưu trữ ghi một lần. Giới hạn ai được ghi thêm. Đừng đưa token uỷ quyền, credential của gói, đối số tool hay dữ liệu endpoint riêng tư vào bằng chứng.

## Tự tay làm

Controller chạy được nằm trong `code/main.py`. Nó chỉ dùng thư viện chuẩn của Python.

Bắt đầu với phần minh hoạ hữu hạn:

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift
python3 code/main.py
```

Phần minh hoạ thực hiện năm thao tác:

1. Tiếp nhận `1.0.0` với namespace, nguồn gốc xuất xứ của gói, giao thức, các khả năng và tool khớp nhau.
2. Tiếp nhận `1.1.0` và đặt nó làm active.
3. Quan sát thấy một tool delete bất ngờ lúc chạy.
4. Quan sát trạng thái Registry của `1.1.0` chuyển thành `deprecated`.
5. Khôi phục định tuyến về pin `1.0.0` vẫn đang được tiếp nhận.

Hình dạng kết quả mong đợi:

```json
{
  "admitted": [true, true],
  "driftAllowed": false,
  "rollbackAllowed": true,
  "activeVersion": "1.0.0",
  "ledgerValid": true
}
```

Đọc phần cài đặt theo thứ tự sau:

1. `namespace_for_domain()` và `namespace_matches()` thiết lập thẩm quyền đặt tên chính xác.
2. `digest()` và `normalized_tools()` tạo ra bằng chứng tất định.
3. `RegistryAdmissionController.admit()` kết nối bản công bố, nguồn gốc xuất xứ, trạng thái lúc chạy và chính sách.
4. `check_live()` so sánh một quan sát mới với pin.
5. `observe_registry_status()` cách ly những phiên bản có trạng thái Registry thay đổi.
6. `rollback()` chỉ kích hoạt một đích đủ điều kiện đã được tiếp nhận trước đó.
7. `AdmissionLedger.verify()` phát hiện thay đổi trên lịch sử đã ghi.

## Áp dụng

Đặt controller giữa bước khám phá và bước định tuyến:

```text
Registry sync -> artifact verifier -> live discovery -> admission controller -> route table
                                               |                 |
                                               v                 v
                                          evidence store    admission ledger
```

Dùng danh tính riêng cho từng công việc. Một worker đồng bộ Registry cần quyền đọc metadata. Một trình xác minh artifact cần quyền tải gói. Một trình đối soát tuyến cần quyền kích hoạt một pin đã được phê duyệt. Không cái nào cần tất cả credential.

Hãy làm cho trạng thái triển khai trở nên tường minh. "Approved" nghĩa là bằng chứng đã vượt qua chính sách. "Active" nghĩa là tuyến hiện đang chọn nó. "Quarantined" nghĩa là nó không thể nhận công việc mới. "Superseded" nghĩa là một phiên bản đã tiếp nhận khác đang active. Đừng mã hoá cả bốn ý nghĩa đó trong một giá trị Boolean.

Hãy chạy bước tiếp nhận trước khi phơi một server trong `tools/list`. Nếu không, client có thể phát hiện một tool trong khoảng trống giữa lúc công bố và lúc chính sách được đánh giá.

## Phòng lab tương tác

Bạn sẽ xem từng ranh giới hỏng, mỗi lần một cái.

### Lab A: đụng độ namespace

Mở một Python shell từ thư mục code:

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/code
python3 -q
```

Rồi chạy:

```python
from main import namespace_matches
namespace_matches("com.example/inventory", "com.example")
namespace_matches("com.exampleevil/inventory", "com.example")
```

Kết quả đầu là `True`; kết quả thứ hai là `False`. Hãy thay phép so sánh chính xác bằng `startswith` ở máy bạn và quan sát vì sao cái tên thứ hai vượt qua ranh giới. Khôi phục lại phép so sánh chính xác trước khi tiếp tục.

### Lab B: lệch descriptor

```python
from main import *
times = iter(f"2026-08-21T12:00:{n:02d}+00:00" for n in range(10))
c = RegistryAdmissionController(clock=lambda: next(times))
meta = {OFFICIAL_META_KEY: {"status": "active"}}
c.admit(sample_record("1.0.0"), meta, "com.example", evidence_for("1.0.0"), sample_live("1.0.0"))
c.check_live("com.example/inventory", "1.0.0", sample_live("1.0.0", True))
```

Hãy xem các lý do và trạng thái tuyến. Gói và bản ghi Registry không đổi. Bề mặt tool lúc chạy thì đổi, nên controller đã cách ly và vô hiệu hoá pin. Đó là lý do việc kiểm soát chuỗi cung ứng phải tiếp tục sau khi cài đặt.

### Lab C: trạng thái và rollback

Tiếp nhận `1.1.0`, đánh dấu nó deprecated, và thử cả hai đích rollback:

```python
c.admit(sample_record("1.1.0"), meta, "com.example", evidence_for("1.1.0"), sample_live("1.1.0"))
c.observe_registry_status("com.example/inventory", "1.1.0", "deprecated")
c.rollback("com.example/inventory", "1.1.0", "unsafe retry")
c.rollback("com.example/inventory", "1.0.0", "restore known release")
c.ledger.verify()
```

Đích bị cách ly bị từ chối. Pin active trước đó được chấp nhận. Sổ vẫn hợp lệ.

## Phòng lab thực hành

Mở rộng controller với một cổng phê duyệt hai người.

Yêu cầu:

- Lưu phê duyệt dưới dạng tham chiếu bằng chứng đã ký, không phải tên có thể sửa trong pin.
- Yêu cầu hai danh tính người review khác nhau cho một toolset có chứa tool với `destructiveHint: true`.
- Từ chối các danh tính người review trùng nhau.
- Giữ lại lần thử tiếp nhận gốc trong sổ khi phê duyệt chưa đầy đủ.
- Thêm test cho trường hợp không có, một, trùng nhau và hai phê duyệt khác nhau.
- Không ghi log chữ ký, credential hay toàn bộ đối số tool riêng tư.

Thành công nghĩa là một tool phá hoại không thể trở thành active cho tới khi cả hai danh tính đã phê duyệt đúng các digest của bản ghi, gói và toolset.

## Sản phẩm đi kèm

Bài này đóng gói `outputs/skill-mcp-registry-admission.md`. Hãy dùng nó như một runbook phẳng, tái sử dụng được khi review một phiên bản Registry mới hoặc khi điều tra lệch trạng thái. Nó định nghĩa đầu vào, quy tắc từ chối, gói bằng chứng, việc đối soát trạng thái và bằng chứng rollback mà không phụ thuộc vào tên class của ví dụ mẫu.

## Kiểm chứng

Chạy phần minh hoạ và bộ test tất định:

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Việc kiểm chứng phải chứng minh:

- ranh giới namespace chính xác từ chối các tiền tố trông giống nhau
- chỉ trạng thái Registry chính thức có namespace mới có thể làm một phiên bản đủ điều kiện
- bằng chứng gói và bằng chứng từ xa chưa xác minh hoặc không khớp đều bị từ chối
- metadata của nhà phát hành không thể mạo danh metadata do Registry quản lý
- thứ tự tool được chuẩn hoá mà không che giấu thay đổi descriptor
- cấu trúc gói và tool dị dạng bị từ chối một cách an toàn
- `serverInfo` vẫn chỉ là chẩn đoán và không bao giờ cấp thẩm quyền tiếp nhận
- lệch descriptor cách ly, vô hiệu hoá và chặn rollback về pin
- thay đổi trạng thái cách ly các pin đang active
- rollback không thể chọn một phiên bản bị cách ly hoặc không rõ
- việc sửa đổi sổ bị phát hiện

## Các kiểu hỏng trong môi trường chạy thật

| Kiểu hỏng | Vì sao xảy ra | Phản ứng bắt buộc |
|---|---|---|
| Tên trông hợp lệ nhưng namespace chưa từng được xác thực | Chính sách tin vào phần chữ trong bản ghi | Từ chối cho tới khi một trình xác minh namespace đáng tin cung cấp đúng tiền tố |
| Cùng một toạ độ gói trả về byte mới | Upstream có thể thay đổi hoặc bản phân phối đã bị xâm nhập | Dừng kích hoạt, giữ lại cả hai digest, điều tra ranh giới tải về |
| "Latest" thay đổi mà không qua review | Lựa chọn trôi nổi đã thoát khỏi pin | Chỉ phân giải đúng các phiên bản và digest đã được tiếp nhận |
| Tool mới xuất hiện sau khi phê duyệt | Lệch trạng thái lúc chạy hoặc một triển khai khác | Cách ly tuyến và ghi lại một quan sát descriptor mới |
| Phiên bản deprecated vẫn active | Thiếu hoặc chậm đồng bộ trạng thái | Đối soát trạng thái theo lịch và trước khi kích hoạt |
| Bản ghi đã xoá biến mất khỏi lần đồng bộ mặc định | Client chỉ yêu cầu các bản ghi active | Dùng đối soát tăng dần hoặc có nhận biết deleted và giữ lịch sử cục bộ |
| Đích rollback chưa từng được tiếp nhận | Kiểm soát tuyến và trạng thái phê duyệt bị tách rời | Từ chối rollback và chạy một lần tiếp nhận mới cho đích đó |
| Sổ vẫn kiểm chứng được ở cục bộ sau khi kẻ tấn công viết lại mọi mục | Chuỗi hash không có neo bên ngoài | Công bố các đầu sổ đã ký sang một miền tin cậy riêng |
| Bằng chứng chứa bearer token hoặc đối số tool | Việc ghi log đã sao chép toàn bộ request | Che thông tin ngay lúc thu thập và chỉ lưu bằng chứng tối thiểu |

## Quy tắc vận hành

Công bố trả lời câu hỏi "danh tính này có được phép công bố cái tên này không?" Tiếp nhận trả lời câu hỏi "chúng ta có thực thi đúng artifact này và phơi đúng hành vi này không?" Hãy giữ hai quyết định đó tách biệt, ghim mọi điểm kết nối, và làm cho rollback chọn bằng chứng thay vì ký ức.

## Đọc thêm

- [Yêu cầu server.json của Official Registry](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [Hợp đồng OpenAPI của Official Registry](https://registry.modelcontextprotocol.io/openapi.yaml)
- [Khám phá server trong MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)

---

[← Bài trước: MCP: độ tin cậy (reliability), huỷ (cancellation) và điều tiết luồng (flow control)](../../29-mcp-reliability-cancellation-and-flow-control/docs/vi.md) · [Bài sau: Kỹ nghệ tuân thủ đặc tả MCP: quản lý phiên bản (versioning), bằng chứng và vận hành (operations) →](../../31-mcp-conformance-versioning-and-operations/docs/vi.md)
