# MCP: độ tin cậy (reliability), huỷ (cancellation) và điều tiết luồng (flow control)

> Một request ID chỉ tương quan một thông điệp. Nó không làm tác dụng phụ trở nên an toàn, không dừng một worker, và không bảo vệ một luồng khỏi consumer chậm.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 13, Bài 09 và 13
**Time:** ~120 phút

## Mục tiêu học tập

- Cài đặt tín hiệu huỷ đúng cho stdio và Streamable HTTP.
- Xử lý các cuộc đua giữa hoàn tất và huỷ mà không gửi thông điệp sau khi đã huỷ.
- Tách biệt việc huỷ yêu cầu với ngữ nghĩa bền vững của `tasks/cancel`.
- Xây dựng quyết định retry (thử lại) từ tác dụng phụ và khoá idempotency (tính bất biến khi lặp) tường minh.
- Giới hạn hàng đợi tiến độ trong khi vẫn giữ phản hồi cuối cùng.
- Khôi phục luồng qua kết nối lại, đọc lại, và backoff có jitter.

## Vấn đề

Happy path che giấu những lỗi hệ phân tán đắt đỏ nhất.

Một client gọi một tool. Server bắt đầu làm việc. Tiến độ về tới. Một proxy đệm luồng. Client chạm timeout (hạn chờ) và ngắt kết nối. Server xong việc muộn hơn một mili giây. Client thử lại với một JSON-RPC id mới. Thao tác biến đổi chạy hai lần.

Mọi thành phần đều đúng ở phạm vi cục bộ. Hệ thống hỏng ở phạm vi toàn cục.

MCP định nghĩa hành vi thông điệp và transport, nhưng ứng dụng của bạn vẫn phải tự lo:

- ngân sách thời gian;
- tính idempotency ở tầng nghiệp vụ;
- hàng đợi có giới hạn;
- phân loại retry;
- trạng thái task bền vững;
- chính sách kết nối lại và đọc lại.

Bài học này đưa những quyết định đó vào một simulator tất định. Không có
sleep, socket hay lỗi ngẫu nhiên. Bạn điều khiển trực tiếp thứ tự
sự kiện huỷ. Một bài test có đồng bộ luồng buộc hai client ledger tranh
nhau cùng một khoá idempotency.

## Huỷ yêu cầu phụ thuộc vào transport

Ý định là như nhau trên mọi transport: client không còn cần kết quả đang bay. Tín hiệu trên đường truyền thì khác nhau.

### stdio

stdio dùng một kênh hai chiều chia sẻ. Client gửi một notification:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/cancelled",
  "params": {
    "requestId": 41,
    "reason": "User closed the operation"
  }
}
```

Notification này là fire-and-forget. Server không phát phản hồi JSON-RPC nào cho nó.

Server nên dừng công việc, giải phóng tài nguyên, và tránh gửi phản hồi cho yêu cầu đã bị huỷ. Nó có thể bỏ qua tín hiệu huỷ khi yêu cầu không tồn tại, đã xong, hoặc không thể dừng an toàn.

Các notification huỷ sai định dạng, không tồn tại, hoặc đã hoàn tất đều bị bỏ qua. Biến những cuộc đua đó thành lỗi mới sẽ chỉ tạo thêm cuộc đua.

### Streamable HTTP

Streamable HTTP hiện đại cấp cho mỗi yêu cầu một luồng phản hồi HTTP hoặc SSE riêng. Client huỷ bằng cách đóng luồng phản hồi của yêu cầu đó.

Đừng POST `notifications/cancelled` cho một yêu cầu HTTP thông thường. Việc đóng luồng chính là tín hiệu huỷ.

Khi server phát hiện ngắt kết nối, nó nên dừng công việc và không được gửi thêm thông điệp cho yêu cầu đó.

### Huỷ do server gửi có phạm vi hẹp

Server không dùng `notifications/cancelled` để huỷ các lệnh gọi tuỳ ý của client. Trên stdio, tín hiệu huỷ do server gửi chỉ dành để kết thúc một yêu cầu `subscriptions/listen`. Hãy giữ đường đó tách biệt khỏi việc huỷ yêu cầu thông thường từ client.

## Huỷ là một cuộc đua

Hai thứ tự sự kiện đều hợp lệ.

### Huỷ thắng

```text
request starts
client sends cancellation signal
server marks request cancelled
worker reaches completion
server suppresses the response
```

### Hoàn tất thắng

```text
request starts
worker commits the result
server sends the response
cancellation arrives late
server ignores the late notification
```

Client cũng phải bỏ qua phản hồi muộn cho một yêu cầu mà nó đã từ bỏ. Độ trễ mạng khiến không bên nào chứng minh được bên kia đã thấy sự kiện nào trước.

```figure
mcp-reliability-race
```

`RequestCoordinator` của bài học lưu một trạng thái kết thúc duy nhất. `complete()` không trả về phản hồi sau khi đã huỷ. Một tín hiệu huỷ muộn không thể thay đổi bản ghi đã hoàn tất.

## Timeout cần hai đồng hồ

Một bộ đếm không hoạt động duy nhất là không đủ.

Hãy dùng hai giới hạn:

1. **Idle timeout.** Yêu cầu được phép không tạo hoạt động hữu ích trong bao lâu.
2. **Maximum timeout.** Ngân sách thời gian tuyệt đối tính từ lúc yêu cầu bắt đầu.

Tiến độ có thể đặt lại đồng hồ idle. Nó không bao giờ được xoá hạn chót tối đa.

```text
start: 0 ms
progress: 400 ms
progress: 800 ms
progress: 1200 ms
idle timeout: 500 ms
maximum timeout: 2000 ms
```

Ở mốc 1500 ms, yêu cầu vẫn đang hoạt động vì tiến độ gần nhất mới 300 ms. Ở mốc 2000 ms, hạn chót tối đa huỷ yêu cầu kể cả khi một sự kiện tiến độ khác vừa tới ở 1999 ms.

Tiến độ là tuỳ chọn. Server có thể nhận một progress token và không phát cập nhật nào. Đừng bao giờ biến sự hiện diện của token thành một timeout vô hạn.

Giá trị tiến độ của MCP phải tăng. Notification dừng sau khi hoàn tất hoặc bị huỷ. Hãy giới hạn tần suất tiến độ để worker nhanh không làm ngập transport.

## Huỷ yêu cầu không phải `tasks/cancel`

Các cơ chế này giải quyết những vòng đời khác nhau.

| Cơ chế | Đối tượng | Tín hiệu | Thành công nghĩa là gì |
|-----------|--------|--------|--------------------|
| Huỷ yêu cầu trên stdio | Một RPC đang bay | `notifications/cancelled` | Client đã từ bỏ yêu cầu; server nên dừng nếu khả thi |
| Huỷ yêu cầu trên HTTP | Một luồng phản hồi đang bay | Đóng luồng | Client đã từ bỏ yêu cầu; server nên dừng nếu khả thi |
| `tasks/cancel` | Một Task bền vững | Yêu cầu MCP thông thường | Server đã ghi nhận ý định huỷ |

Một kết quả `tasks/cancel` thành công không chứng minh worker đã dừng. Task có thể vẫn ở trạng thái `working` cho tới khi một checkpoint của worker thấy cờ đó. Công việc có thể xong trước checkpoint đó.

Đừng xoá trạng thái task bền vững khi kết nối HTTP đóng. Lý do để tạo một Task là vòng đời của nó dài hơn một yêu cầu và một kết nối.

## JSON-RPC ID mới không phải idempotency

JSON-RPC id tương quan yêu cầu với phản hồi. Chúng không định danh một thao tác nghiệp vụ.

Giả sử client gửi một khoản trừ tiền với id `41`, mất phản hồi, rồi thử lại với id `42`. Server thấy hai thông điệp khác nhau. Không có một khoá ở tầng ứng dụng, nó không thể biết chúng đại diện cho cùng một lần thanh toán.

Một khoá idempotency định danh ý định nghiệp vụ:

```json
{
  "name": "charge_account",
  "arguments": {
    "account": "acct-7",
    "cents": 1200,
    "idempotencyKey": "checkout-7"
  }
}
```

Server lưu:

- khoá;
- một fingerprint của các tham số thao tác;
- kết quả đã commit.

Cùng khoá và cùng tham số thì trả về kết quả đã lưu. Cùng khoá nhưng tham số khác thì bị từ chối. Điều này ngăn việc dùng lại khoá do nhầm lẫn làm biến đổi một thao tác nghiệp vụ khác.

### Ranh giới ledger phải nguyên tử và bền vững

Chuỗi này không an toàn:

```text
check key
run mutation
store result
```

Hai worker có thể cùng thấy khoá chưa tồn tại và cùng chạy thao tác biến đổi. Một sự cố
sau khi hiệu ứng xảy ra nhưng trước khi lưu sẽ tạo ra cùng sự mơ hồ khi thử lại.

Bài học dùng một ledger SQLite lưu bằng file. `BEGIN IMMEDIATE` tuần tự hoá việc
kiểm tra khoá, hiệu ứng nghiệp vụ mô phỏng, bộ đếm thực thi, và kết quả đã lưu vào
một transaction. Do đó hai kết nối ledger độc lập tranh nhau cùng một khoá sẽ
chỉ thấy một kết quả đã commit và một lần thực thi. Đóng rồi mở lại ledger vẫn
giữ được bản ghi đó.

Mọi giá trị trả về đều được dựng lại từ JSON đã lưu. Người gọi không bao giờ nhận
được đối tượng khả biến mà ledger đang giữ, nên việc sửa một dict được trả về không thể
làm hỏng kết quả replay về sau.

Hiệu ứng nghiệp vụ của simulator là biên lai và bộ đếm thực thi nằm trong cùng
transaction SQLite đó. Một khoản thanh toán, một lần deploy, hay một lệnh gọi API
bên ngoài thật không trở thành nguyên tử chỉ vì ghi vào một bảng cục bộ. Môi trường
production cần một transaction cơ sở dữ liệu chia sẻ bền vững, một transactional
outbox, hoặc một nhà cung cấp phía trên thực thi đúng cùng khoá idempotency. Một
process lock đơn lẻ không bảo vệ được nhiều replica hay sống sót qua khởi động lại.

### Ma trận retry

Phân loại retry trước khi hiện thực chúng.

| Lớp | Ví dụ | Quy tắc retry |
|------|---------|------------|
| An toàn | Đọc tất định không có tác dụng phụ | Thử lại với một JSON-RPC id mới sau khi đã hiểu rõ ranh giới lỗi |
| Có điều kiện | Thao tác biến đổi có khoá idempotency bền vững | Thử lại với cùng khoá và tham số giống hệt |
| Không an toàn | Thao tác biến đổi không khử trùng ở tầng nghiệp vụ | Không tự động thử lại; đối soát trước |

Các annotation của tool như `readOnlyHint` và `idempotentHint` vẫn chỉ là gợi ý không đáng tin. Hợp đồng ứng dụng và cách hiện thực server quyết định độ an toàn khi retry.

## Backpressure (áp lực ngược) là một phần của tính đúng đắn

Một producer SSE có thể sinh tiến độ nhanh hơn client, proxy, hoặc mạng tiêu thụ. Một hàng đợi không giới hạn biến sự chậm thành cạn kiệt bộ nhớ.

Hãy dùng hàng đợi có giới hạn và định nghĩa rõ cái gì có thể mất.

Tiến độ là thứ thay thế được. Một giá trị tiến độ sau thay thế giá trị trước với cùng token. Một phản hồi JSON-RPC cuối cùng thì không thể thay thế.

Buffer của bài học áp dụng chính sách này:

1. Gộp các tiến độ liền nhau của cùng một token.
2. Bỏ tiến độ cũ nhất khi đầy dung lượng.
3. Đánh dấu luồng là cần đọc lại từ nguồn có thẩm quyền.
4. Giữ nguyên phản hồi cuối cùng.
5. Từ chối trạng thái mà việc giữ phản hồi cuối cùng đòi hỏi phải bỏ một phản hồi cuối khác.

Đây là mất mát có giới hạn đi kèm khôi phục tường minh. Mất mát âm thầm không phải là một chiến lược.

### Đệm ở proxy

Server có thể stream đúng trong khi một reverse proxy giữ các sự kiện trong bộ đệm.

Với một phản hồi SSE, hãy gửi:

```http
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
```

Đặc tả Streamable HTTP 2026 khuyến nghị `X-Accel-Buffering: no` để các proxy tương thích chuyển sự kiện ngay.

Với những luồng dài và im lặng, hãy định kỳ phát một comment SSE:

```text
:
```

Client bỏ qua các dòng comment. Các thành phần trung gian thấy có lưu lượng nên ít khả năng đóng một kết nối idle.

Keepalive không phải tiến độ. Đừng đặt lại timeout idle ngữ nghĩa của một thao tác chỉ vì có một comment transport tới.

## Kết nối lại nghĩa là đọc lại

Streamable HTTP hiện đại không hỗ trợ SSE có thể tiếp tục qua `Last-Event-ID`.

Sau khi một luồng `subscriptions/listen` bị đứt:

1. Mở một yêu cầu listen mới với một JSON-RPC id mới.
2. Khôi phục bộ lọc subscription mong muốn.
3. Đọc lại các tool, resource, prompt, hoặc Task bị ảnh hưởng từ những phương thức có thẩm quyền.
4. Khử trùng trạng thái ứng dụng theo các định danh ổn định.
5. Đừng replay một thao tác biến đổi không an toàn chỉ vì phản hồi của nó bị mất.

Kế hoạch khôi phục mẫu đặt `sendLastEventId` thành false một cách tường minh và liệt kê các resource cần đọc lại.

### Ngăn đám đông kết nối lại

Nếu 10.000 client kết nối lại đúng một giây, server đang hồi phục sẽ lại sập.

Hãy dùng exponential backoff có jitter và có trần. Bài học tính jitter tất định từ client id và số lần thử để test vẫn tái lập được:

```text
attempt 0: up to 250 ms
attempt 1: up to 500 ms
attempt 2: up to 1000 ms
...
cap: 8000 ms
```

Production có thể dùng ngẫu nhiên mật mã hoặc ngẫu nhiên lúc chạy. Bất biến là phân bố, không phải một công thức cụ thể.

## Tự tay làm

`code/main.py` dựng năm thành phần độ tin cậy nhỏ.

### `RequestCoordinator`

- bắt đầu một yêu cầu đang bay với hạn idle và hạn tối đa;
- phát notification tiến độ đơn điệu tăng;
- tạo tín hiệu huỷ stdio hoặc HTTP đúng;
- bỏ qua các notification huỷ không hợp lệ;
- làm rõ các cuộc đua kết thúc giữa huỷ và hoàn tất;
- dành tín hiệu huỷ do server gửi cho subscription trên stdio.

### `MutationLedger`

- chứng minh rằng hai JSON-RPC id thực thi hai lần khi không có khoá nghiệp vụ;
- dùng một transaction SQLite lưu bằng file cho việc kiểm tra khoá, hiệu ứng mô phỏng,
  bộ đếm thực thi, và commit kết quả;
- khử trùng các tham số khớp nhau dưới một khoá idempotency trên những kết nối
  ledger độc lập;
- từ chối một khoá bị dùng lại với tham số khác;
- trả về bản sao phòng vệ và giữ các bản ghi đã commit qua lần mở lại.

### `DurableTaskService`

- ghi nhận một yêu cầu huỷ;
- giữ task ở trạng thái `working` cho tới checkpoint của worker;
- minh hoạ vì sao ghi nhận không phải trạng thái cuối.

### `BoundedSseBuffer`

- gộp hoặc bỏ tiến độ khi bị sức ép;
- ghi nhận rằng cần đọc lại từ nguồn có thẩm quyền;
- không bao giờ bỏ phản hồi cuối cùng.

### Các helper khôi phục

- trả về header SSE an toàn với proxy và comment keepalive;
- tạo kế hoạch kết nối lại và đọc lại;
- rải các lần thử lại bằng exponential backoff tất định có jitter.

## Áp dụng

Từ thư mục gốc của repo:

```bash
cd phases/13-tools-and-protocols/29-mcp-reliability-cancellation-and-flow-control/code
python3 main.py
python3 -m unittest discover tests -v
```

Demo chạy cả hai phía của cuộc đua trung tâm, thực hiện một thao tác biến đổi được
khử trùng ở mức transaction trong một ledger tạm lưu bằng file, làm quá tải một
buffer tiến độ có giới hạn, và cho thấy một Task bền vững đi từ trạng thái đã ghi nhận
huỷ tới trạng thái worker đã thấy huỷ.

## Lab tương tác

Chạy bốn thứ tự sự kiện mà không thêm sleep.

1. Bắt đầu yêu cầu `A`, huỷ nó, rồi gọi `complete()`.
2. Bắt đầu yêu cầu `B`, hoàn tất nó, rồi gửi tín hiệu huỷ.
3. Bắt đầu yêu cầu `C`, phát tiến độ trước mỗi hạn idle, rồi vượt hạn tối đa.
4. Bắt đầu yêu cầu `D` qua Streamable HTTP và đóng luồng phản hồi của nó.

Ghi lại cho mỗi tình huống:

- trạng thái kết thúc của yêu cầu;
- có tồn tại phản hồi cuối cùng hay không;
- tín hiệu huỷ được đặt lên đường truyền;
- sự kiện nào client nên bỏ qua.

Sau đó đổi `D` sang stdio. Thao tác là như nhau, nhưng tín hiệu huỷ phải đổi.

## Lab thực hành

Thêm một thao tác biến đổi `reserve_inventory` vào `MutationLedger`.

Yêu cầu:

1. Khoá ràng buộc SKU, số lượng, tenant, và tên thao tác.
2. Một lần thử lại với cùng khoá và cùng tham số trả về lượt giữ chỗ đầu tiên.
3. Một lần thử lại với số lượng đã đổi sẽ thất bại mà không tạo thêm lượt giữ chỗ.
4. Một lần thực thi đã commit nhưng mất phản hồi có thể được đối soát theo khoá.
5. Kết quả không ghi lại bí mật hay dữ liệu thanh toán.
6. Tự động thử lại bị tắt khi client không cung cấp khoá.
7. Thêm một tình huống subscription bị đứt mô phỏng và đọc lại bản ghi inventory trước khi quyết định làm gì tiếp.
8. Khởi động hai kết nối ledger tại một barrier và gửi cùng một khoá
   đồng thời. Khẳng định rằng chỉ một lượt giữ chỗ được commit.
9. Sửa đối tượng giữ chỗ đầu tiên được trả về. Replay khoá đó và chứng minh
   kết quả đã lưu không đổi.
10. Đóng rồi mở lại file ledger, sau đó đối soát lượt giữ chỗ theo khoá.

Hãy giữ lab trung thực: nếu inventory nằm ở một service khác, giải thích xem service
đó có chấp nhận cùng khoá idempotency hay không, hay một transactional outbox
nối commit cục bộ với hiệu ứng ở xa.

## Sản phẩm bàn giao

`outputs/skill-mcp-reliability-reviewer.md` là một skill đánh giá độ tin cậy dạng phẳng. Hãy đưa cho nó một thao tác MCP, transport, chính sách timeout, hành vi retry, chính sách hàng đợi, và kế hoạch khôi phục. Nó trả về một bảng race, phân loại retry, ranh giới idempotency, các kiểm tra điều tiết luồng, và fixture lỗi.

## Kiểm chứng

Bài học hoàn tất khi những khẳng định sau đúng:

- huỷ trên stdio gửi `notifications/cancelled` và không nhận phản hồi nào.
- huỷ trên Streamable HTTP đóng luồng yêu cầu và không gửi POST huỷ nào.
- Huỷ-trước-khi-hoàn-tất sẽ chặn phản hồi cuối cùng.
- Hoàn-tất-trước-khi-huỷ giữ phản hồi và bỏ qua tín hiệu huỷ muộn.
- Tiến độ có thể đặt lại idle timeout nhưng không bao giờ đặt lại maximum timeout.
- Chỉ một JSON-RPC id mới sẽ thực thi thao tác biến đổi thêm lần nữa.
- Một khoá idempotency và các tham số giống hệt chỉ thực thi một lần dưới cuộc đua
  hai kết nối đồng thời.
- Một bản ghi đã commit sống sót qua lần mở lại và replay trả về một bản sao phòng vệ.
- Sửa một kết quả được trả về không thể thay đổi kết quả đã lưu.
- Buffer có giới hạn vẫn nằm trong dung lượng và giữ phản hồi cuối cùng.
- Kết nối lại dùng yêu cầu mới, không gửi `Last-Event-ID`, và đọc lại trạng thái bị ảnh hưởng.
- Việc ghi nhận `tasks/cancel` để task ở trạng thái chưa kết thúc cho tới khi worker thấy nó.

## Các kiểu hỏng trong production

| Sự cố | Triệu chứng quan sát được | Phản ứng đúng |
|---------|--------------------|------------------|
| Client HTTP POST notification huỷ | Server và client bất đồng về vòng đời yêu cầu | Đóng luồng phản hồi SSE của yêu cầu |
| Server phản hồi sau khi đã chấp nhận huỷ | Client nhận một kết quả muộn không dùng được | Dừng việc và chặn các thông điệp tiếp theo khi huỷ thắng |
| Tiến độ đặt lại mọi hạn chót | Công việc bị treo sống mãi | Giữ một maximum timeout tuyệt đối riêng |
| Id RPC mới bị coi là khử trùng | Khoản trừ tiền, deploy, hoặc xoá chạy hai lần | Thêm một khoá idempotency bền vững ở tầng ứng dụng |
| Kiểm tra khoá và hiệu ứng tách rời | Các worker đồng thời cùng thấy khoá chưa tồn tại | Commit việc nhận khoá, bản ghi hiệu ứng, và kết quả một cách nguyên tử |
| Ledger trong bộ nhớ dùng chung giữa các replica | Khởi động lại hoặc worker khác quên các commit trước | Dùng lưu trữ bền vững chia sẻ hoặc idempotency ở phía trên |
| Trả trực tiếp kết quả khả biến đã lưu | Việc sửa của người gọi làm hỏng các lần replay sau | Serialize kết quả đã commit và trả về bản sao phòng vệ |
| Khoá bị dùng lại với tham số đã đổi | Một khoá trỏ tới hai ý định nghiệp vụ | Lưu và so sánh fingerprint của tham số |
| Hàng đợi tiến độ không giới hạn | Bộ nhớ tăng khi consumer chậm | Gộp và bỏ tiến độ thay thế được trong một giới hạn |
| Phản hồi cuối cùng bị bỏ khi bị sức ép | Client không thể biết kết cục của yêu cầu | Dành dung lượng hoặc loại tiến độ, không bao giờ loại phản hồi cuối |
| Proxy đệm SSE | Tiến độ tới theo đợt hoặc sau timeout | Tắt đệm và cấu hình timeout proxy tương thích |
| Giả định có `Last-Event-ID` | Client tiếp tục từ trạng thái mà server không hỗ trợ | Kết nối lại bằng yêu cầu mới và đọc lại |
| Mọi client kết nối lại ngay lập tức | Việc khôi phục tạo ra một sự cố mới | Dùng exponential backoff có trần kèm jitter |
| Ghi nhận task bị coi là huỷ cuối cùng | Worker vẫn chạy sau khi UI báo đã dừng | Poll Task cho tới khi có trạng thái kết thúc |

## Kết nối với capstone

Capstone về hệ sinh thái tool nên coi độ tin cậy là bằng chứng chạy được, không phải một đoạn văn trong sơ đồ kiến trúc.

Yêu cầu các artefact sau:

- một bản ghi cuộc đua huỷ cho mỗi transport;
- một bảng retry cho mọi thao tác biến đổi được phơi ra;
- một bản ghi khoá idempotency và một fixture lệch tham số;
- một bản ghi cùng khoá đồng thời, một lần kiểm tra mở lại, và một lần kiểm tra khoá bị alias;
- một kết quả quá tải của buffer có giới hạn;
- header SSE cho reverse proxy và chính sách idle;
- một kế hoạch kết nối lại nêu tên các phương thức đọc lại có thẩm quyền;
- một dấu vết huỷ Task bền vững khi capstone dùng Task.

Một yêu cầu trả về xanh trong tiến trình cục bộ chỉ chứng minh happy path. Capstone sẵn sàng cho production khi phản hồi bị mất, huỷ muộn, consumer chậm, và đám đông kết nối lại đều có kết cục tất định.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa thực sự |
|------|---------|
| Request cancellation | Từ bỏ một yêu cầu MCP đang bay |
| Cancellation race | Sự tranh chấp giữa sự kiện hoàn tất và sự kiện huỷ |
| Idle timeout | Giới hạn tính từ hoạt động hữu ích gần nhất của yêu cầu |
| Maximum timeout | Giới hạn tuyệt đối tính từ lúc yêu cầu bắt đầu, không bị tiến độ ảnh hưởng |
| Idempotency key | Định danh ở tầng ứng dụng khử trùng một ý định nghiệp vụ |
| Atomic ledger | Ranh giới bền vững commit việc nhận khoá, bản ghi hiệu ứng, và kết quả như một khối |
| Backpressure | Sự điều tiết áp dụng khi producer nhanh hơn consumer |
| Progress coalescing | Thay tiến độ cũ bằng một giá trị có thẩm quyền mới hơn |
| Refetch | Đọc lại trạng thái hiện tại sau một khoảng trống của luồng |
| Jitter | Sự biến thiên có chủ đích để rải các lần thử lại theo thời gian |

## Đọc thêm

- [MCP Cancellation](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/cancellation)
- [MCP Progress](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/progress)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Tasks Extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)

---

[← Bài trước: Hợp đồng công cụ và nội dung trong MCP](../../28-mcp-tool-contracts-and-content/docs/vi.md) · [Bài sau: Chuỗi cung ứng MCP Registry: tiếp nhận, lệch trạng thái và rollback →](../../30-mcp-registry-supply-chain-and-drift/docs/vi.md)
