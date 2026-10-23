# Gọi kỹ năng và định tuyến

> Invocation (gọi kỹ năng) là một quyết định về thẩm quyền, rồi sau đó là một quyết định về độ liên quan. Một mô tả tốt giúp model chọn; một policy tốt quyết định lựa chọn đó có được phép hay không.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 13, Bài 24 (khám phá kỹ năng và tiết lộ dần)
**Time:** ~105 phút

## Mục tiêu học tập

- Phân biệt gọi kỹ năng tường minh bởi người dùng, gọi ngầm bởi model, gọi bởi ứng dụng, và gọi kỹ năng này từ kỹ năng khác.
- Mô hình hoá khả năng hiển thị với người và điều kiện hợp lệ với model như hai chiều policy độc lập.
- Viết mô tả định tuyến với trigger (điều kiện kích hoạt) dạng tích cực và ranh giới gần đúng.
- Tách tính hợp lệ, lựa chọn, kích hoạt, gắn tham số, và thực thi trong trace và test.
- Thích ứng các trường gọi kỹ năng đặc thù runtime mà không trình bày chúng như frontmatter di động được.

## Vấn đề

Bạn cài một kỹ năng `database-migration`. Người dùng có thể chạy nó theo tên, nhưng model cũng thấy mô tả của nó và chọn nó khi ai đó hỏi một câu chung về cơ sở dữ liệu. Kỹ năng sau đó đề xuất đổi schema cho một tác vụ chỉ cần giải thích.

Bạn thêm `user-invocable: false`, tưởng rằng sẽ chặn người dùng chạy tay. Trong runtime khác, trường đó bị bỏ qua. Bạn thêm `disable-model-invocation: true`, tưởng rằng kỹ năng sẽ biến mất hoàn toàn. Trong runtime hiểu nó, người dùng vẫn gọi được nó một cách tường minh.

Không có gì sai với tên trường. Model mới là thứ sai. "Người dùng thấy được nó", "model chọn được nó", "ứng dụng nạp trước được nó", và "công cụ bên trong nó chạy được" là những sự thật riêng biệt. Một boolean duy nhất tên `invocable` không thể diễn đạt hết.

Routing (định tuyến) có dạng lỗi thứ hai. Nếu mô tả mơ hồ, nhiều kỹ năng cùng có vẻ hợp lý. Nếu mô tả nhồi từ khoá, những tác vụ không liên quan sẽ kích hoạt chúng. Catalog là một giao diện xác suất: đủ gọn để vừa context, đủ cụ thể để định tuyến.

## Khái niệm

### Năm kênh có thể khởi động vòng đời

| Tác nhân | Hình thức gọi | Cách dùng điển hình | Rủi ro chính |
|---|---|---|---|
| Người dùng | Nêu tên kỹ năng trong UI hoặc prompt | Chọn quy trình có chủ đích | Người dùng kỳ vọng có sẵn hoặc có thẩm quyền mà host không cấp |
| Model hoặc agent tự hành | Chọn một mục trong catalog từ ngữ cảnh tác vụ | Thủ tục chuyên môn tự động | Định tuyến dương tính giả |
| Ứng dụng | Kích hoạt hoặc nạp trước một kỹ năng qua code runtime | Quy trình sản phẩm cố định | Ghép nối ẩn với một host |
| Kỹ năng khác hoặc subagent | Yêu cầu một kỹ năng chính xác như dependency của quy trình | Kết hợp | Chu trình, dependency thiếu, hoặc rò rỉ ngữ cảnh |
| Harness đánh giá | Kích hoạt một kỹ năng chính xác trong kịch bản cố định | Đo lường lặp lại được | Kiểm thử kỹ năng nhưng vô tình bỏ qua policy production đang được nghiên cứu |

Đặc tả Agent Skills di động định nghĩa gói. Nó không chuẩn hoá một UI slash-command phổ quát, một cờ định tuyến ngầm, một API ứng dụng, hay vòng đời subagent.

### Năm giai đoạn gọi kỹ năng

```figure
skill-invocation-stages
```

Dùng những từ này thật chính xác:

- **Eligible** nghĩa là policy cho phép tác nhân này yêu cầu kỹ năng.
- **Selected** nghĩa là người dùng đã nêu tên nó hoặc router đã đánh giá nó liên quan.
- **Activated** nghĩa là chỉ dẫn của nó đã vào ngữ cảnh làm việc.
- **Executing** nghĩa là agent đã bắt đầu công việc model hoặc công cụ theo các chỉ dẫn đó.
- **Completed** nghĩa là đầu ra đã qua một kiểm tra thành công độc lập.

Một trace chỉ ghi `skill_used=true` sẽ che mất ranh giới nơi lỗi xảy ra.

### Gọi bởi người và gọi bởi model tạo thành ma trận 2x2

| Người gọi được | Model gọi được | Chế độ | Ví dụ phù hợp |
|:---:|:---:|---|---|
| Có | Có | Chia sẻ | Giải thích code, lập kế hoạch test, review tài liệu |
| Có | Không | Chỉ người | Chuẩn bị phát hành, xuất hoá đơn, kế hoạch dọn dẹp có tính phá huỷ |
| Không | Có | Chỉ model | Hướng dẫn style nội bộ, tài liệu tham chiếu chuyên ngành, thủ tục hỗ trợ tự động |
| Không | Không | Tắt hoặc chỉ ứng dụng | Triển khai theo giai đoạn, gói đã ngừng, nạp trước bằng chương trình |

Ma trận này là một mô hình policy, không phải YAML chuẩn.

Một host hiện nay dùng `disable-model-invocation: true` cho hàng chỉ-người và `user-invocable: false` cho hàng chỉ-model. Mặc định là cả hai. Một host khác dùng `agents/openai.yaml` với `allow_implicit_invocation: false` để giữ việc gọi tường minh trong khi tắt việc chọn ngầm. Đây là các adapter runtime. Host lạ có thể bỏ qua chúng.

Chi tiết dễ nhầm mới quan trọng: `user-invocable: false` không có nghĩa là "model không dùng được cái này." Nó xoá việc người dùng gọi trực tiếp trong host định nghĩa nó. `disable-model-invocation: true` không có nghĩa là "kỹ năng bị tắt." Nó xoá việc model tự khởi tạo lựa chọn trong khi vẫn giữ quyền truy cập tường minh cho người dùng.

### Gọi tường minh là danh tính trước

Một lần gọi tường minh cung cấp danh tính trực tiếp:

```text
/release-readiness v2.4.0
```

hoặc:

```text
release-readiness check v2.4.0 without publishing
```

Các giao diện Codex hiện tại ghi nhận `/skills` để chọn và tên kỹ năng trần trong yêu cầu để gọi tường minh. Claude Code ghi nhận `/skill-name` và việc mở rộng tham số đặc thù host. Cú pháp chính xác, khả năng hiển thị menu, quy tắc trích dẫn, và mở rộng biến thuộc về host.

Một yêu cầu tường minh vẫn phải qua policy. Nêu tên một kỹ năng không được phép bỏ qua quyền còn thiếu, ràng buộc workspace, cổng phê duyệt, hay cách ly runtime.

### Gọi ngầm là mô tả trước

Với định tuyến ngầm, thoạt đầu model chỉ thấy metadata của catalog chứ không thấy toàn bộ thân kỹ năng. Vì vậy mô tả chính là giao diện định tuyến của kỹ năng.

Yếu:

```yaml
description: Helps with releases.
```

Quá rộng:

```yaml
description: Use for release, version, package, build, deploy, publish, tag, changelog, GitHub, CI, or software tasks.
```

Có ranh giới:

```yaml
description: Inspect an already prepared release candidate and produce a readiness report. Use when the user asks whether a version, tag, package, or image is ready to publish; do not use for ordinary build failures or feature development.
```

Bản có ranh giới gồm:

1. **Năng lực:** kiểm tra một bản ứng viên đã được chuẩn bị.
2. **Đầu ra:** báo cáo mức sẵn sàng.
3. **Ranh giới tích cực:** hỏi liệu một artifact phát hành đã sẵn sàng chưa.
4. **Ranh giới tiêu cực:** build thông thường và phát triển tính năng nằm ngoài phạm vi.

Ranh giới tiêu cực hữu ích khi hai kỹ năng gần nhau chia sẻ cùng từ vựng. Chúng không thay thế được eval ca gần đúng.

### Định tuyến là phân loại có lựa chọn kiêng quyết

Với một kỹ năng `s` và yêu cầu `x`, hãy hình dung điểm của router:

```text
score(s, x) = capability_match + trigger_match + context_match - exclusion_match - ambiguity_penalty
```

Cách tính điểm chính xác có thể là quyết định của LLM chứ không phải số học. Nguyên lý kỹ thuật vẫn đúng: lựa chọn phải vượt ngưỡng và phải thắng một kỹ năng cạnh tranh. Khi bằng chứng yếu, hãy kiêng quyết.

```figure
skill-routing-abstention
```

Với những kỹ năng có tác động lớn, định tuyến ngầm có thể không phù hợp dù mô tả mạnh. Dùng policy chỉ-người khi cái giá của một dương tính giả lớn hơn sự tiện lợi của việc chọn tự động.

### Tính hợp lệ phải đứng trước xếp hạng

Đừng chấm điểm mọi kỹ năng đã phát hiện, chọn kết quả khớp mạnh nhất, rồi mới kiểm tra policy của kỹ năng đó. Một kết quả khớp đầu bảng bị chặn sẽ ngăn sai cách một ứng viên hợp lệ có điểm thấp hơn được xét tới.

Dùng thứ tự ưu tiên (precedence) này cho định tuyến ngầm:

1. Lọc các kỹ năng đã phát hiện theo tác nhân đang yêu cầu và adapter host đang hoạt động.
2. Chỉ chấm điểm các ứng viên hợp lệ.
3. Chọn kết quả khớp hợp lệ mạnh nhất nếu nó vượt ngưỡng và các quy tắc về ambiguity (nhập nhằng).
4. Kiêng quyết khi không ứng viên nào hợp lệ hoặc không điểm hợp lệ nào đủ mạnh.

Giả sử `incident-triage` được `0.80` nhưng phần mở rộng host của nó tắt việc model gọi. `incident-review` được `0.55` và cho phép model gọi. Router nên đánh giá `incident-review` là ứng viên hợp lệ tốt nhất. Nó không nên chọn `incident-triage`, từ chối nó, rồi dừng.

Thứ tự này cũng giữ cho thay đổi policy không làm đổi ý nghĩa của điểm liên quan. Tính hợp lệ định nghĩa tập lựa chọn. Độ liên quan xếp hạng tập đó.

### Eval định tuyến cần các ca gần đúng

Ca dương tính chứng minh recall:

```json
{"prompt":"Is version 2.4.0 ready to publish?","expected":"release-readiness"}
```

Ca âm rõ ràng chứng minh precision cơ bản:

```json
{"prompt":"Explain rotary position embeddings.","expected":null}
```

Ca gần đúng phơi ra chất lượng ranh giới:

```json
{"prompt":"Why did today's package build fail?","expected":"build-diagnostics"}
```

Ca gần đúng chia sẻ `package` và `build` với kỹ năng phát hành nhưng thuộc chỗ khác. Một bộ định tuyến chỉ gồm dương tính hiển nhiên và âm tính không liên quan sẽ nói quá chất lượng của nó.

### Tham số có ba dạng biểu diễn

Một tham số của lần gọi đi qua nhiều ranh giới:

```figure
skill-argument-boundaries
```

Ở mỗi ranh giới, hãy giữ nguyên ý định mà không coi text là code.

- Bộ phân tích của host quyết định cú pháp lệnh và cách trích dẫn.
- Kỹ năng nhận text hoặc biến đã gắn theo quy tắc của host.
- Chỉ dẫn kiểm tra các giá trị bắt buộc và giá trị mặc định.
- Một lệnh gọi công cụ chuyển giá trị thành schema có kiểu và kiểm tra lại chúng.

Đừng nội suy tham số thô vào lệnh shell. Ưu tiên một script được gọi với argument vector hoặc một công cụ MCP có kiểu.

### Gọi từ ứng dụng là điều phối tường minh

Một sản phẩm có thể kích hoạt một kỹ năng vì quy trình của nó đã biết loại tác vụ. Ví dụ, một dịch vụ review pull-request có thể nạp trước `pull-request-risk-review` sau khi người dùng bấm Review.

Cách này loại bỏ sự không chắc chắn của định tuyến nhưng tạo phụ thuộc vào API runtime. Hãy giữ adapter đó bên ngoài thân kỹ năng di động:

```figure
skill-host-adapter
```

Kỹ năng vẫn phải dễ hiểu khi được một client tương thích khác mở ra.

### Gọi kỹ năng từ kỹ năng là một cạnh kiểu công cụ

Giả sử `release-readiness` yêu cầu `security-change-review` khi các tệp dependency thay đổi.

Bên gọi nên cung cấp:

- danh tính kỹ năng đích;
- một tác vụ có giới hạn và đường dẫn artifact;
- hợp đồng phản hồi mong đợi;
- lý do gọi;
- một fallback (phương án dự phòng) nếu không có sẵn;
- quy tắc độ sâu tối đa hoặc chống chu trình.

```json
{
  "target_skill": "security-change-review",
  "task": "Review dependency changes in the candidate diff",
  "inputs": ["artifacts/release.diff"],
  "expected": "risk-report.json",
  "max_depth": 2
}
```

Kỹ năng thứ hai không bị dán mù vào kỹ năng thứ nhất. Host quyết định cách kích hoạt nó và liệu nó chia sẻ ngữ cảnh, chạy trong một fork, hay trả về qua kết quả công cụ.

### Vòng đời ngữ cảnh tuỳ host

Sau khi kích hoạt, thân kỹ năng có thể nằm lại trong hội thoại, bị tóm tắt khi nén, hoặc chạy trong ngữ cảnh được giao phó. Quyền dùng công cụ có thể chỉ kéo dài một lượt trong khi chỉ dẫn tồn tại lâu hơn. Một subagent có thể nhận kỹ năng mà không có toàn bộ lịch sử của agent cha.

Đừng viết kỹ năng phụ thuộc vào một giả định vòng đời vô hình. Đặt đầu ra bền vững vào tệp hoặc trạng thái có kiểu, làm cho việc vào lại an toàn, và nêu rõ điều gì phải nạp lại sau khi bị ngắt.

```markdown
On resume, read `artifacts/release-readiness.json` if it exists.
Revalidate the candidate commit before continuing.
Do not repeat an external write whose idempotency key is already recorded.
```

## Tự tay làm

`code/main.py` triển khai policy và định tuyến như hai adapter riêng.

Model bao gồm:

- `Actor` cho người, model, agent tự hành, ứng dụng, kỹ năng, và harness;
- `SkillMetadata` cho danh tính định tuyến;
- `InvocationPolicy` cho ma trận người/model;
- `InvocationRequest` và `InvocationDecision` cho đầu vào và kết quả truy vết được;
- `CorePolicyAdapter` cho hành vi di động không có phần mở rộng host;
- `ExtensionPolicyAdapter` cho các trường runtime được nhận diện;
- `build_invocation_matrix(policy)` cho góc nhìn 2x2;
- `route_request(skills, request, adapter)` cho việc lọc tính hợp lệ trước khi xếp hạng độ liên quan, lựa chọn, và từ chối.

Chạy nó:

```bash
cd phases/13-tools-and-protocols/25-skill-invocation-and-routing
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Bản demo in ra một ma trận và các quyết định cho các kênh người gọi tường minh, model gọi ngầm, agent tự hành, ứng dụng, kết hợp kỹ năng, và harness. Kết quả của adapter mở rộng cho thấy một kết quả khớp từ vựng đầu bảng bị chặn được loại trước khi một phương án thay thế hợp lệ được xếp hạng. Nó cũng gồm allowlist theo tên chính xác. Không cần API model. Router tất định tồn tại để làm cho ranh giới policy có thể kiểm tra được, không phải để khẳng định rằng khớp từ vựng tái tạo được định tuyến model trong production.

### Vì sao adapter lõi và adapter mở rộng tách riêng

Nếu một bộ phân tích gán ý nghĩa cho mọi trường frontmatter quan sát được, nó lặng lẽ nâng quy ước runtime thành một chuẩn giả. Tách adapter buộc bên gọi phải nêu rõ ngữ nghĩa host nào đang hoạt động.

Adapter `CorePolicyAdapter` chỉ dùng policy do ứng dụng cung cấp. Adapter `ExtensionPolicyAdapter` nhận diện một tập trường host rõ ràng và ghi lại trường nào đã làm thay đổi quyết định.

## Áp dụng

Viết một hợp đồng gọi kỹ năng trước khi phát hành kỹ năng:

```yaml
actors:
  human: allow
  model: deny
  application: allow
  skill: deny
explicit_name: release-readiness
arguments:
  candidate: required
  publish: fixed_false
ambiguity: ask_user
missing_dependency: stop
context:
  durable_state: artifacts/release-readiness.json
  max_composition_depth: 2
```

Hợp đồng này là tài liệu thiết kế cho adapter và test. Nó không phải frontmatter `SKILL.md` di động trừ khi một chuẩn nào đó chính thức chấp nhận nó.

## Đóng gói

Bài này tạo ra bundle `skill-invocation-router`. Nó gồm tài liệu tham chiếu mô hình gọi kỹ năng, một policy host mẫu, và một CLI không thực thi, đánh giá một yêu cầu từ người, model, agent tự hành, ứng dụng, kết hợp kỹ năng, hoặc harness rồi trả về quyết định JSON với kênh, adapter, điểm, và lý do.

CLI một-yêu-cầu là bộ dò policy, không phải một lần đánh giá trigger đầy đủ. Dùng thiết kế dương tính và ca gần đúng có nhãn ở Bài 27 để tính confusion count, precision, recall, và độ ổn định khi chạy lặp.

## Bài tập

1. Tạo đủ bốn hàng của ma trận người/model và viết một ca sử dụng hợp lý cho mỗi hàng.
2. Thêm kích hoạt chỉ-ứng-dụng vào `CorePolicyAdapter`. Chứng minh rằng bên gọi là người và model vẫn bị từ chối.
3. Viết mười ca gần đúng cho một kỹ năng triển khai. Mỗi prompt phải chia sẻ từ vựng với kỹ năng nhưng thuộc quy trình khác.
4. Thêm một biên ambiguity giữa hai điểm định tuyến cao nhất. Trả `ask` khi biên quá nhỏ.
5. Thêm độ sâu kết hợp tối đa cho yêu cầu kỹ năng-từ-kỹ-năng và phát hiện chu trình hai kỹ năng.
6. Chạy cùng tập có nhãn qua adapter lõi và adapter mở rộng. Giải thích mọi quyết định đã thay đổi.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| Explicit invocation | "Slash command" | Một tác nhân cung cấp danh tính kỹ năng trực tiếp, tuỳ thuộc policy |
| Implicit invocation | "Model tự chọn" | Router chọn từ metadata catalog hợp lệ dựa trên ngữ cảnh tác vụ |
| User-invocable | "Người dùng dùng được" | Thuộc tính menu hoặc gọi trực tiếp đặc thù host, không phải trường lõi |
| Model-invocable | "Agent dùng được" | Điều kiện hợp lệ cho việc model chọn ngầm theo policy của host |
| Invocation adapter | "Bộ phân tích frontmatter" | Code ánh xạ trường và API của một host vào một mô hình policy đã khai báo |
| Near miss | "Hard negative" | Một yêu cầu không kích hoạt nhưng giống đầu vào dự định của kỹ năng |
| Abstention | "Không chọn kỹ năng nào" | Kết quả định tuyến có chủ đích khi thiếu bằng chứng hoặc còn nhập nhằng |

## Đọc thêm

- [Optimizing skill descriptions](https://agentskills.io/skill-creation/optimizing-descriptions) về trigger tích cực, độ cụ thể, và đánh giá.
- [Evaluating skills](https://agentskills.io/skill-creation/evaluating-skills) về thiết kế eval cho trigger và đầu ra.
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills) về các điều khiển gọi tường minh và gọi ngầm hiện tại của Codex.
- [Claude Code skills](https://code.claude.com/docs/en/skills) về `user-invocable`, `disable-model-invocation`, tham số, và ngữ cảnh được giao phó ở một host cụ thể.

---

[← Bài trước: Khám phá skill và tiết lộ dần](../../24-skill-discovery-and-progressive-disclosure/docs/vi.md) · [Bài sau: Quyền (permission) của kỹ năng, hộp cát (sandbox) và sự tin cậy →](../../26-skill-permissions-sandboxes-and-trust/docs/vi.md)
