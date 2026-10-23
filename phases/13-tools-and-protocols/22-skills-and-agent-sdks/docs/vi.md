# Agent Skills: hợp đồng portable và ranh giới thực thi

> Một skill (kỹ năng) không phải là một prompt dài với cái tên tệp hay hơn. Nó là một gói hướng dẫn, tài nguyên và trình trợ giúp thực thi có thể khám phá được, đi vào ngữ cảnh của agent thông qua một hợp đồng runtime (hợp đồng thực thi).

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 13, Bài 01 (giao diện công cụ), Giai đoạn 13, Bài 05 (thiết kế lược đồ công cụ)
**Time:** ~90 phút

## Mục tiêu học tập

- Định nghĩa một agent skill mà không nhầm nó với prompt, hướng dẫn trong repo, công cụ, hook, subagent hay plugin.
- Đọc hợp đồng `SKILL.md` portable và tách nó khỏi các phần mở rộng riêng của từng runtime.
- Giải thích khám phá, chọn lựa, kích hoạt, nạp tài nguyên, dùng công cụ và kiểm chứng như những giai đoạn vòng đời riêng biệt.
- Kiểm tra một gói skill trước khi runtime đặt nó vào danh mục của agent.
- Chọn giữa skill, công cụ MCP, hook, subagent hay code thường cho một tác vụ cụ thể.

## Thành công đầu tiên trong mười phút

Hãy làm việc này trước phần giải thích dài. Bạn sẽ tạo một skill nhỏ, cài
bộ reviewer đầy đủ vào một agent host thật, gọi nó, kiểm chứng
kết quả, rồi gỡ nó đi. Việc này chứng minh vòng đời bằng một kết quả quan sát được.

### Chuẩn bị trước cho lab trên host thật

Điểm kiểm tra trên host thật cần Node.js, `npx`, Python 3, một host có hỗ trợ
skill mà bạn chọn, và quyền ghi vào scope project hoặc user mà bạn chọn trong
installer. Hãy kiểm tra các lệnh cục bộ trước:

```bash
node --version
npx --version
python3 --version
```

Hãy quyết định host và scope bạn sẽ dùng trước khi cài. Nếu thiếu bất kỳ
yêu cầu nào, hãy đọc bài này trên website hoặc tiếp tục với
bài tập gói thủ công bên dưới. Cách dự phòng đó dạy hợp đồng, nhưng
không chứng minh được khám phá trên host, việc gọi, chạy script đi kèm hay
hành vi gỡ cài đặt. Hãy đánh dấu những quan sát đó là còn chờ.

### 1. Bắt đầu trong một thư mục làm việc trống

Chạy các lệnh này từ bất kỳ thư mục cha nào nơi bạn lưu công việc học tập:

```bash
mkdir -p agent-skills-first-run
cd agent-skills-first-run
TARGET_ROOT="$(pwd -P)"
printf 'TARGET_ROOT=%s\n' "$TARGET_ROOT"
ls -A
```

Lệnh cuối cùng sẽ không in ra gì. Nếu nó in ra tệp, hãy chọn một thư mục
trống khác để phần review có ranh giới rõ ràng.

Tạo một thư mục cho skill đầu tiên của bạn:

```bash
mkdir -p my-first-skill
```

Tạo `my-first-skill/SKILL.md` với nội dung sau:

```markdown
---
name: my-first-skill
description: Turn rough meeting notes into a compact decision record when the user asks to capture a technical decision.
---

# Decision record

Extract the decision, context, alternatives, owner, and next review date.
If the notes do not contain a decision, ask one clarifying question instead
of inventing one.
```

Kiểm tra rằng bạn đã tạo tệp đúng trong thư mục mong muốn:

```bash
test -f my-first-skill/SKILL.md
```

Không có output và exit code 0 nghĩa là tệp tồn tại.

### 2. Cài bộ reviewer đầy đủ

Vẫn ở trong `agent-skills-first-run` và chạy:

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-contract-reviewer --full-depth
```

Chọn agent host và scope bạn đang dùng. Installer sẽ liệt kê
`skill-contract-reviewer` và đích mà nó đã ghi. `--full-depth` là
bắt buộc vì skill của bài này là một bundle lồng nhau gồm references, một
script và một asset.

Đặt `SKILL_ROOT` thành thư mục tuyệt đối mà installer báo. Nó phải
là thư mục chứa `SKILL.md` đã cài, không phải thư mục nguồn của bài học
và cũng không phải workspace hiện tại:

```bash
# Replace the placeholder with the destination printed by the installer.
SKILL_ROOT="$(cd "/absolute/path/to/skill-contract-reviewer" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\n' "$SKILL_ROOT"
```

Nếu phiên agent đã mở từ trước, hãy mở phiên mới hoặc dùng lệnh
quét lại skill của host đó. Đừng cho rằng host nào cũng hot-reload danh mục của nó.

### 3. Gọi nó một cách tường minh

Trong agent đã cài, với `agent-skills-first-run` là thư mục làm việc,
hãy dùng cú pháp mà host đó hỗ trợ:

| Host | Gọi tường minh |
|---|---|
| Codex | `skill-contract-reviewer`, hoặc chọn nó từ `/skills`, rồi đưa yêu cầu review |
| Claude Code | `/skill-contract-reviewer` rồi tới yêu cầu review |
| Portable fallback | `Use skill-contract-reviewer to review the target package.` |

Dùng các giá trị tuyệt đối đã in cho `SKILL_ROOT` và `TARGET_ROOT` trong
yêu cầu. Yêu cầu host mở rộng chúng trước khi thực thi và hiển thị đúng
lệnh đã resolve, không phải một lệnh phụ thuộc thư mục làm việc của tiến trình:

```text
Use skill-contract-reviewer to review <TARGET_ROOT>/my-first-skill. The installed bundle root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/check_skill.py <TARGET_ROOT>/my-first-skill. Before running it, show the fully resolved argv. Return the validation report, selected primitives, and one sentence for each selection. Include the resolved script path, resolved target path, cwd, argv, and exit code as execution evidence.
```

Lệnh đã resolve sẽ có dạng như sau, không còn placeholder nào:

```bash
python3 "/absolute/install/path/skill-contract-reviewer/scripts/check_skill.py" \
  "/absolute/workspace/path/agent-skills-first-run/my-first-skill"
```

Một kết quả thành công có cả ba thuộc tính:

1. Host tìm thấy `skill-contract-reviewer` theo tên.
2. Reviewer đọc hợp đồng của gói và chạy validator đi kèm.
3. Phản hồi chứa một báo cáo kiểm tra không có lỗi cấu trúc cho
   mẫu, kèm một lựa chọn primitive có lý lẽ.

Bằng chứng thực thi cũng phải nêu đường dẫn script, đường dẫn đích, cwd,
vector đối số chính xác và exit code. Một báo cáo trôi chảy mà thiếu
những trường đó không chứng minh rằng script đi kèm đã thực sự chạy.

Nếu host báo skill không khả dụng, hãy kiểm tra lại đích cài,
quét lại hoặc khởi động lại một lần, rồi thử lại yêu cầu tường minh. Đừng
viết lại phần mô tả skill để che một thất bại cài đặt.

### 4. Thăm dò chọn lựa ngầm

Bắt đầu một lượt agent mới và nhập cùng tác vụ mà không nêu tên skill:

```text
Review <TARGET_ROOT>/my-first-skill as a reusable agent package and tell me whether its package contract is valid.
```

Nếu host có hiển thị skill được chọn, hãy ghi lại xem nó có chọn
`skill-contract-reviewer` không. Nếu host không hiển thị routing, hãy
đánh dấu chọn lựa ngầm là chưa kiểm chứng. Gọi tường minh là cách dự phòng portable.

### 5. Dọn dẹp

Chỉ gỡ bộ reviewer đã cài:

```bash
npx skills remove skill-contract-reviewer
```

Chọn đúng host và scope đã dùng khi cài. Sau khi quét lại hoặc mở
phiên mới, một yêu cầu tường minh cho `skill-contract-reviewer` sẽ phải báo
là không khả dụng. Hãy giữ `my-first-skill` cho các bài sau, hoặc gỡ
thư mục lab sau khi bạn học xong track.

## Vấn đề

Giả sử nhóm bạn có một quy trình phát hành đáng tin cậy. Nó tìm các thay đổi đã merge, kiểm tra ghi chú migration, cập nhật changelog, chạy một lệnh packaging (đóng gói) và tạo một checklist review.

Đặt quy trình đó vào một prompt thì dễ dán nhưng khó vận hành. Prompt không có danh tính ổn định, không có quy tắc khám phá, không có ranh giới tài nguyên, không có hình dạng gói kiểm thử được, và không trả lời được những câu hỏi cơ bản: Ai được gọi nó? Khi nào mô hình nên chọn nó? Nó chạy được script nào? Tệp nào đáng tin? Điều gì sống sót khi ngữ cảnh bị nén?

Sai lầm ngược lại là coi mọi hướng dẫn tái sử dụng là một skill. Quy ước trong repo, tự động hóa tất định, công cụ ngoài, hook sự kiện và agent được ủy quyền giải quyết những vấn đề khác nhau. Đóng gói tất cả vào `SKILL.md` tạo ra một thư mục trông như portable nhưng lại phụ thuộc vào hành vi không được ghi lại của một host.

Nhiệm vụ kỹ thuật đầu tiên là phân loại. Hãy quyết định artifact là gì trước khi quyết định đóng gói nó thế nào.

## Khái niệm

### Skill mã hóa tri thức thủ tục

Một agent skill là một thư mục có điểm vào là `SKILL.md`. Tệp điểm vào chứa frontmatter (phần khai báo đầu tệp) YAML, theo sau là hướng dẫn Markdown. Thư mục cũng có thể chứa references, script và asset.

```figure
skill-package-anatomy
```

Đơn vị triển khai được là cả thư mục, không phải chỉ tệp Markdown. Một `SKILL.md` bị sao chép mà thiếu references là một gói hỏng, kể cả khi frontmatter của nó parse được.

### Những abstraction lân cận

| Artifact | Việc chính | Được nạp hoặc chạy khi | Nó không nên giả dạng |
|---|---|---|---|
| Prompt | Định hình một tương tác với mô hình | Được ứng dụng hoặc người dùng đưa vào | Một gói có phiên bản kèm tài nguyên |
| Hướng dẫn trong repo | Giải thích các quy tắc thường trực của một codebase | Một runtime lập trình bước vào scope đó | Một quy trình tác vụ tái sử dụng |
| Agent skill | Cung cấp tri thức thủ tục tái sử dụng | Kích hoạt tường minh hoặc ngầm | Một ranh giới ủy quyền cứng |
| Công cụ MCP | Phơi ra một năng lực từ xa có kiểu | Mô hình hoặc ứng dụng gọi nó | Một quy trình vận hành chi tiết |
| Hook | Chạy logic tất định trên một sự kiện | Sự kiện đã khai báo xảy ra | Routing mô hình theo xác suất |
| Subagent | Ủy quyền công việc với ngữ cảnh và trạng thái riêng | Một orchestrator tạo hoặc gọi nó | Một bó hướng dẫn tĩnh |
| Plugin | Phân phối một phần mở rộng runtime lớn hơn | Host cài hoặc bật nó | Bản thân hợp đồng skill portable |
| Thư viện skill học được | Lưu hành vi khám phá qua trải nghiệm | Một policy lấy lại chương trình hoặc trajectory trước đó | Một gói `SKILL.md` theo chuẩn |

Một skill phát hành có thể nói cho agent cách kiểm tra một bản phát hành. Một MCP server có thể phơi ra registry phát hành. Một hook có thể cấm push trực tiếp. Một subagent có thể tự kiểm toán ứng viên. Những mảnh này ghép lại được vì chúng giữ trách nhiệm khác nhau.

### Từ "skill" chỉ hai ý tưởng khác nhau

Các hệ nghiên cứu đôi khi gọi một chương trình học được, một trajectory thành công hay một mảnh policy riêng cho môi trường là skill. Một agent có thể tạo những artifact này trong lúc khám phá, lấy lại chúng theo độ tương đồng của tác vụ, thực thi chúng và sửa thư viện từ phản hồi. Giai đoạn 14 · 10 xây loại thư viện học suốt đời đó.

Một Agent Skill trong mini-track này thì khác. Nó là một gói được viết ra với hợp đồng hệ thống tệp đã khai báo, metadata danh mục, progressive disclosure (tiết lộ dần), việc gọi qua runtime trung gian và công cụ do host kiểm soát. Nó có thể được agent tạo hoặc cải thiện, nhưng định dạng không bắt buộc phải học.

| Khía cạnh | Gói Agent Skill | Thư viện skill học được |
|---|---|---|
| Đơn vị chính | Thư mục `SKILL.md` | Chương trình, policy, trajectory hay bản ghi bộ nhớ |
| Cách tạo | Được viết, sinh ra hoặc tuyển chọn | Thường khám phá từ trải nghiệm môi trường |
| Chọn lựa | Mô tả trong danh mục cộng policy runtime | Truy hồi hoặc policy trên trạng thái tác vụ |
| Thực thi | Mô hình làm theo hướng dẫn và gọi công cụ của host | Môi trường chạy một hành vi hoặc artifact code đã lưu |
| Tính portable | Hợp đồng gói có thể đi qua các host tương thích | Thường gắn với một môi trường và không gian hành động |
| Đánh giá | Routing, artifact, an toàn và tương thích host | Phần thưởng, tỉ lệ thành công, chuyển giao và tăng trưởng thư viện |

Cả hai ý tưởng đều đóng gói năng lực tái sử dụng. Chúng không nên dùng chung các tuyên bố triển khai chỉ vì trùng tên.

### Phần lõi portable

Đặc tả Agent Skills yêu cầu hai trường frontmatter:

```yaml
---
name: release-readiness
description: Inspect a release candidate when the user asks whether a version is ready to publish.
---
```

`name` là định danh ổn định. Nó phải thỏa quy tắc đặt tên của đặc tả và khớp với thư mục cha. `description` vừa là tài liệu vừa là metadata routing. Nó nên nói skill làm gì và áp dụng khi nào.

Các trường tùy chọn portable là:

| Trường | Mục đích | Ghi chú portable |
|---|---|---|
| `license` | Nêu điều khoản cho gói | Đặc tả lõi |
| `compatibility` | Nêu yêu cầu môi trường | Đặc tả lõi |
| `metadata` | Mang dữ liệu mở rộng dạng chuỗi | Đặc tả lõi |
| `allowed-tools` | Gợi ý công cụ đã được duyệt trước | Thử nghiệm; hỗ trợ tùy host |

Phần thân Markdown chứa hướng dẫn vận hành. Nó nên định nghĩa quy trình, các điểm quyết định, hành vi khi thất bại và đường dẫn trực tiếp tới tài nguyên hỗ trợ.

```markdown
# Release readiness

Use this workflow for a release candidate, not for ordinary development builds.

1. Read `references/release-policy.md`.
2. Run `python3 scripts/inspect_release.py --format json`.
3. Stop if the report contains a blocking failure.
4. Produce the checklist from `assets/release-checklist.md`.
5. Ask for approval before any publish or tag action.
```

### Phần mở rộng runtime là lớp thứ hai

Một số host chấp nhận frontmatter thêm hoặc cấu hình đi kèm. Những trường đó có thể hữu ích, nhưng chúng không tự động portable.

| Hành vi | Ví dụ phần mở rộng của host | Lõi portable? |
|---|---|:---:|
| Ẩn một skill khỏi routing của mô hình nhưng vẫn cho người dùng gọi trực tiếp | `disable-model-invocation` | Không |
| Ẩn một skill khỏi menu lệnh của người dùng nhưng vẫn cho mô hình routing | `user-invocable` | Không |
| Hiện trợ giúp đối số trong menu lệnh | `argument-hint` | Không |
| Chạy skill trong ngữ cảnh được ủy quyền | `context`, `agent` | Không |
| Ghim cài đặt mô hình hoặc mức suy luận | `model`, `effort` | Không |
| Đăng ký tự động hóa vòng đời | `hooks` | Không |
| Tắt gọi ngầm trong Codex | policy `agents/openai.yaml` | Không |

Hãy coi mỗi phần mở rộng là một adapter. Giữ quy trình lõi hợp lệ khi không có nó, ghi lại cách dự phòng và kiểm thử host dùng nó. Một runtime có thể bỏ qua trường lạ, từ chối nó, hoặc giữ nó mà không triển khai hành vi.

### Frontmatter là metadata có thể thực thi

Metadata thay đổi hành vi hệ thống trước khi phần thân skill được đọc.

- Một `name` sai định dạng có thể làm khám phá thất bại.
- Một `description` mơ hồ có thể route sai yêu cầu.
- Một cờ chỉ dành cho người có thể gỡ skill khỏi danh mục của mô hình.
- Một danh sách công cụ được phép có thể thay đổi việc host có xin quyền hay không.
- Một cài đặt ngữ cảnh có thể đẩy thực thi sang một phiên agent riêng.

Hãy review frontmatter như code cấu hình. Kiểm tra nó, đánh phiên bản cho nó, và đưa hành vi của nó vào eval.

### Vòng đời của skill

```figure
skill-runtime-lifecycle
```

Mỗi mũi tên là một runtime boundary (ranh giới thực thi) với những kiểu thất bại riêng.

1. **Khám phá** tìm các gói khả dĩ ở những vị trí đã cấu hình.
2. **Kiểm tra** loại các gói sai định dạng hoặc không an toàn trước khi công bố danh mục.
3. **Lập danh mục** phơi ra một `name` và `description` ngắn gọn, không phải toàn bộ gói.
4. **Chọn lựa** quyết định skill có liên quan hay không.
5. **Kích hoạt** nạp phần thân vào ngữ cảnh mà mô hình thấy được.
6. **Tiết lộ** chỉ đọc references hoặc asset khi một nhánh cần tới chúng.
7. **Thực thi** dùng công cụ của host theo quy tắc quyền và cô lập của host.
8. **Kiểm chứng** kiểm tra artifact tạo ra một cách độc lập với tuyên bố của mô hình.

Gộp các giai đoạn này lại sinh ra mô hình tư duy sai. Một skill đã khám phá không phải là đã kích hoạt. Một skill đã kích hoạt không được phép làm mọi thứ nó mô tả. Một lời gọi công cụ được cho phép không phải bằng chứng rằng kết quả đúng.

### Skill và công cụ trực giao nhau

MCP trả lời câu hỏi: "Ứng dụng này gọi được những năng lực nào, và lược đồ của chúng ra sao?" Một skill trả lời: "Agent nên tiếp cận lớp tác vụ này thế nào?"

```figure
skill-tool-orthogonality
```

Skill có thể nêu tên một công cụ, nhưng host sở hữu registry năng lực thực tế. Nếu công cụ vắng mặt, skill nên nêu cách dự phòng hoặc thất bại rõ ràng. Nó không bao giờ nên ngụ ý rằng nêu tên một năng lực là tạo ra nó.

### Skill và hướng dẫn trong repo là hai scope khác nhau

Hướng dẫn trong repo mô tả môi trường bạn đang ở: lệnh, quy ước, tệp sinh tự động và ranh giới. Một skill cung cấp thủ tục tái sử dụng cho một tác vụ có thể xảy ra ở nhiều repo.

Khi cả hai cùng áp dụng, yêu cầu hiện tại của người dùng và quy tắc repo ràng buộc skill. Một skill refactor chung chung không được vượt qua quy tắc repo cấm sửa tệp sinh tự động.

### Skill không import lẫn nhau

Một skill có thể hướng dẫn agent gọi một skill khác, nhưng đây không phải import ở mức ngôn ngữ. Skill thứ hai vẫn phải đi qua khám phá runtime, điều kiện hợp lệ, kích hoạt, quyền và xử lý ngữ cảnh.

Hãy viết phụ thuộc chéo giữa các skill như những cạnh quy trình quan sát được:

```markdown
After producing the candidate changelog, invoke the `release-risk-review` skill.
Pass the candidate path and require a blocking or non-blocking verdict.
If that skill is unavailable, stop and report the missing dependency.
```

Điều này làm phụ thuộc kiểm thử được và cho host cơ hội thực thi policy.

## Tự tay làm

`code/main.py` hiện thực một validator nhỏ hướng theo chuẩn và một bộ chọn artifact. Nó chỉ dùng stdlib để mọi quy tắc đều nhìn thấy được.

Validator phơi ra:

- `parse_frontmatter(text)` để tách metadata khỏi phần thân.
- `validate_skill_text(text, directory_name, allowed_runtime_extensions=())` để kiểm tra trường bắt buộc, cách đặt tên, phần mở rộng lạ, sự có mặt của phần thân và các giới hạn portable.
- `ValidationIssue` và `SkillReport` để trả bằng chứng có cấu trúc thay vì một giá trị boolean mờ nghĩa.
- `FrontmatterSyntaxError` cho input không thể diễn giải an toàn.

Bộ chọn phơi ra `TaskShape` và `select_primitives(task)`. Nó ánh xạ nhu cầu của một tác vụ sang code thường, hướng dẫn trong repo, skill, hook, subagent hoặc công cụ MCP.

Chạy lab:

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/22-skills-and-agent-sdks
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Khối lệnh này cần một bản clone cục bộ và phải bắt đầu từ bất kỳ đâu bên trong
bản clone đó để `git rev-parse --show-toplevel` resolve được gốc repo.

Bản demo in JSON cho một skill portable hợp lệ, một skill có phần mở rộng của host, một gói không hợp lệ và vài quyết định về hình dạng tác vụ. Hãy xem các mã issue. Một validator gói nên giải thích cách sửa artifact mà không đoán thay tác giả.

### Thứ tự kiểm tra quan trọng

Hãy kiểm tra các sự kiện cấu trúc rẻ tiền trước những quy tắc nội dung sâu hơn:

```figure
skill-validation-order
```

Thứ tự này ngăn các lỗi thứ cấp che khuất bất biến đầu tiên bị phá.

## Áp dụng

Trước khi viết một skill, hãy điền thẻ quyết định này:

| Câu hỏi | Nếu có | Primitive khả dĩ |
|---|---|---|
| Việc này có cần phán đoán tái sử dụng của mô hình qua nhiều bước? | Thủ tục ổn định nhưng quyết định thay đổi | Skill |
| Việc này có phải xảy ra mỗi khi một sự kiện kích hoạt? | Bỏ lỡ một lần thực thi là không chấp nhận được | Hook hoặc code ứng dụng |
| Mô hình có cần một năng lực ngoài với input có kiểu? | Thao tác nằm ngoài ngữ cảnh mô hình | Công cụ hoặc MCP server |
| Công việc có cần ngữ cảnh, trạng thái hay quyền sở hữu riêng? | Một worker riêng trả về kết quả có giới hạn | Subagent |
| Hướng dẫn này có riêng cho một repo? | Nó mô tả lệnh và ràng buộc cục bộ | Hướng dẫn trong repo |
| Một tương tác là đủ? | Không cần vòng đời gói | Prompt |

Nhiều quy trình production dùng hơn một dòng. Thẻ này ngăn một artifact giả vờ cung cấp mọi thuộc tính.

## Đóng gói

Bài này tạo ra bộ `skill-contract-reviewer` trong `outputs/`. Nó chứa:

- một `SKILL.md` portable để review một gói skill được đề xuất;
- checklist tham chiếu cho hợp đồng portable và việc chọn primitive;
- một script kiểm tra tất định;
- fixture hình dạng tác vụ bao phủ prompt, skill, công cụ, hook, code thường và subagent.

Hãy cài toàn bộ gói, không chỉ tệp điểm vào:

```bash
cd "$(git rev-parse --show-toplevel)"
python3 scripts/install_skills.py /tmp/aiefs-skills --phase 13 --type skill
```

Installer của khóa học báo từng skill Giai đoạn 13 đã sao chép và ghi
`/tmp/aiefs-skills/manifest.json`. Đích sạch này kiểm tra hình dạng gói;
vòng thành công đầu tiên ở trên kiểm tra khám phá và việc gọi trên một host thật.

Các bài sau đi sâu vào từng giai đoạn vòng đời. Bài 24 xây khám phá và progressive disclosure. Bài 25 xây policy gọi và routing. Bài 26 tách quyền khỏi sandbox. Bài 27 biến toàn bộ gói thành một artifact phát hành được đánh giá.

## Bài tập

1. Phân loại năm quy trình từ nhóm của bạn bằng `TaskShape`. Biện luận mọi trường hợp bạn chọn hơn một primitive.
2. Thêm kiểm thử biên chứng minh rằng giá trị `compatibility` 500 ký tự pass và giá trị 501 ký tự fail như một lỗi đặc tả.
3. Thêm một phần mở rộng runtime vào allowlist. Viết một test chứng minh cùng tệp đó vẫn phân biệt được với một skill chỉ portable.
4. Tách một prompt 400 dòng thành `SKILL.md`, một reference, một hợp đồng script và một template output. Giữ mỗi tệp chịu trách nhiệm một loại thông tin.
5. Thiết kế phản ứng khi thất bại cho một skill tham chiếu tới công cụ MCP không khả dụng. Đừng lặng lẽ thay bằng một công cụ có quyền rộng hơn.
6. Review một skill có sẵn và gắn nhãn mỗi câu là routing, thủ tục, policy, con trỏ tham chiếu hay hợp đồng output. Chuyển đi những gì không thuộc chỗ đó.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| Agent skill | "Một prompt đã lưu" | Một thư mục hướng dẫn thủ tục có thể khám phá, kèm tài nguyên tùy chọn |
| Portable core | "Những trường mọi runtime đều có" | Hợp đồng do đặc tả Agent Skills định nghĩa |
| Runtime extension | "Frontmatter thêm" | Cấu hình riêng của host mà hành vi cần một adapter tương thích |
| Activation | "Skill đã chạy" | Phần thân skill đã vào ngữ cảnh mô hình thấy được; thực thi có thể đến sau |
| Skill dependency | "Import một skill khác" | Một cạnh gọi qua runtime trung gian kèm kiểm tra khả dụng và policy |
| Tool contract | "Một lược đồ hàm" | Input, output, quyền, tác dụng phụ, lỗi và bằng chứng cho một năng lực |

## Đọc thêm

- [Đặc tả Agent Skills](https://agentskills.io/specification) cho hợp đồng thư mục và frontmatter portable.
- [Thực hành tốt nhất cho Agent Skills](https://agentskills.io/skill-creation/best-practices) cho phạm vi, hướng dẫn và cách tổ chức tài nguyên.
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills) cho hành vi khám phá và gọi hiện tại của Codex.
- [Claude Code skills](https://code.claude.com/docs/en/skills) cho phần mở rộng gọi, đối số, công cụ và ngữ cảnh ủy quyền của một runtime.

---

[← Bài trước: Lớp routing (định tuyến) LLM](../../21-llm-routing-layer/docs/vi.md) · [Bài sau: Bài tổng kết (capstone): Hệ sinh thái công cụ (tool ecosystem) không trạng thái →](../../23-capstone-tool-ecosystem/docs/vi.md)
