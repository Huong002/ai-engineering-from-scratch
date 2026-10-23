# Khám phá skill và tiết lộ dần

> Một skill trở nên hữu ích trước khi phần thân của nó được nạp. Tên và mô tả của nó giành được một chỗ trong skill catalog (danh mục kỹ năng); các file sâu hơn chỉ giành được ngữ cảnh khi tác vụ chạm tới chúng.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 13, Bài 22 (Agent Skills: hợp đồng di động và ranh giới thực thi)
**Time:** ~105 phút

## Mục tiêu học tập

- Xây một pipeline discovery (khám phá) trên filesystem, tách biệt phạm vi, kiểm tra hợp lệ, chính sách xung đột, và việc công bố danh mục.
- Giải thích ba mức tiết lộ: metadata (siêu dữ liệu) của danh mục, chỉ dẫn đang hoạt động, và tài nguyên riêng cho từng tác vụ.
- Thiết kế các tham chiếu để agent chạm thẳng tới chi tiết cần thiết mà không phải nạp toàn bộ gói.
- Lập ngân sách cho không gian danh mục độc lập với ngữ cảnh của skill đang hoạt động.
- Từ chối path traversal và symlink escape khi skill đọc tài nguyên của chính nó.

## Vấn đề

Agent của bạn có 200 skill đã cài. Nạp mọi `SKILL.md`, file tham chiếu, script và template ngay đầu phiên sẽ chôn vùi tác vụ hiện tại trong những thủ tục không liên quan. Không nạp gì thì buộc người dùng phải nhớ chính xác đường dẫn filesystem.

Thoả hiệp thường thấy là một danh mục: cho model thấy một bản sắc gọn và routing hint (gợi ý định tuyến) cho từng skill đủ điều kiện, rồi chỉ nạp toàn bộ phần thân sau khi đã chọn. Điều đó tạo ra hai bài toán kỹ thuật mới.

Thứ nhất, discovery không chỉ là tìm file đệ quy. Skill có thể nằm ở phạm vi project, user, administrator, plugin, hoặc built-in. Hai gói có thể trùng tên. Một symlink có thể trỏ ra ngoài gốc tin cậy. Một gói sai định dạng có thể chiếm chỗ trong danh mục hoặc trở nên không thể gọi được.

Thứ hai, progressive disclosure (tiết lộ dần) có thể biến thành mơ hồ dần. Nếu `SKILL.md` nói "đọc hướng dẫn liên quan" mà gói có mười hai hướng dẫn, model phải đoán. Nếu mỗi hướng dẫn lại trỏ tới ba file nữa, việc nạp trở thành một cuộc duyệt đồ thị không có biên.

Một runtime tốt làm cho discovery mang tính tất định và việc tiết lộ có chủ đích.

## Khái niệm

### Discovery là một pipeline biên dịch

Hãy coi filesystem như đầu vào nguồn. Đừng công bố đường dẫn thô trực tiếp cho model.

```figure
skill-discovery-pipeline
```

Mỗi chặng nên sinh ra dữ liệu có cấu trúc và lỗi có cấu trúc. Nhật ký discovery nên trả lời:

- Những gốc nào đã được tìm?
- Những ứng viên nào đã được tìm thấy?
- Những ứng viên nào bị loại, và vì sao?
- Gói nào thắng khi xảy ra xung đột?
- Mục danh mục nào bị rút gọn hoặc bỏ đi vì ngân sách?

Không có bằng chứng đó, câu "model không dùng skill của tôi" gần như không thể chẩn đoán.

### Phạm vi là chính sách runtime

Đặc tả di động định nghĩa một gói skill, không định nghĩa một đường cài đặt phổ quát hay một thứ tự ưu tiên chung. Host quyết định nơi nó tìm kiếm.

Một runtime tổng quát có thể dùng các phạm vi sau:

| Phạm vi | Thư mục gốc ví dụ | Quyền sở hữu dự kiến |
|---|---|---|
| Workspace | `<repo>/.agents/skills/` | Người duy trì project |
| User | `<user-data>/skills/` | Một lập trình viên |
| Administrator | `<system>/skills/` | Chính sách máy hoặc tổ chức |
| Plugin | Một gói plugin đã ký | Nhà phát hành và người cài plugin |
| Built-in | Gói runtime | Nhà cung cấp runtime |

Tính đến tháng 8 năm 2026, Codex mô tả việc discovery trong project từ `$CWD/.agents/skills` qua các thư mục cha lên tới gốc repository, cộng với các vị trí user, administrator, và built-in. Nó hỗ trợ thư mục skill là symlink. Các tên trùng có thể cùng xuất hiện thay vì được gộp. Đó là hành vi của Codex, không phải yêu cầu của `SKILL.md`; hãy kiểm tra [tài liệu skill của Codex](https://learn.chatgpt.com/docs/build-skills) hiện hành khi viết adapter.

Đừng bao giờ suy ra thứ tự ưu tiên từ tên thư mục. Hãy công bố nó như chính sách và kiểm thử nó. Lab của bài dùng một hạng số nguyên tường minh cho mỗi `Scope` để cùng một tập ứng viên luôn được giải quyết giống nhau.

### Xung đột cần bản sắc vượt ra ngoài `name`

Hai gói cùng tên `release-readiness` có thể đều hợp lệ. Một gói có thể là bản ghi đè ở workspace, gói kia là mặc định của user. Vì thế một mục danh mục cần ít nhất:

```json
{
  "name": "release-readiness",
  "description": "Inspect a release candidate for this repository.",
  "scope": "workspace",
  "source": "/repo/.agents/skills/release-readiness",
  "selected": true
}
```

Các chính sách xung đột thường gặp:

| Chính sách | Lợi ích | Rủi ro |
|---|---|---|
| Giữ mọi ứng viên | Không che giấu gì | Model thấy các tên mơ hồ |
| Phạm vi ưu tiên cao nhất thắng | Gọi đơn giản | Một gói cục bộ có thể che gói tin cậy |
| Từ chối bản trùng | Không che khuất âm thầm | Các bản ghi đè hợp lệ ngừng hoạt động |
| Định danh tên theo nguồn | Bản sắc tường minh | Tên hiển thị cho người dùng dài hơn |

Hãy chọn một chính sách cho host. Giữ lại các ứng viên bị loại hoặc bị che trong chẩn đoán ngay cả khi chúng không có trong danh mục mà model thấy.

### Ba mức tiết lộ

Đặc tả Agent Skills mô tả việc nạp theo chặng. Điểm mấu chốt là mỗi mức có một mục đích khác nhau.

```figure
skill-disclosure-levels
```

#### Mức 1: metadata của danh mục

Model cần đủ thông tin để phân biệt skill này với các skill lân cận. Đặc tả ước tính khoảng 100 token cho mỗi mục danh mục, nhưng việc serialize và tokenize thực tế thuộc về host.

Một mô tả hữu ích có hai mệnh đề:

```yaml
description: Validate a release candidate and produce a readiness report. Use when the user asks whether a version, tag, or package is ready to publish.
```

Mệnh đề đầu nêu năng lực. Mệnh đề sau nêu ranh giới kích hoạt. Bài 25 đánh giá ranh giới này bằng các prompt trúng và prompt gần trúng.

#### Mức 2: chỉ dẫn đang hoạt động

Sau khi kích hoạt, phần thân nên hoạt động như một bản đồ và một thủ tục. Đặc tả khuyến nghị giữ `SKILL.md` dưới 500 dòng. Đó là tín hiệu thiết kế, không phải chỉ tiêu để lấp đầy.

Phần thân nên chứa:

- ranh giới tác vụ;
- workflow mặc định;
- các điều kiện rẽ nhánh;
- tham chiếu trực tiếp tới các file sâu hơn;
- hợp đồng của tool và script;
- hành vi khi lỗi và khi dừng;
- kết quả mong đợi và cách kiểm chứng nó.

Đừng chuyển workflow cốt lõi vào một tham chiếu chỉ để làm file đầu vào ngắn lại. Kích hoạt phải cho model đủ ngữ cảnh để bắt đầu đúng.

#### Mức 3: tài nguyên hỗ trợ

Tham chiếu cung cấp văn bản hoặc dữ liệu. Script cung cấp tính toán tất định. Asset được sao, điền, hoặc biến đổi thành sản phẩm bàn giao chứ không được coi là chỉ dẫn.

| Thư mục | Model có đọc? | Model có thực thi? | Nội dung điển hình |
|---|:---:|:---:|---|
| `references/` | Có, khi cần | Không | schema, chính sách, hướng dẫn chuyên môn |
| `scripts/` | Có thể xem | Qua một tool được phép | validator, converter, collector |
| `assets/` | Chỉ khi hữu ích | Không | template, fixture, ảnh, file khởi đầu |

Những tên này là quy ước, không phải năng lực thần kỳ. Host vẫn cần quyền truy cập file và một tool thực thi.

### Tham chiếu theo nhánh tốt hơn đổ cả chủ đề

Viết file đầu vào như một bản đồ quyết định:

```markdown
## Choose the path

- For a Python package, read `references/python-release.md`.
- For a container image, read `references/container-release.md`.
- For a documentation-only release, read `references/docs-release.md`.
- If the release combines artifact types, read only the guides for those artifacts.
```

Điều này cho mỗi tham chiếu một điều kiện nạp quan sát được. "Đọc `references/` để biết thêm" thì không.

Giữ đồ thị tham chiếu nông. Hướng dẫn chính thức khuyến nghị liên kết trực tiếp từ `SKILL.md` và tránh chuỗi sâu. Một bước nhảy làm khả năng tiếp cận kiểm thử được và giảm nguy cơ một ràng buộc cần thiết không bao giờ vào ngữ cảnh.

```figure
skill-reference-map
```

### Ngân sách danh mục và ngữ cảnh hoạt động là hai ngân sách khác nhau

Gọi `c_i` là chi phí danh mục đã serialize của skill `i`, `B_c` là ngân sách token (token budget) của danh mục, `b_j` là chi phí phần thân đang hoạt động, và `r_k` là các tài nguyên thực sự được nạp.

```text
catalog_cost = sum(c_i for every published skill)
active_cost = sum(b_j for every activated skill) + sum(r_k for every disclosed resource)
```

Giảm một ngân sách không tự động giảm ngân sách kia. Mô tả ngắn có thể tiết kiệm chỗ trong danh mục trong khi phần thân 900 dòng đã kích hoạt vẫn nhấn chìm tác vụ. Chia phần thân thành các tham chiếu chỉ giảm chi phí hoạt động khi runtime và chỉ dẫn thực sự tránh nạp những nhánh không liên quan.

Codex hiện cấp ngân sách cho danh sách skill ban đầu là 2 phần trăm cửa sổ
ngữ cảnh khi đã biết kích thước cửa sổ ngữ cảnh. Giá trị 8.000 ký tự là phương án
dự phòng chỉ khi chưa biết kích thước đó; nó không phải một giới hạn thứ hai cộng dồn
với quy tắc 2 phần trăm. Khi danh mục vượt ngân sách áp dụng, mô tả có thể bị rút
gọn hoặc bỏ đi. Hãy coi những con số đó là chính sách Codex hiện hành, không phải
thuộc tính của chuẩn Agent Skills.

### Đường dẫn tài nguyên là một ranh giới tin cậy

Skill chỉ nên đọc các file bên trong gói của nó. Kiểm tra tiền tố chuỗi theo nghĩa đen là không đủ:

```text
references/../../../../.ssh/config
references/external-link -> /private/company-secrets
```

Hãy giải quyết gốc gói và ứng viên theo ngữ nghĩa filesystem, từ chối đầu vào tuyệt đối, và kiểm chứng rằng ứng viên sau khi giải quyết vẫn nằm dưới gốc đã giải quyết. Quyết định xem có cho phép symlink trước khi discovery. Nếu cho phép, kiểm tra đích đã giải quyết mỗi lần.

```figure
skill-resource-containment
```

Việc giữ đường dẫn trong biên không thiết lập được niềm tin vào nội dung. Một tham chiếu hợp lệ trong gói vẫn có thể chứa chỉ dẫn độc hại. Bài 26 xử lý mối đe dọa đó.

### Việc nạp phải quan sát được

Ghi lại các sự kiện tiết lộ mà không log bí mật:

```json
{
  "event": "skill.resource.loaded",
  "skill": "release-readiness",
  "resource": "references/python-release.md",
  "reason": "candidate contains pyproject.toml",
  "bytes": 2840
}
```

Lý do biến một lựa chọn ngữ cảnh thành bằng chứng có thể xem xét. Nó cũng giúp nhận ra những chỉ dẫn khiến agent nạp mọi file "để phòng trường hợp".

## Tự tay làm

`code/main.py` xây một engine discovery và tiết lộ mang tính tất định.

Bề mặt discovery gồm:

- `Scope` cho metadata nguồn và thứ tự ưu tiên;
- `SkillCandidate` cho một ứng viên filesystem chưa được kiểm tra hợp lệ;
- `discover_scope(scope)` để liệt kê các thư mục skill trực tiếp;
- `resolve_collisions(candidates, precedence)` để áp một chính sách đã công bố;
- `CatalogEntry` và `build_catalog(...)` để công bố metadata có biên;
- `CatalogBudget` để tính các mục đã serialize mà không giả vờ ký tự là token phổ quát.

Bề mặt tiết lộ gồm:

- `load_skill_body(entry, ...)` cho việc kích hoạt Mức 2;
- `validate_reference(skill_dir, reference)` cho việc giữ đường dẫn trong biên;
- `load_reference(...)` cho các lần đọc Mức 3 có biên.

Chạy lab:

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Khối này cần một bản clone cục bộ và tự tìm gốc repository từ bất kỳ
thư mục làm việc nào bên trong bản clone đó.

Demo tạo các phạm vi project và user tạm thời, chèn một xung đột, dựng một danh mục dưới ngân sách cố ý nhỏ, kích hoạt một skill, và thử cả một lần đọc tham chiếu hợp lệ lẫn một lần thoát bằng traversal. Không cài file vĩnh viễn nào.

### Vì sao discovery chỉ nông

`discover_scope` chỉ kiểm tra các thư mục con trực tiếp để tìm `SKILL.md`. Nó không đệ quy coi mọi `SKILL.md` lồng nhau là một gói riêng. Điều này giữ nguyên ranh giới gói và tránh vô tình công bố ví dụ hay fixture bên trong một skill đã cài.

### Vì sao lab không phân tích YAML tuỳ ý

Lab hỗ trợ frontmatter dạng scalar cần cho danh mục của nó. Một runtime production nên dùng YAML parser an toàn với schema tường minh, giới hạn kích thước, và tắt việc tạo đối tượng tuỳ biến. "Chỉ dùng stdlib" là một ràng buộc giảng dạy, không phải giấy phép âm thầm phát minh một phương ngữ YAML thiếu đủ.

## Áp dụng

Hãy áp danh sách kiểm này cho bất kỳ adapter discovery nào:

1. Liệt kê mọi gốc đã cấu hình và ai có thể ghi vào đó.
2. Nêu rõ có cho phép gói là symlink hay không.
3. Kiểm tra tên gói, tên thư mục, metadata bắt buộc, và kích thước phần thân đầu vào.
4. Giữ nguồn và phạm vi trong bản sắc nội bộ.
5. Công bố và kiểm thử hành vi với tên trùng.
6. Đo chính xác danh mục đã serialize gửi cho model.
7. Ghi lại lý do một phần thân hoặc một tài nguyên được nạp.
8. Giữ việc đọc tài nguyên bên trong gốc gói đã giải quyết.
9. Báo lỗi rõ ràng khi một file được tham chiếu bị thiếu.
10. Dựng lại danh mục khi các bản cài hoặc chính sách thay đổi.

## Đóng gói

Bài này tạo ra gói `skill-catalog-builder`. Nó quét các gốc được sắp thứ tự tường minh, từ chối file đầu vào là symlink và tên không khớp thư mục, giải quyết xung đột giữa các phạm vi, từ chối bản trùng cùng ưu tiên, và đưa metadata đã chọn vào ngân sách mục, mô tả, và ký tự serialize đã công bố.

Báo cáo JSON của nó chứa các mục đã chọn, ứng viên bị che, mục bị bỏ, lỗi kiểm tra hợp lệ, thứ tự ưu tiên, và mức dùng ngân sách. Việc nạp phần thân và tham chiếu vẫn là các thao tác runtime riêng, nên catalog builder không thực thi script hay đưa cả gói vào ngữ cảnh.

## Bài tập

1. Thêm một phạm vi plugin và đặt nó giữa ưu tiên của user và built-in. Chứng minh kết quả xung đột bằng một test.
2. Đổi chính sách xung đột từ ưu tiên cao nhất sang định danh tên. Giữ cả hai mục trong danh mục.
3. Thêm giới hạn kích thước byte cho `load_reference`. Kiểm thử một file đúng bằng giới hạn và một file hơn một byte.
4. Tạo hai mô tả nghe gần như giống nhau. Viết lại để ranh giới kích hoạt không chồng lấn.
5. Thêm một manifest chứa hash cho mọi tham chiếu và script. Phát hiện tài nguyên bị sửa trước khi nạp nó.
6. Gắn đo đạc vào demo để báo riêng số byte của Mức 1, Mức 2 và Mức 3.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| Skill discovery | "Tìm mọi SKILL.md" | Tìm trong các phạm vi đã cấu hình, kiểm tra hợp lệ gói, gắn nguồn gốc, và áp chính sách |
| Skill catalog | "Danh sách skill đã cài" | Routing metadata gọn mà model thấy cho các gói đủ điều kiện |
| Collision policy | "Bản trùng nào thắng" | Một quy tắc đã công bố cho các ứng viên cùng tên từ các nguồn khác nhau |
| Progressive disclosure | "Nạp lười" | Đưa ngữ cảnh vào theo chặng: từ danh mục tới phần thân tới tài nguyên riêng theo nhánh |
| Reference graph | "Các file được skill liên kết" | Cấu trúc tài nguyên tiếp cận được và điều kiện nạp của nó |
| Path containment | "Ở yên trong thư mục" | Kiểm chứng đích tài nguyên sau khi giải quyết vẫn nằm trong gốc gói đã giải quyết |

## Đọc thêm

- [Đặc tả Agent Skills](https://agentskills.io/specification) về hình dạng gói và các mức progressive disclosure.
- [Tối ưu mô tả skill](https://agentskills.io/skill-creation/optimizing-descriptions) về routing metadata của danh mục.
- [Thực hành tốt nhất cho Agent Skills](https://agentskills.io/skill-creation/best-practices) về tham chiếu trực tiếp và kích thước file đầu vào.
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills) về các phạm vi discovery và giới hạn danh mục hiện hành của Codex.

---

[← Bài trước: Bài tổng kết (capstone): Hệ sinh thái công cụ (tool ecosystem) không trạng thái](../../23-capstone-tool-ecosystem/docs/vi.md) · [Bài sau: Gọi kỹ năng và định tuyến →](../../25-skill-invocation-and-routing/docs/vi.md)
