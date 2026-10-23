# Thư viện kỹ năng và học tập suốt đời (Voyager)

> Voyager (Wang và cộng sự, TMLR 2024) coi mã thực thi được như một skill (kỹ năng). Skill có tên, truy xuất được, kết hợp được và được tinh chỉnh nhờ phản hồi từ môi trường. Đây là kiến trúc tham chiếu cho skill của Claude Agent SDK, skillkit và mẫu thư viện kỹ năng của năm 2026.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 07 (MemGPT), Giai đoạn 14, Bài 08 (khối Letta)
**Time:** ~75 phút

## Mục tiêu học tập

- Gọi tên ba thành phần của Voyager — curriculum (lộ trình học) tự động, skill library (thư viện kỹ năng), cơ chế nhắc lặp — và vai trò của từng thành phần.
- Giải thích vì sao Voyager biến không gian hành động thành mã, chứ không phải các lệnh nguyên thuỷ.
- Cài đặt một thư viện kỹ năng bằng thư viện chuẩn, có đăng ký, truy xuất, kết hợp và tinh chỉnh dựa trên thất bại.
- Ánh xạ mẫu của Voyager sang skill của Claude Agent SDK năm 2026 và hệ sinh thái skillkit.

## Vấn đề

Những agent dựng lại mọi năng lực từ đầu trong từng phiên làm sai ba điều:

1. **Lãng phí token.** Mỗi tác vụ lại gợi ra đúng một lập luận như cũ.
2. **Mất tiến độ.** Một chỉnh sửa học được ở phiên A không chuyển sang phiên B.
3. **Hỏng khi kết hợp tầm xa.** Tác vụ phức tạp cần hệ năng lực phân cấp; prompt một phát không diễn đạt được điều đó.

Câu trả lời của Voyager: coi mỗi năng lực reuse (tái sử dụng) như một đoạn mã có tên, lưu trong thư viện, truy xuất theo độ tương đồng, kết hợp được với skill khác và tinh chỉnh nhờ phản hồi khi thực thi.

## Khái niệm

### Ba thành phần

Voyager (arXiv:2305.16291) tổ chức một agent quanh:

1. **Curriculum tự động.** Một bộ đề xuất chạy bằng tò mò chọn tác vụ tiếp theo dựa trên tập skill hiện có và trạng thái môi trường. Việc khám phá diễn ra từ dưới lên.
2. **Skill library.** Mỗi skill là mã thực thi được. Skill mới được thêm khi một tác vụ thành công. Skill được truy xuất theo độ tương đồng giữa truy vấn và mô tả.
3. **Cơ chế nhắc lặp.** Khi thất bại, agent nhận lỗi thực thi, phản hồi môi trường và đầu ra tự verification (kiểm chứng), rồi tinh chỉnh skill.

Đánh giá trên Minecraft (Wang và cộng sự, 2024): nhiều vật phẩm độc nhất gấp 3.3 lần, công cụ đá nhanh hơn 8.5 lần, công cụ sắt nhanh hơn 6.4 lần, đi xa trên bản đồ lâu hơn 2.3 lần so với các baseline. Những con số này gắn riêng với Minecraft, nhưng mẫu thì chuyển được.

### Không gian hành động = mã

Phần lớn agent phát ra các lệnh nguyên thuỷ. Voyager phát ra các hàm JavaScript. Một skill là:

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

Được kết hợp từ các skill con. Được lưu với khoá là mô tả và embedding. Được truy xuất như một chương trình, không phải một prompt.

Đây chính là skill của Claude Agent SDK năm 2026: một đoạn mã có tên, truy xuất được, kèm chỉ dẫn mà agent nạp khi cần.

### Truy xuất skill

Tác vụ mới "chế một cây cúp kim cương". Agent:

1. Nhúng mô tả tác vụ.
2. Truy vấn thư viện kỹ năng để lấy top-k skill tương tự.
3. Truy xuất `craftIronPickaxe`, `mineDiamond`, `placeCraftingTable` v.v.
4. Kết hợp skill mới từ những đơn vị nguyên thuỷ đã truy xuất + logic mới.

Đây là mẫu mà MCP resources (Giai đoạn 13) và skill của Agent SDK cài đặt: truy xuất trên một bề mặt tri thức/mã, giới hạn trong phạm vi tác vụ hiện tại.

### Tinh chỉnh lặp

Vòng phản hồi của Voyager:

1. Agent viết một skill.
2. Skill chạy với môi trường.
3. Một trong ba tín hiệu trả về: `success`, `error` (kèm stack trace), `self-verification failure`.
4. Agent viết lại skill, dùng tín hiệu đó làm ngữ cảnh.
5. Lặp cho tới khi thành công hoặc hết số vòng tối đa.

Đây là Self-Refine (Bài 05) áp dụng cho việc sinh mã với kiểm chứng dựa trên môi trường. CRITIC (Bài 05) là cùng mẫu đó, nhưng lấy công cụ bên ngoài làm bộ kiểm chứng.

### Curriculum và khám phá

Module curriculum của Voyager đề xuất những tác vụ như "dựng một chỗ trú gần hồ" dựa trên thứ agent đã có và thứ nó chưa làm. Bộ đề xuất dùng trạng thái môi trường + danh mục skill để chọn một tác vụ chỉ cao hơn năng lực hiện tại một chút — điểm ngọt của khám phá.

Với agent production, điều này thành một toán tử "còn thiếu gì": cho thư viện kỹ năng hiện tại và một lĩnh vực, những skill nào ta chưa phủ? Các nhóm thường làm việc này thủ công dưới dạng rà soát curriculum.

### Mẫu này hỏng ở đâu

- **Thư viện kỹ năng bị mục.** Cùng một skill được thêm 10 lần với mô tả hơi khác nhau. Hãy khử trùng lặp khi ghi; truy xuất chỉ trả về một.
- **Trôi lệch skill kết hợp.** Skill cha phụ thuộc vào một skill con đã được tinh chỉnh. Hãy đánh version cho skill; skill cha ghim ở v1 không tự dưng nhận v3.
- **Chất lượng truy xuất.** Truy xuất vector trên mô tả skill giảm chất lượng khi thư viện vượt vài trăm mục. Bổ sung bộ lọc tag và ràng buộc cứng ("chỉ skill có `category=tooling`").

```figure
voyager-skills
```

## Tự tay làm

`code/main.py` cài đặt một thư viện kỹ năng bằng thư viện chuẩn:

- `Skill` — tên, mô tả, mã (dạng chuỗi), version, tag, phụ thuộc.
- `SkillLibrary` — đăng ký, tìm kiếm (trùng lặp token), kết hợp (sắp xếp topo các phụ thuộc) và tinh chỉnh (tăng version khi cập nhật).
- Một agent kịch bản đăng ký ba skill nguyên thuỷ, kết hợp skill thứ tư, gặp một lỗi rồi tinh chỉnh.

Chạy thử:

```
python3 code/main.py
```

Vết chạy cho thấy các lần ghi vào thư viện, truy xuất, kết hợp, một lần thực thi thất bại và một lần tinh chỉnh lên v2 — vòng lặp của Voyager từ đầu đến cuối.

## Áp dụng

- **Skill của Claude Agent SDK** (Anthropic) — tham chiếu năm 2026: mỗi skill có mô tả, mã và chỉ dẫn; được nạp khi cần trong phiên của agent.
- **skillkit** (npm: skillkit) — quản lý skill xuyên agent cho hơn 32 agent lập trình AI.
- **Thư viện kỹ năng tự tạo** — theo từng lĩnh vực (skill SQL cho agent dữ liệu, skill Terraform cho agent hạ tầng). Mẫu Voyager co nhỏ lại được.
- **`tools` của OpenAI Agents SDK** — ở mức đơn giản nhất; mỗi tool là một skill nhẹ.

## Đóng gói

`outputs/skill-skill-library.md` tạo ra một thư viện kỹ năng theo hình Voyager, có sẵn đăng ký, truy xuất, đánh version và tinh chỉnh cho mọi runtime đích.

## Bài tập

1. Thêm bộ phát hiện chu trình phụ thuộc vào `compose()`. Điều gì xảy ra khi skill A phụ thuộc B mà B lại phụ thuộc A? Lỗi hay cảnh báo?
2. Cài đặt ghim version cho từng skill. Khi skill cha kết hợp skill con `crafting@1`, một lần tinh chỉnh lên `crafting@2` không được âm thầm nâng cấp skill cha.
3. Thay truy xuất trùng lặp token bằng embedding của sentence-transformers (hoặc một bản BM25 thuần thư viện chuẩn). Đo retrieval@5 trên một thư viện đồ chơi 50 skill.
4. Thêm một agent "curriculum": cho thư viện hiện tại và mô tả lĩnh vực, đề xuất 5 skill còn thiếu. Gọi nó hằng tuần.
5. Đọc tài liệu skill của Claude Agent SDK (Anthropic). Chuyển thư viện đồ chơi sang schema skill của SDK. Khả năng khám phá thay đổi thế nào?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Skill | "Năng lực dùng lại được" | Đoạn mã có tên + mô tả, truy xuất được theo độ tương đồng |
| Skill library | "Bộ nhớ cách làm của agent" | Kho skill bền vững, tìm kiếm được và kết hợp được |
| Curriculum | "Bộ đề xuất tác vụ" | Bộ sinh mục tiêu từ dưới lên, chạy theo khoảng trống năng lực hiện tại |
| Composition | "DAG skill" | Skill gọi skill; sắp xếp topo khi thực thi |
| Iterative refinement | "Vòng tự sửa" | Phản hồi môi trường + lỗi + tự kiểm chứng được đưa vào version kế tiếp |
| Action-space-as-code | "Hành động kiểu lập trình" | Phát ra hàm, không phải lệnh nguyên thuỷ, để có hành vi kéo dài theo thời gian |
| Dedup on write | "Skill sụp lại" | Các mô tả gần trùng sụp về một skill chuẩn duy nhất |

## Đọc thêm

- [Wang và cộng sự, Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) — bài báo gốc về thư viện kỹ năng
- [Tổng quan Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — skill như sản phẩm hoá của năm 2026
- [Anthropic, Xây dựng agent với Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — skill và subagent trong thực tế
- [Madaan và cộng sự, Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — vòng tinh chỉnh nằm dưới Voyager

---

[← Bài trước: Bộ nhớ lai: Vector + Graph + KV](../../09-hybrid-memory-mem0/docs/vi.md) · [Bài sau: Lập kế hoạch với HTN (hierarchical task network - mạng tác vụ phân cấp) và Tìm kiếm Tiến hoá →](../../11-planning-htn-and-evolutionary/docs/vi.md)
