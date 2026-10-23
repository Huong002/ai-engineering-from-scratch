# Spec: dịch 1 bài học (en.md -> vi.md), repo ai-engineering-from-scratch

Mục tiêu: tạo `docs/vi.md` nằm cạnh `docs/en.md`, dịch **trung thành từng dòng** — không thêm, không bớt, không tóm tắt, không "cải thiện".

## Bất biến (bắt buộc)

1. **1 dòng output = 1 dòng nguồn.** Tổng số dòng `vi.md` PHẢI bằng `en.md`. Không gộp/tách dòng, không xoá dòng trống.
2. **Khối code giữ nguyên byte-for-byte**: fence + language tag, inline code, lệnh, đường dẫn, URL, comment trong code, output `text`/`console`, và toàn bộ ```figure```. Không dịch bất cứ gì bên trong code.
3. **LaTeX/math giữ nguyên.**
4. **Mermaid**: giữ nguyên node id, mũi tên, `\n`, `<br/>`, `style`, `classDef`, `subgraph` id. Chỉ dịch **nhãn**: text trong `[...]`, `{...}`, `["..."]`, `"..."`, và nhãn cạnh `|...|`.
   - `subgraph "Tiêu đề có ngoặc kép"` -> dịch phần trong ngoặc kép.
   - `subgraph TenId` (không ngoặc kép) -> GIỮ NGUYÊN, không thêm nhãn mới.
   - `A[Raw Data]` -> `A[Dữ liệu thô]` (giữ id `A`).
5. Không có chữ Việt trong code block / output.
6. Bảng: giữ nguyên số cột, số dòng, thứ tự. Header dịch theo quy tắc Key Terms bên dưới. Cột chứa code/giá trị kỹ thuật giữ nguyên.
7. Link: dịch nhãn, giữ nguyên URL. Tiêu đề paper/tên sách/tên riêng để nguyên tiếng Anh.

## Heading (dùng CHÍNH XÁC các chuỗi này)

| Nguồn | Đích |
|---|---|
| `## Learning Objectives` | `## Mục tiêu học tập` |
| `## The Problem` | `## Vấn đề` |
| `## The Concept` | `## Khái niệm` |
| `## Build It` / `## Building It` | `## Tự tay làm` |
| `## Use It` | `## Áp dụng` |
| `## Ship It` | `## Đóng gói` |
| `## Connections` | `## Liên hệ` |
| `## Exercises` | `## Bài tập` |
| `## Key Terms` | `## Thuật ngữ chính` |
| `## Further Reading` | `## Đọc thêm` |
| `### Step N: ...` | `### Bước N: ...` |

- H1 (`# ...`): dịch tiêu đề bài, giữ tên riêng/thuật ngữ.
- `###` khác: dịch prose, giữ identifier trong heading.
- Các `##` khác không có trong bảng: dịch tự nhiên, giữ cấp heading.
- Số lượng heading phải bằng nguồn.

## Metadata (dòng đầu bài)

Giữ **nhãn tiếng Anh** (`**Type:**`, `**Languages:**`, `**Prerequisites:**`, `**Time:**`).

- `**Type:**` giá trị là enum, **GIỮ NGUYÊN** `Learn` / `Build` (không dịch).
- `**Languages:**` giữ nguyên.
- `**Time:**` -> `~NN phút` (dịch `minutes`).
- `**Prerequisites:**` dịch sang tiếng Việt theo mẫu, KHÔNG được còn `Phase`, `Pha`, `Lessons`:
  - `Phase 2, Lesson 10 (Bias-Variance Tradeoff)` -> `Giai đoạn 2, Bài 10 (Đánh đổi Bias-Variance)`
  - `Phase 2, Lessons 01-07 (classification, Bayes' theorem)` -> `Giai đoạn 2, Bài 01-07 (phân loại, định lý Bayes)`
  - `Phase 1 (Statistics for ML, Linear Algebra), Phase 2 Lessons 1-7` -> `Giai đoạn 1 (Thống kê cho Học máy, Đại số tuyến tính), Giai đoạn 2, Bài 01-07`
  - `Phase 2, Lessons 09 (Model Evaluation) and 13 (ML Pipelines)` -> `Giai đoạn 2, Bài 09 (Đánh giá mô hình) và 13 (Pipeline ML)`
- Không tự thêm tiền tố `Giai đoạn N, ` khi nguồn không có.
- `**Prerequisites:** None` -> `**Prerequisites:** Không có`

## Bảng Key Terms

Đếm số cột của header nguồn:

- Nguồn 3 cột `| Term | What people say | What it actually means |` -> header CHÍNH XÁC:
  `| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |`
- Nguồn 3 cột `| Term | What people say | What it means |` -> header CHÍNH XÁC:
  `| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |`
- Nguồn 2 cột -> header CHÍNH XÁC: `| Thuật ngữ | Ý nghĩa thực sự |`
- Nguồn 2 cột nhưng cột 2 có thêm ngữ cảnh (`| Term | Meaning in 2026-07-28 |`,
  `| Term | Meaning in the current extension |`) -> giữ nguyên ngữ cảnh, dịch:
  `| Thuật ngữ | Ý nghĩa trong 2026-07-28 |`, `| Thuật ngữ | Ý nghĩa trong phần mở rộng hiện tại |`
- Quy tắc chung: giữ ĐÚNG số cột của nguồn; cột 1 đổi `Term` -> `Thuật ngữ`;
  mọi cột còn lại dịch sang tiếng Việt (không để tiếng Anh).

Cột 1 luôn là **thuật ngữ tiếng Anh gốc** (canonical), không dịch. Các cột còn lại dịch.
Xoá dòng `|---|`? KHÔNG — giữ nguyên separator, số cột khớp.

## Văn phong

- Tiếng Việt tự nhiên, xưng `bạn`, chủ động, câu ngắn. Không dịch word-by-word.
- Thuật ngữ: lần xuất hiện ĐẦU TIÊN gloss một lần dạng `thuật ngữ (nghĩa tiếng Việt)`, các lần sau chỉ dùng thuật ngữ.
  Không gloss: tên sản phẩm/thư viện/ngôn ngữ (Python, NumPy, sklearn, PyTorch, XGBoost...), API, SDK, CLI, URL, model, data, pipeline, token, prompt, batch, cache, embedding, overfitting/fine-tuning nếu đã quá quen — nhưng phải gloss đúng 1 lần nếu là thuật ngữ chuyên môn của bài.
- Chỉ gloss thuật ngữ CÓ trong bài; không thêm gloss cho khái niệm nguồn không nhắc tới.
- Số liệu, công thức, version, thông báo lỗi: giữ nguyên literal.
- Không dùng từ Hán-Việt tối nghĩa; không thêm ví dụ mới.

## Điều hướng prev/next

Không tự thêm dòng nav khi dịch. `scripts/translate-vi/add_nav.py` gắn khối nav vào
cuối mỗi `vi.md` sau khi batch dịch xong (idempotent, chạy lại vô hại).

## Quy trình

1. `read` toàn bộ `docs/en.md`.
2. Ghi `docs/vi.md` (1 lần `write`).
3. Chạy verify: `python3 scripts/translate-vi/verify_vi.py <đường_dẫn_thư_mục_bài>` và sửa cho tới khi in `RESULT: PASS`.
4. Chạy audit cắt/thêm: `python3 scripts/translate-vi/audit_batch.py <đường_dẫn_thư_mục_bài>`; mọi issue còn lại phải là false positive đã biết.
5. Báo cáo ngắn (< 8 dòng): số dòng en/vi, số fence, các mục verify, lỗi đã sửa.
