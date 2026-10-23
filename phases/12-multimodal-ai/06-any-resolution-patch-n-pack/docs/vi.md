# Thị giác any-resolution (độ phân giải tuỳ ý): Patch-n'-Pack và NaFlex

> Ảnh thực tế không phải hình vuông 224x224. Một hoá đơn là 9:16, một biểu đồ là 16:9, một ảnh quét y tế có thể là 4096x4096, một ảnh chụp màn hình điện thoại là 9:19.5. Câu trả lời của VLM trước 2024 — resize (đổi kích thước) mọi thứ thành một hình vuông cố định — đã ném đi tín hiệu làm cho OCR, hiểu tài liệu và phân tích cảnh độ phân giải cao hoạt động được. NaViT (Google, 2023) cho thấy có thể pack các patch có độ phân giải thay đổi vào một batch transformer duy nhất với mask block-diagonal. M-RoPE của Qwen2-VL (2024) bỏ hẳn bảng vị trí tuyệt đối. AnyRes của LLaVA-NeXT chia ảnh độ phân giải cao thành một ảnh gốc cộng các ảnh con. Biến thể NaFlex của SigLIP 2 (2025) hiện là encoder mặc định cho các VLM mở muốn một checkpoint duy nhất phục vụ mọi tỉ lệ khung hình (aspect ratio). Bài học này triển khai patch-n'-pack (ghép patch theo lô) từ đầu đến cuối.

**Type:** Build
**Languages:** Python (stdlib, patch packer + block-diagonal mask)
**Prerequisites:** Giai đoạn 12, Bài 01 (Patch ViT), Giai đoạn 12, Bài 05 (LLaVA)
**Time:** ~120 phút

## Mục tiêu học tập

- Pack patch từ một batch ảnh có độ phân giải thay đổi thành một chuỗi duy nhất và dựng mask attention block-diagonal.
- Chọn giữa AnyRes tiling (LLaVA-NeXT), NaFlex (SigLIP 2) và M-RoPE (Qwen2-VL) cho một tác vụ cụ thể.
- Tính ngân sách token (token budget) cho OCR, biểu đồ và ảnh chụp mà không cần resize.
- Gọi tên ba kiểu thất bại của square-resize: chữ bị bóp méo, nội dung bị cắt, token lãng phí vào padding.

## Vấn đề

Transformer cần một chuỗi. Một batch là một chồng các chuỗi có cùng độ dài. Nếu ảnh của bạn là 224x224, bạn luôn nhận 196 token patch, không cần padding, xong việc. Train trên 224, suy luận trên 224, không bao giờ phải nghĩ về độ phân giải nữa.

Thực tế thì không hợp tác như vậy. Tài liệu là khổ dọc (8.5x11 inch, khoảng 2:3). Ảnh chụp màn hình biểu đồ là khổ ngang (16:9). Hoá đơn thì cao và hẹp (1:3). Ảnh y tế thường là 2048x2048 hoặc lớn hơn. Ảnh chụp màn hình di động là 1170x2532 (0.46:1).

Ba lựa chọn trước 2024 và lý do mỗi cách đều thất bại:

1. Resize về một hình vuông cố định (224x224 hoặc 336x336). Việc bóp méo làm biến dạng chữ và khuôn mặt. Việc thu nhỏ phá hủy nhãn biểu đồ và nội dung OCR. Đây là cách làm chuẩn cho tới LLaVA-1.5.
2. Cắt theo một tỉ lệ khung hình cố định. Bạn ném đi phần lớn bức ảnh, và việc chọn vị trí cắt lại là một bài toán thị giác riêng.
3. Padding về cạnh dài nhất. Cách này sửa được méo nhưng lãng phí hơn 50% token vào padding với ảnh khổ dọc. Chi phí attention bậc hai trên toàn bộ số token padding đó.

Câu trả lời của 2024-2025: để transformer ăn patch ở đúng độ phân giải gốc của ảnh, và tìm cách pack một batch không đồng nhất vào một chuỗi duy nhất mà không lãng phí tính toán.

## Khái niệm

### NaViT và patch-n'-pack

NaViT (Dehghani và cộng sự, 2023) là bài báo cho thấy ý tưởng này hoạt động ở quy mô lớn. Ý tưởng thuần cơ học:

1. Với mỗi ảnh trong batch, tính lưới (grid) patch gốc của nó ở một patch size đã chọn (ví dụ 14).
2. Trải phẳng patch của từng ảnh thành chuỗi có độ dài thay đổi riêng.
3. Nối patch của tất cả ảnh thành một chuỗi dài duy nhất cho batch.
4. Dựng mask attention block-diagonal để patch của ảnh A chỉ attend trong phạm vi ảnh A.
5. Mang theo thông tin vị trí cho từng patch (2D RoPE hoặc embedding vị trí phân số).

Một batch gồm ba ảnh 336x336 (576 token), 224x224 (256 token) và 448x336 (768 token) trở thành một chuỗi 1600 token với mask block-diagonal 1600x1600. Không padding. Không lãng phí tính toán. Transformer xử lý được tỉ lệ khung hình tuỳ ý.

NaViT còn giới thiệu fractional patch dropping khi train — bỏ ngẫu nhiên 50% số patch trong batch — vừa giúp regularization vừa tăng tốc train. SigLIP 2 kế thừa ý tưởng này.

### AnyRes (LLaVA-NeXT)

AnyRes của LLaVA-NeXT là phương án thực dụng hơn. Cho một ảnh độ phân giải cao và một encoder cố định (CLIP hoặc SigLIP ở 336), hãy chia ảnh thành ô ảnh (tile):

1. Chọn một bố cục lưới trong tập định sẵn — (1x1), (1x2), (2x1), (1x3), (3x1), (2x2), v.v. — khớp nhất với tỉ lệ khung hình của ảnh.
2. Chia toàn bộ ảnh theo lưới; mỗi ô ảnh trở thành một crop 336x336.
3. Đồng thời tạo một ảnh thu nhỏ: toàn bộ ảnh được resize về 336x336 làm token ngữ cảnh toàn cục.
4. Mã hoá từng ô ảnh qua encoder 336 đóng băng. Nối token của các ô ảnh với token của ảnh thu nhỏ.

Với một ảnh 672x672 ở lưới 2x2 cộng ảnh thu nhỏ: 4 * 576 + 576 = 2880 token thị giác. Đắt nhưng hiệu quả — LLM thấy được cả chi tiết cục bộ lẫn ngữ cảnh toàn cục.

AnyRes là lựa chọn ưu tiên khi encoder của bạn bị đóng băng và chỉ hỗ trợ một độ phân giải. Nó làm số token bùng nổ với ảnh lớn (một ảnh 1344x1344 ở lưới 4x4 là 9216 + 576 ≈ 9800 token, gần như lấp đầy ngữ cảnh 8k của LLM).

### M-RoPE (Qwen2-VL)

Qwen2-VL giới thiệu Multimodal Rotary Position Embedding. Thay vì vị trí phân số của NaViT hay kiểu chia ô cộng ảnh thu nhỏ của AnyRes, mỗi patch mang một vị trí 3D (thời gian, chiều cao, chiều rộng). Các phép quay query/key xử lý được H, W và độ dài thời gian tuỳ ý.

M-RoPE có sẵn độ phân giải động mà không cần train lại. Khi suy luận bạn đưa vào một ảnh HxW bất kỳ, bộ nhúng patch sinh ra H/14 x W/14 token, mỗi token nhận vị trí (t=0, r=hàng, c=cột) của nó, RoPE quay attention với đúng tần số, xong. Qwen2.5-VL và Qwen3-VL tiếp tục hướng này. V2PE của InternVL3 là cùng ý tưởng với cách mã hoá thay đổi theo từng modality.

Khác với AnyRes, M-RoPE cho O(H x W / P^2) token ở độ phân giải gốc — không có chi phí nhân lên do chia ô. Khác với NaViT, nó vẫn cần một ảnh mỗi lượt forward. Batch hoá qua nhiều độ phân giải vẫn cần patch-n'-pack chồng lên trên.

### NaFlex (SigLIP 2)

NaFlex là chế độ native-flex của checkpoint SigLIP 2. Một mô hình duy nhất phục vụ nhiều độ dài chuỗi (256, 729, 1024 token) khi suy luận. Bên trong, nó dùng patch-n'-pack kiểu NaViT khi train và vị trí phân số tuyệt đối cho từng patch. Điểm bán được: một checkpoint, chọn ngân sách token khi suy luận tuỳ theo tác vụ.

Với tác vụ ngữ nghĩa (phân loại, truy hồi), 256 token. Với OCR hoặc hiểu biểu đồ, 1024 token. Không cần train lại.

### Ma trận mask khi pack

Mask block-diagonal là chỗ hầu hết các cài đặt vấp ngã. Với một chuỗi đã pack có độ dài `N_total` phủ các ảnh `i=0..B-1` với độ dài `n_i`, mask `M` kích thước `(N_total, N_total)` bằng 1 nếu cả hai chỉ số cùng nằm trong block của một ảnh, ngược lại bằng 0. Bạn có thể dựng nó từ một danh sách độ dài tích luỹ:

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

Đây là một dòng trong PyTorch với `torch.block_diag` hoặc một lệnh gather tường minh. Đường dẫn độ dài thay đổi của FlashAttention (`cu_seqlens`) bỏ qua mask hoàn toàn và attend trong từng chuỗi bằng chính tensor độ dài tích luỹ — nhanh hơn khoảng 10 lần so với mask dày đặc với các batch thường gặp.

### Ngân sách token

Chọn chiến lược theo tác vụ:

- OCR / tài liệu: 1024-4096 token. SigLIP 2 NaFlex ở 1024, hoặc AnyRes 3x3 cộng ảnh thu nhỏ.
- Biểu đồ và UI: 729-1024 token ở độ phân giải gốc 384-448. Qwen2.5-VL độ phân giải động với giới hạn số pixel tối đa.
- Ảnh tự nhiên: 256-576 token là đủ. LLM phía sau thấy đủ thông tin. Trả tiền cho token ở nơi mật độ nội dung cao.
- Video: 64-128 token mỗi khung sau khi pooling không gian, 2-8 FPS. Bài 12.17 đề cập nội dung này.

Quy tắc sản xuất năm 2026: chọn một giới hạn số pixel tối đa cho từng tác vụ, mã hoá ở tỉ lệ khung hình gốc cho tới giới hạn đó, pack batch, và bỏ padding. Qwen2.5-VL phơi ra `min_pixels` và `max_pixels` đúng cho nút vặn này.

```figure
mm-patch-n-pack
```

## Áp dụng

`code/main.py` cài đặt patch-n'-pack cho một batch ảnh không đồng nhất với toạ độ pixel nguyên. Nó:

- Nhận một danh sách kích thước ảnh (H, W).
- Tính độ dài chuỗi patch của từng ảnh ở patch size 14.
- Pack chúng vào một chuỗi có tổng độ dài `sum(n_i)`.
- Dựng mask attention block-diagonal (dạng dày đặc, để dễ đọc).
- So sánh chi phí khi pack với square-resize và AnyRes tiling.
- In một bảng ngân sách token cho một batch hỗn hợp (hoá đơn, biểu đồ, ảnh chụp màn hình, ảnh chụp).

Chạy thử. Những con số hiện ra chính là lý do mọi VLM mở năm 2026 đều dùng patch-n'-pack.

## Đóng gói

Bài học này tạo ra `outputs/skill-resolution-budget-planner.md`. Cho một khối lượng công việc với tỉ lệ khung hình hỗn hợp (OCR, biểu đồ, ảnh chụp, khung video) và một tổng ngân sách token, nó chọn chiến lược phù hợp (NaFlex, AnyRes, M-RoPE hoặc hình vuông cố định) và xuất ra cấu hình cho từng request. Dùng skill này khi bạn đang chọn kích cỡ VLM cho một sản phẩm — nó ngăn chặn kiểu bùng nổ token 10 lần âm thầm làm sập ngân sách độ trễ.

## Bài tập

1. Một hoá đơn là 600x1500 (1:2.5). Ở patch size 14, có bao nhiêu token ở độ phân giải gốc? Sau khi resize về hình vuông 336 thì còn bao nhiêu? Cách nào làm mất độ chính xác OCR nhiều hơn trên thực tế?

2. Dựng mask block-diagonal cho một batch bốn ảnh có độ dài 256, 576, 729, 1024. Kiểm tra rằng ma trận attention là 2585x2585 và có đúng `256^2 + 576^2 + 729^2 + 1024^2` phần tử khác không.

3. Với một ảnh 1792x896 ở patch 14, hãy so sánh: (a) resize về hình vuông 336 rồi mã hoá, (b) AnyRes 2x1 cộng ảnh thu nhỏ, (c) M-RoPE ở độ phân giải gốc. Cách nào dùng ít token nhất? Cách nào giữ được nhiều chi tiết nhất?

4. Cài đặt fractional patch dropping: cho một chuỗi đã pack, bỏ 50% token phân bố đều ngẫu nhiên, và cập nhật mask block-diagonal tương ứng. Đo mức thay đổi độ thưa của mask.

5. Đọc Mục 3.2 của bài báo Qwen2-VL (arXiv:2409.12191). Mô tả trong hai câu `min_pixels` và `max_pixels` điều khiển gì và vì sao cả hai chặn đều quan trọng.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| Patch-n'-pack | "Pack kiểu NaViT" | Nối các chuỗi patch có độ dài thay đổi từ những ảnh khác nhau vào một chiều batch duy nhất |
| Block-diagonal mask | "Mask khi pack" | Mask attention giới hạn patch của mỗi ảnh chỉ attend với chính chúng, không với ảnh khác trong cùng gói |
| AnyRes | "Chia ô của LLaVA-NeXT" | Chia một ảnh độ phân giải cao thành lưới các ô ảnh kích thước cố định cộng một ảnh thu nhỏ toàn cục; mã hoá từng ô ảnh bằng encoder cố định |
| NaFlex | "Native-flex của SigLIP 2" | Một checkpoint SigLIP 2 duy nhất phục vụ ngân sách 256/729/1024 token khi suy luận mà không cần train lại |
| M-RoPE | "RoPE đa phương thức" | Mã hoá vị trí quay 3D (thời gian, hàng, cột) xử lý H, W, T tuỳ ý mà không cần bảng vị trí |
| cu_seqlens | "Pack với FlashAttention" | Tensor độ dài tích luỹ mà đường dẫn varlen của FlashAttention dùng thay cho mask block-diagonal dày đặc |
| min_pixels / max_pixels | "Giới hạn độ phân giải" | Nút vặn theo từng request của Qwen2.5-VL, chặn số token với đầu vào quá nhỏ hoặc quá lớn |
| Visual token budget | "Bao nhiêu token mỗi ảnh" | Ước lượng số token patch phát ra cho mỗi ảnh; quyết định ngân sách prompt và chi phí attention của LLM |

## Đọc thêm

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)

---

[← Bài trước: LLaVA và tinh chỉnh theo chỉ dẫn thị giác](../../05-llava-visual-instruction-tuning/docs/vi.md) · [Bài sau: Công thức cho VLM open-weight (trọng số mở): điều gì thực sự quan trọng →](../../07-open-weight-vlm-recipes/docs/vi.md)
