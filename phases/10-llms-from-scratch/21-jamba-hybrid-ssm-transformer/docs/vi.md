# Jamba — Kiến trúc lai SSM-Transformer

> State space model (mô hình không gian trạng thái) và Transformer muốn những thứ khác nhau. Transformer mua chất lượng bằng attention với chi phí bậc hai. SSM mua suy luận linear time (thời gian tuyến tính) và bộ nhớ hằng số nhờ recurrence (hồi quy) nhưng chất lượng còn thua kém. Jamba của AI21 (tháng 3 năm 2024) và Jamba 1.5 (tháng 8 năm 2024) đặt cả hai vào cùng một mô hình: 1 lớp Transformer cho mỗi 7 lớp Mamba, mixture of experts (MoE, hỗn hợp chuyên gia) trên mỗi block xen kẽ, và cửa sổ ngữ cảnh 256k vừa trên một GPU 80GB. Mamba-3 (ICLR 2026) siết chặt phía SSM bằng không gian trạng thái có giá trị phức và các phép chiếu MIMO. Bài này đọc cả hai kiến trúc từ đầu đến cuối và giải thích vì sao công thức hybrid architecture (kiến trúc lai) đã trụ vững qua ba năm mở rộng quy mô, trong khi các nỗ lực ngữ cảnh dài thuần SSM và thuần Transformer thì không.

**Type:** Learn
**Languages:** Python (stdlib, layer-mix calculator)
**Prerequisites:** Giai đoạn 10, Bài 14 (kiến trúc mô hình mở), Giai đoạn 10, Bài 17 (native sparse attention)
**Time:** ~60 phút

## Mục tiêu học tập

- Giải thích ba primitive trong một block Jamba — lớp Transformer, lớp Mamba, MoE — và công thức xen kẽ 1:7:even.
- Nêu ở mức khái quát recurrence của một SSM trông như thế nào và vì sao nó cho phép suy luận với bộ nhớ hằng số.
- Tính dấu chân KV cache của một mô hình Jamba ở ngữ cảnh 256k và so sánh với mức mà một mô hình thuần Transformer sẽ cần.
- Kể tên ba đổi mới của Mamba-3 (rời rạc hóa mũ-hình thang, cập nhật trạng thái phức, MIMO) và vấn đề mà mỗi đổi mới nhắm tới.

## Vấn đề

Attention là bậc hai theo độ dài chuỗi. State space model thì tuyến tính. Khác biệt đó cộng dồn: ở 256k token, bản đồ attention của một Transformer là 65B phần tử mỗi head; trạng thái hồi quy của một SSM có kích thước cố định bất kể độ dài chuỗi.

Các mô hình thuần SSM (Mamba, Mamba-2) sánh ngang perplexity của Transformer ở quy mô nhỏ nhưng thua trên các tác vụ theo dõi trạng thái và thất bại ở một số loại truy xuất trong ngữ cảnh. Trực giác: SSM nén lịch sử vào một trạng thái cố định, và khi lịch sử dài thì thông tin bị rò rỉ. Attention nhớ mọi thứ chính xác nhưng trả chi phí bậc hai.

Cách sửa hiển nhiên: dùng cả hai. Đặt các lớp Transformer ở nơi cần nhớ chính xác. Dùng lớp SSM ở những chỗ khác. Tinh chỉnh tỉ lệ. Jamba là mô hình cấp sản xuất đầu tiên tung công thức kiến trúc lai này ra ở quy mô lớn (52B tổng, 12B hoạt động, ngữ cảnh 256k, một GPU 80GB). Jamba 1.5 mở rộng họ này lên 398B tổng / 94B hoạt động. Mamba-3 (ICLR 2026) là baseline thuần SSM tốt nhất hiện tại để xây lại các kiến trúc lai quanh nó.

Bài này đọc cả ba bài báo và dựng ra mô hình tư duy để "chọn đúng tỉ lệ".

## Khái niệm

### Một SSM trong một trang

Một state space model xử lý một chuỗi `x_1, ..., x_N` qua một trạng thái `h` có kích thước cố định:

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

Ở mỗi bước, trạng thái tiến hóa qua một động lực học tuyến tính `A`, nhận đầu vào `B x_t`, và phát ra đầu ra `C h_t`. `A, B, C` có thể học được. Để ý tính chất then chốt: tính `y_t` chỉ cần `h_{t-1}` và `x_t`, không cần bất kỳ `x` nào trước đó. Bộ nhớ là hằng số. Suy luận là O(1) mỗi token.

Mẹo để có chất lượng mô hình hóa nằm ở cấu trúc của `A`. S4 (Gu 2021) dùng một ma trận có cấu trúc cao để có thể tính hiệu quả như một phép tích chập dài trong lúc huấn luyện. Mamba (Gu, Dao 2023) thay `A, B, C` cố định bằng loại phụ thuộc dữ liệu (phần "selective"). Mamba-2 (2024) đơn giản hóa cấu trúc hơn nữa. Mamba-3 (2026) thêm lại độ phức tạp ở những chỗ cụ thể.

Tính chất then chốt: với một decoder LLM, một lớp SSM là thứ thay thế trực tiếp cho attention layer (lớp attention), với trạng thái mỗi lớp có kích thước cố định thay cho KV cache ngày càng lớn.

### Block Jamba

Một block Jamba xen kẽ các lớp theo hai con số:

- `l`: tỉ lệ attention trên Mamba. Jamba dùng `l = 8`, nghĩa là 1 lớp Transformer cho mỗi 7 lớp Mamba (7 Mamba + 1 Attention = 8 lớp mỗi nhóm).
- `e`: tần suất MoE. Jamba dùng `e = 2`, nghĩa là cứ cách một lớp thì áp dụng MoE.

Trình tự lớp trong một block:

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

Mỗi block Jamba gồm 8 lớp. Ở độ sâu 4 block (tổng 32 lớp), bạn có 28 lớp Mamba và 4 lớp Attention. 16 trong số đó dùng MoE.

### Vì sao tỉ lệ 1:7

AI21 chạy các ablation: tỉ lệ attention trên Mamba nào cho perplexity-trên-tham-số tốt nhất VÀ khả năng nhớ trong ngữ cảnh tốt nhất trên các bài đánh giá ngữ cảnh dài của họ?

- Quá nhiều attention (1:1): chất lượng tăng nhưng bộ nhớ và tốc độ giảm.
- Quá ít attention (1:15): bộ nhớ tuyệt vời nhưng truy xuất trong ngữ cảnh thất bại.
- Điểm ngọt: 1:7 hoặc 1:8.

Trực giác: các lớp Transformer đảm nhận việc nhớ chính xác và theo dõi trạng thái. Các lớp Mamba đảm nhận phần xử lý số lượng lớn nhưng rẻ.

### Mã hóa vị trí

Bản thân các lớp Mamba đã biết vị trí (nhờ recurrence). Các lớp attention trong những kiến trúc lai dựa trên Mamba đời đầu không dùng RoPE — các lớp SSM đảm nhiệm thông tin vị trí. Jamba 1.5 thêm RoPE cho các lớp attention để tổng quát hóa với ngữ cảnh dài hơn, một bước tinh chỉnh hậu nghiệm dựa trên đánh giá ngữ cảnh dài thực nghiệm.

### Ngân sách bộ nhớ

Với hình dạng Jamba-1 (32 lớp: 28 Mamba + 4 Attention, hidden 4096, 32 head attention):

- KV cache (chỉ các lớp attention): `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB` ở 256k BF16. Chỉ 4 lớp attention đóng góp.
- Trạng thái SSM: `28 * hidden * state_size` mỗi tiền tố token, nhưng đây là kích thước cố định theo từng lớp, không tăng theo độ dài chuỗi. Trạng thái Mamba điển hình là 16 mỗi feature, hidden 4096: `28 * 4096 * 16 * 2 = 3.7 MB` tổng cộng.

So với một Transformer thuần ở 32 lớp, cùng hidden, MHA đầy đủ ở 32 head: `2 * 32 * 32 * 128 * 256k * 2 = 128 GB` ở 256k BF16. Giảm 8 lần KV cache. Ngay cả so với baseline GQA(8) mà hầu hết mô hình năm 2024 dùng (`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`), kiến trúc lai 1:7 của Jamba ở 16 GB vẫn nhỏ hơn 2 lần.

Đó là ý AI21 khi nói "ngữ cảnh 256k trên một GPU 80GB". KV cache của một Transformer thuần MHA đầy đủ sẽ không vừa; ngay cả baseline GQA cũng không còn chỗ cho trọng số và activation; còn Jamba thì vừa.

### Mamba-3: baseline thuần SSM năm 2026

Mamba-3 (ICLR 2026, arXiv:2603.15569) giới thiệu ba đổi mới ở phía thuần SSM:

1. **Rời rạc hóa mũ-hình thang.** Thay rời rạc hóa bằng phương pháp Euler trong Mamba-2 bằng một recurrence diễn đạt được nhiều hơn. Phép toán kiểu tích chập được áp lên đầu vào-trạng thái bên trong recurrence lõi, thay vì như một tích chập ngoài trên `x_t`.

2. **Cập nhật trạng thái phức.** Các Mamba trước đó giảm ma trận trạng thái từ phức (S4) xuống đường chéo thực (Mamba) rồi xuống ma trận đơn vị có tỉ lệ (Mamba-2). Mamba-3 thêm lại giá trị phức — tương đương một rotary embedding phụ thuộc dữ liệu trên trạng thái. Điều này khôi phục năng lực theo dõi trạng thái mà các bước đơn giản hóa giá trị thực trước đó đã làm mất.

3. **Phép chiếu đa đầu vào đa đầu ra (MIMO).** Thay vì phép chiếu vô hướng theo từng feature, dùng phép chiếu có giá trị ma trận. Cải thiện năng lực mô hình hóa và mức tận dụng phần cứng khi suy luận mà không tăng độ trễ decode.

Ở 1.5B tham số, Mamba-3 cải thiện độ chính xác trung bình downstream 0.6 điểm so với Gated DeltaNet; biến thể MIMO thêm 1.2 điểm nữa, tổng cộng 1.8 điểm. Ở cùng kích thước trạng thái, Mamba-3 sánh ngang Mamba-2 với một nửa trạng thái.

Mamba-3 chưa được tung ra trong một kiến trúc lai cấp sản xuất ở quy mô lớn — nhưng nó là ứng viên hiển nhiên cho phía SSM của mô hình hạng Jamba tiếp theo.

### Khi nào nên dùng kiến trúc lai

Kiến trúc lai thắng khi:

- Ngữ cảnh đủ dài để KV cache của Transformer thuần trở nên đau đớn (64k+).
- Tác vụ trộn cấu trúc tầm ngắn (tốt cho SSM) với nhu cầu nhớ tầm xa (cần Transformer).
- Bạn muốn triển khai trong ngân sách bộ nhớ một GPU, nơi chỉ riêng KV cache của Transformer đã không vừa.

Kiến trúc lai thua khi:

- Ngữ cảnh ngắn (dưới 16k). Chi phí phụ của SSM bị lãng phí; Transformer thuần là đủ.
- Tác vụ cần attention mọi-nơi-tới-mọi-nơi (suy luận sâu, tham chiếu chéo nhiều tài liệu). Độ thưa của các lớp attention trong kiến trúc lai gây hại.
- Bạn đang mở rộng lên các mô hình biên hàng nghìn tỉ tham số. Transformer thuần + MLA + MoE (kiểu DeepSeek-V3) hiện đang thắng trong cuộc đua năng lực.

### Bối cảnh cạnh tranh

| Model | Họ | Quy mô | Tuyên bố riêng |
|-------|--------|------|-------------|
| Mamba-2 | SSM thuần | 3B | thời gian tuyến tính, bộ nhớ hằng số |
| Jamba | lai | 52B/12B | 256k trên 80GB |
| Jamba 1.5 Large | lai | 398B/94B | ngữ cảnh dài cấp doanh nghiệp |
| Mamba-3 | SSM thuần | 1.5B (bài báo) | khôi phục khả năng theo dõi trạng thái |
| DeepSeek-V3 | Transformer thuần + MoE | 671B/37B | năng lực biên |

Bối cảnh năm 2026: MoE Transformer thuần thống trị vùng biên, nhưng kiến trúc lai sở hữu ngách ngữ cảnh 256k trở lên. Những thắng lợi về theo dõi trạng thái của Mamba-3 có thể đẩy tỉ lệ lai xuống thấp hơn (nhiều SSM hơn, ít attention hơn) ở thế hệ tiếp theo.

```figure
swiglu-ffn
```

## Áp dụng

`code/main.py` là một máy tính bộ nhớ cho kiến trúc lai. Cho một tỉ lệ SSM-Transformer và một cấu hình hidden-size / số lớp, nó tính:

- KV cache ở ngữ cảnh mục tiêu.
- Bộ nhớ trạng thái SSM.
- Tổng bộ nhớ ở ngữ cảnh N cho một dải hình dạng mô hình.

Máy tính hỗ trợ:

- Baseline Transformer thuần (KV cache tăng theo N).
- Kiến trúc lai 1:7 kiểu Jamba.
- SSM thuần (không có KV cache).

Các con số lấy trực tiếp từ bài báo Jamba-1 và Jamba-1.5 cho những hình dạng đã công bố và ngoại suy cho các biến thể giả định.

Cân nhắc tích hợp cho một lần triển khai thật:

- Hầu hết server suy luận cấp sản xuất (vLLM, SGLang) hỗ trợ Jamba và Mamba. Hãy kiểm tra phiên bản cụ thể.
- Ở ngữ cảnh 256k, lợi thế bộ nhớ của Jamba thể hiện ở throughput (thông lượng) khi xử lý nhiều yêu cầu đồng thời. Trên cùng một lượng VRAM, bạn chứa được nhiều chuỗi Jamba hơn chuỗi Transformer.
- Mamba-3 như một mô hình độc lập chưa được tung ra trong sản xuất — mới là bản xem trước nghiên cứu ở 1.5B.

## Đóng gói

Bài này tạo ra `outputs/skill-hybrid-picker.md`. Cho một đặc tả khối lượng công việc (hồ sơ độ dài ngữ cảnh, hỗn hợp tác vụ, ngân sách bộ nhớ), nó đề xuất giữa Transformer thuần, kiến trúc lai kiểu Jamba, và SSM thuần, kèm lập luận rõ ràng về đánh đổi bộ nhớ và chất lượng.

## Bài tập

1. Chạy `code/main.py` để tính KV cache ở ngữ cảnh 256k cho một Transformer thuần 32 lớp (hidden 4096, 32 head) và cho một kiến trúc lai Jamba-1 cùng hình dạng. Kiểm chứng mức giảm bộ nhớ ~8 lần mà bài báo AI21 tuyên bố.

2. Sửa máy tính để mô hình hóa một kiến trúc lai 1:3 (4 Mamba : 1 Attention) và một kiến trúc lai 1:15 (14 Mamba : 1 Attention). Vẽ KV cache theo tỉ lệ. Ở tỉ lệ nào thì KV cache bằng bộ nhớ trạng thái SSM?

3. Đọc Mục 3 của bài báo Jamba (arXiv:2403.19887). Giải thích vì sao AI21 dùng Mamba-1 thay vì Mamba-2 dù Mamba-2 nhanh hơn. Gợi ý: phần ablation kiến trúc lai có ghi lại điều này.

4. Tính chi phí tham số của việc áp MoE cách một lớp trong Jamba 1.5 Large (398B tổng, 94B hoạt động). So tỉ lệ hoạt động với DeepSeek-V3 (37B/671B) và giải thích vì sao kiến trúc của Jamba đẩy tỉ lệ hoạt động cao hơn.

5. Đọc Mục 3 của bài báo Mamba-3 (arXiv:2603.15569). Giải thích trong ba câu vì sao cập nhật trạng thái có giá trị phức tương đương một rotary embedding phụ thuộc dữ liệu. Gắn câu trả lời với phần dẫn xuất RoPE ở Giai đoạn 7 · Bài 04.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| State space model (SSM) | "Hồi quy với một trạng thái cố định" | Một lớp với recurrence học được `h_t = A h_{t-1} + B x_t`; bộ nhớ hằng số mỗi token |
| Selective SSM | "Mẹo của Mamba" | Tham số A, B, C phụ thuộc dữ liệu, cho mô hình khả năng chọn lọc kiểu cổng với linear time |
| Attention-to-Mamba ratio | "Bao nhiêu lớp attention" | Trong Jamba, `l = 8` nghĩa là 1 lớp attention cho mỗi 7 lớp Mamba |
| Jamba block | "Nhóm 8 lớp" | Một attention + bảy Mamba + MoE ở các vị trí xen kẽ |
| SSM state | "Bộ đệm ẩn" | Trạng thái mỗi lớp có kích thước cố định, thay thế KV cache cho các lớp Mamba |
| 256k context | "Con số hào nhoáng của Jamba" | Độ dài chuỗi mà Jamba-1 vừa trên một GPU 80GB; Transformer thuần không thể ở kích thước đó |
| Mamba-3 | "SSM thuần 2026" | Kiến trúc thuần SSM tốt nhất hiện tại với trạng thái phức + MIMO; baseline để các kiến trúc lai xây quanh |
| MIMO | "Multi-input multi-output" | Đổi mới của Mamba-3 dùng phép chiếu có giá trị ma trận thay cho vô hướng theo từng feature |
| Exponential-trapezoidal discretization | "Recurrence của Mamba-3" | Recurrence diễn đạt được nhiều hơn, bao hàm rời rạc hóa bằng phương pháp Euler của Mamba-2 |
| Hybrid architecture | "Trộn attention và SSM" | Bất kỳ mô hình nào xen kẽ lớp Transformer và SSM; Jamba là nguyên mẫu cấp sản xuất |

## Đọc thêm

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — bài báo Jamba gốc, các ablation tỉ lệ, tuyên bố ngữ cảnh 256k
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — họ mô hình mở rộng, các bản công bố 398B/94B và 12B/52B
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — bài báo selective SSM mà Jamba dựa trên
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — hậu duệ structured state space đơn giản hóa
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — trạng thái có giá trị phức, MIMO, biên thuần SSM năm 2026
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — bài báo S4, điểm khởi đầu của phả hệ SSM cho LLM

---

[← Bài trước: Điểm qua kiến trúc DeepSeek-V3](../../20-deepseek-v3-walkthrough/docs/vi.md) · [Bài sau: Suy luận bất đồng bộ và Hogwild! →](../../22-async-hogwild-inference/docs/vi.md)
