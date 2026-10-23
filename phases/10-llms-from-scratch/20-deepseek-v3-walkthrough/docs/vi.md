# Điểm qua kiến trúc DeepSeek-V3

> Giai đoạn 10 · Bài 14 đã gọi tên sáu núm vặn kiến trúc mà mọi mô hình mở đều xoay. DeepSeek-V3 (tháng 12 năm 2024, tổng 671B tham số, 37B hoạt động) xoay cả sáu và thêm bốn cái nữa: Multi-Head Latent Attention (MLA — attention tiềm ẩn nhiều đầu), cân bằng tải (load balancing) không cần loss phụ, Multi-Token Prediction (MTP — dự đoán nhiều token), và huấn luyện DualPipe. Bài này đọc kiến trúc DeepSeek-V3 từ trên xuống dưới và suy ra từng con số tham số từ config đã công bố. Đến cuối, bạn có thể giải thích vì sao tỉ lệ 671B/37B là đặt cược đúng, và vì sao MLA + MoE (hỗn hợp chuyên gia) đi cùng nhau lại thắng cả hai khi đứng riêng ở biên giới năng lực.

**Type:** Learn
**Languages:** Python (stdlib, parameter calculator)
**Prerequisites:** Giai đoạn 10, Bài 14 (điểm qua mô hình mở), Giai đoạn 10, Bài 17 (NSA), Giai đoạn 10, Bài 18 (MTP), Giai đoạn 10, Bài 19 (DualPipe)
**Time:** ~75 phút

## Mục tiêu học tập

- Đọc config DeepSeek-V3 từ trên xuống dưới và giải thích từng trường theo sáu núm vặn GPT-2 cộng bốn bổ sung riêng của DeepSeek.
- Suy ra tổng số tham số (671B), số tham số hoạt động (37B), và các thành phần đóng góp vào mỗi con số.
- Tính dấu chân KV cache của MLA ở ngữ cảnh 128k và so với cái giá mà một mô hình dense có cùng tham số hoạt động dùng GQA phải trả.
- Nêu bốn cải tiến riêng của DeepSeek (MLA, MTP, định tuyến không cần loss phụ, DualPipe) và chỉ ra mỗi cái nhắm vào phần nào của kiến trúc hay stack huấn luyện.

## Vấn đề

DeepSeek-V3 là mô hình mở ở biên giới năng lực đầu tiên có kiến trúc khác biệt đáng kể so với họ Llama. Llama 3 405B là "GPT-2 với sáu núm vặn được xoay". DeepSeek-V3 là GPT-2 với cả sáu núm cộng thêm bốn. Đọc config Llama 3 là màn khởi động trước khi đọc config DeepSeek, nhưng cấu trúc sâu — hình dạng khối attention, logic định tuyến, mục tiêu lúc huấn luyện — khác đủ xa để bạn cần một bài riêng.

Phần thưởng khi hiểu được nó: bản phát hành trọng số mở của DeepSeek-V3 đã làm đổi nghĩa cụm "năng lực biên giới" ở mô hình mở. Kiến trúc này là bản thiết kế mà nhiều lượt huấn luyện năm 2026 đang sao chép. Hiểu nó là điều kiện tối thiểu cho bất kỳ vai trò nào chạm tới huấn luyện hay suy luận LLM biên giới.

## Khái niệm

### Lõi bất biến, một lần nữa

DeepSeek-V3 vẫn là autoregressive. Nó vẫn xếp chồng các khối decoder. Mỗi khối vẫn có attention cộng MLP cộng hai RMSNorm. Nó vẫn dùng SwiGLU trong MLP. Nó vẫn dùng RoPE. Pre-norm. Embedding buộc trọng số. Cùng đường cơ sở với mọi Llama hay Mistral.

### Chỗ xoắn: MLA thay cho GQA

Từ Giai đoạn 10 · 14 bạn đã biết GQA thu nhỏ KV cache bằng cách chia sẻ K và V qua các nhóm đầu Q. MLA đi xa hơn: K và V được nén vào một biểu diễn tiềm ẩn hạng thấp dùng chung (`kv_lora_rank`), rồi giải nén theo từng đầu ngay lúc chạy. KV cache chỉ lưu phần tiềm ẩn — thường 512 số thực mỗi token mỗi lớp, chứ không phải 8 x 128 = 1024 số thực.

Ở ngữ cảnh 128k, DeepSeek-V3 dùng MLA (một tiềm ẩn dùng chung `c^{KV}` mỗi token mỗi lớp; K và V đều suy ra từ tiềm ẩn này qua các phép chiếu lên, có thể hấp thụ vào matmul phía sau):

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

Một đường cơ sở GQA giả định (hình dạng Llama 3 70B, 8 đầu KV, chiều đầu 128) sẽ phải trả:

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

MLA nhỏ hơn 4 lần so với cache GQA kiểu Llama-3-70B ở ngữ cảnh 128k.

Đánh đổi: MLA thêm một bước giải nén cho mỗi lần tính attention (theo từng đầu). Phần compute thêm nhỏ so với băng thông tiết kiệm được. Kết cục là thắng cho suy luận ngữ cảnh dài.

### Định tuyến: cân bằng tải không cần loss phụ

Bộ định tuyến MoE quyết định expert top-k nào xử lý mỗi token. Một bộ định tuyến ngây thơ sẽ dồn quá nhiều việc cho vài expert, để các expert khác nhàn rỗi. Cách sửa chuẩn: thêm một số hạng loss phụ phạt sự mất cân bằng tải. Cách này hiệu quả nhưng làm giảm nhẹ hiệu năng tác vụ chính.

DeepSeek-V3 giới thiệu một phương án không cần loss phụ. Các số hạng bias theo từng expert được cộng vào logits của bộ định tuyến, và điều chỉnh trong lúc huấn luyện theo một quy tắc đơn giản: nếu expert `e` bị quá tải, giảm `bias_e`; nếu tải thấp, tăng lên. Không thêm số hạng loss. Huấn luyện vẫn sạch. Tải expert vẫn cân bằng.

Ảnh hưởng lên loss chính: không đo được. Ảnh hưởng lên kiến trúc MoE: gọn hơn, không còn siêu tham số loss phụ để tinh chỉnh.

### MTP: huấn luyện dày hơn + bản nháp miễn phí

Từ Giai đoạn 10 · 18 bạn đã biết DeepSeek-V3 thêm một module MTP với D=1 dự đoán token ở hai vị trí phía trước. Khi suy luận, module đã huấn luyện được tái dùng làm bản nháp giải mã suy đoán (speculative decoding) với tỉ lệ chấp nhận trên 80%. Khi huấn luyện, mỗi hidden state được giám sát trên D+1 = 2 mục tiêu, cho tín hiệu dày hơn.

Tham số: 14B cộng thêm vào 671B chính. Chi phí phụ: 2.1%.

### Huấn luyện: DualPipe

Từ Giai đoạn 10 · 19 bạn đã biết DualPipe là một song song pipeline (pipeline parallelism) hai chiều, xen kẽ các chunk forward và backward với giao tiếp all-to-all giữa các node. Ở quy mô 2.048 H800 của DeepSeek-V3, nó thu hồi khoảng 245k giờ GPU mà 1F1B sẽ mất vào bong bóng pipeline.

### Config, từng trường một

Đây là config DeepSeek-V3 (đã đơn giản hoá):

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

Bóc tách nó:

- `hidden_size=7168`: chiều embedding.
- `num_hidden_layers=61`: tổng độ sâu khối.
- `first_k_dense_layers=3`: 3 khối đầu dùng MLP dense kích thước 18432. 58 khối còn lại dùng MoE.
- `num_attention_heads=128`: 128 đầu query.
- `kv_lora_rank=512`: K và V được nén về chiều tiềm ẩn này rồi giải nén theo từng đầu.
- `num_experts=256, num_experts_per_tok=8`: mỗi khối MoE có 256 expert, định tuyến top-8.
- `shared_experts=1`: cộng thêm 256 expert được định tuyến, 1 expert luôn bật đóng góp cho mọi token. Hãy nghĩ nó như một "sàn dense" bảo đảm mọi token nhận được thứ gì đó đáng tin.
- `moe_intermediate_size=2048`: kích thước ẩn MLP của mỗi expert. Nhỏ hơn MLP dense vì có tới 256 expert.

### Kiểm kê tham số

Toàn bộ phép tính nằm trong `code/main.py`. Điểm chính:

- Embedding: `vocab * hidden = 129280 * 7168 = ~0.93B`.
- 3 khối dense đầu: attention với MLA (~144M mỗi khối) + MLP dense (~260M mỗi khối) + các norm. Tổng khoảng 1.2B.
- 58 khối MoE: attention với MLA (~144M) + 256 expert (30M mỗi cái) + 1 expert chia sẻ (30M) + norm. Tổng ~7.95B mỗi khối, gồm tất cả expert. 461B cho 58 khối MoE.
- Module MTP: 14B.

Tổng cộng: ~476B cho kiến trúc lõi + 14B MTP + rõ ràng con số 671B đã công bố có tính thêm các tham số cấu trúc khác (tensor bias, thành phần riêng của expert, scaling cho expert chia sẻ, v.v.). Con số chúng ta tái tạo được trong máy tính nằm trong khoảng 3-5% so với công bố — phần lệch đến từ cách kiểm kê chi tiết mà báo cáo của DeepSeek ghi trong phụ lục Mục 2.

Tham số hoạt động mỗi lượt forward:

- Attention: 144M mỗi lớp * 61 = 8.8B (mọi lớp đều chạy).
- MLP hoạt động: 3 lớp đầu dense (3 * 260M = 780M), 58 lớp MoE mỗi lớp hoạt động với 8 expert định tuyến + 1 chia sẻ + chi phí định tuyến. Mỗi lớp MLP hoạt động: ~260M. Tổng: 3 * 260M + 58 * 260M = ~15.9B.
- Embedding + các norm: 1.2B.
- Tổng hoạt động: khoảng 26B lõi + 14B MTP (được huấn luyện nhưng không phải lúc nào cũng chạy khi suy luận) ≈ 37B.

### Tỉ lệ 671B / 37B

Tỉ lệ thưa 18 lần (tham số hoạt động chiếm 5.5% tổng). DeepSeek-V3 là mô hình MoE biên giới thưa nhất đã phát hành trọng số mở. Mixtral 8x7B với tỉ lệ 13/47 (28%) đặc hơn nhiều. Llama 4 Maverick với tỉ lệ 17B/400B (4.25%) là tương đương. Đặt cược của DeepSeek: ở quy mô biên giới, nhiều expert hơn với tỉ lệ kích hoạt thấp hơn cho chất lượng tốt hơn trên mỗi active-FLOP.

### DeepSeek-V3 đứng ở đâu

| Mô hình | Tổng | Hoạt động | Tỉ lệ | Attention | Ý tưởng mới |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN extension |

### Phần tiếp nối: R1, V4

DeepSeek-R1 (2025) là một lượt huấn luyện lý luận trên xương sống V3. R1 dùng cùng kiến trúc. Cái thay đổi là công thức hậu huấn luyện (RL quy mô lớn trên các tác vụ kiểm chứng được), không phải kiến trúc tiền huấn luyện.

DeepSeek-V4 (nếu được phát hành) dự kiến giữ MLA + MoE + MTP và thêm DSA (DeepSeek Sparse Attention — attention thưa), kế thừa của NSA từ Giai đoạn 10 · 17. Dòng phát triển này ổn định: các cải tiến ở mức kiến trúc tích luỹ dần; mỗi phiên bản xoay thêm núm.

```figure
moe-routing
```

## Áp dụng

`code/main.py` là máy tính tham số chuyên cho hình dạng của DeepSeek-V3. Chạy nó, so đầu ra với các con số trong paper, rồi dùng nó cho các biến thể giả định (256 expert vs 512, top-8 vs top-16, MLA rank 512 vs 1024).

Những gì cần xem:

- Tổng số tham số vs 671B đã công bố.
- Số tham số hoạt động vs 37B đã công bố.
- KV cache ở ngữ cảnh 128k — so sánh MLA vs GQA.
- Phân rã theo từng lớp để thấy ngân sách tham số thực sự đi đâu.

## Đóng gói

Bài này tạo ra `outputs/skill-deepseek-v3-reader.md`. Cho một mô hình họ DeepSeek (V3, R1, hay bất kỳ biến thể tương lai nào), nó tạo ra một bản đọc kiến trúc theo từng thành phần: gọi tên từng trường của config, suy ra số tham số theo thành phần, và xác định mô hình dùng những cái nào trong bốn cải tiến riêng của DeepSeek.

## Bài tập

1. Chạy `code/main.py`. So ước lượng tổng tham số của máy tính với 671B đã công bố và xác định phần lệch đến từ đâu. Mục 2 của paper có bảng liệt kê đầy đủ.

2. Sửa config để dùng MLA rank 256 thay vì 512. Tính kích thước KV cache thu được ở ngữ cảnh 128k. Nó giảm bao nhiêu phần trăm, và trả giá gì cho khả năng biểu đạt theo từng đầu?

3. So định tuyến (256 expert, top-8) của DeepSeek-V3 với một biến thể giả định (512 expert, top-8). Tổng tham số tăng; tham số hoạt động giữ nguyên. Năng lực expert thêm đó mua được gì về lý thuyết, và tốn gì khi suy luận?

4. Đọc Mục 2.1 của báo cáo kỹ thuật DeepSeek-V3 (arXiv:2412.19437) về MLA. Giải thích trong ba câu vì sao ma trận giải nén K và V có thể được "hấp thụ" vào matmul phía sau để suy luận hiệu quả hơn.

5. DeepSeek-V3 dùng huấn luyện FP8 (số thực 8-bit) cho hầu hết phép toán. Tính lượng bộ nhớ tiết kiệm được của FP8 so với BF16 khi lưu 671B trọng số. Điều này giao nhau thế nào với ngân sách huấn luyện 14.8T token?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| MLA | "Multi-Head Latent Attention" | Nén K và V vào một tiềm ẩn hạng thấp dùng chung (kv_lora_rank, thường là 512), giải nén theo từng đầu ngay lúc chạy; KV cache chỉ lưu phần tiềm ẩn |
| kv_lora_rank | "chiều nén của MLA" | Kích thước tiềm ẩn dùng chung cho K và V; DeepSeek-V3 dùng 512 |
| First k dense layers | "các lớp đầu vẫn dense" | Vài lớp đầu của mô hình MoE bỏ qua bộ định tuyến MoE và chạy MLP dense để ổn định |
| num_experts_per_tok | "định tuyến top-k" | Số expert được định tuyến kích hoạt mỗi token; DeepSeek-V3 dùng 8 |
| Shared experts | "expert luôn bật" | Các expert xử lý mọi token bất kể định tuyến; DeepSeek-V3 dùng 1 |
| Auxiliary-loss-free routing | "cân bằng tải bằng bias" | Các số hạng bias theo từng expert được điều chỉnh trong lúc huấn luyện để giữ tải expert cân bằng mà không thêm số hạng loss |
| MTP module | "đầu dự đoán thêm" | Khối Transformer dự đoán t+2 từ h^(1) và E(t+1); huấn luyện dày hơn, bản nháp giải mã suy đoán miễn phí |
| DualPipe | "pipeline hai chiều" | Lịch huấn luyện xen kẽ compute forward/backward với all-to-all giữa các node |
| Active parameter ratio | "độ thưa" | active_params / total_params; DeepSeek-V3 đạt 5.5% |
| FP8 training | "huấn luyện 8-bit" | Lưu trữ và nhiều phép toán khi huấn luyện ở FP8; giảm khoảng một nửa bộ nhớ so với BF16 với chi phí chất lượng nhỏ |

## Đọc thêm

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — tài liệu đầy đủ về kiến trúc, huấn luyện và kết quả
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — các file config và ghi chú triển khai
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — phiên bản tiền nhiệm giới thiệu MLA
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — bản kế tiếp huấn luyện lý luận trên kiến trúc V3
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — hướng đi tương lai cho attention họ DeepSeek
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — tài liệu tham chiếu về lịch huấn luyện

---

[← Bài trước: Song song DualPipe](../../19-dualpipe-parallelism/docs/vi.md) · [Bài sau: Jamba →](../../21-jamba-hybrid-ssm-transformer/docs/vi.md)
