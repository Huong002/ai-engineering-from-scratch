# CLIP và tiền huấn luyện tương phản thị giác-ngôn ngữ

> CLIP (2021) của OpenAI đã chứng minh một ý tưởng đủ lớn để nuôi sống năm năm tiếp theo: căn chỉnh một bộ mã hoá ảnh và một bộ mã hoá văn bản vào cùng một không gian vector chỉ bằng các cặp ảnh-chú thích nhiễu lấy từ web và contrastive learning (học tương phản). Không cần nhãn giám sát. 400M cặp. Không gian embedding (embedding space) thu được làm được zero-shot classification (phân loại zero-shot), truy hồi ảnh-văn bản, và cắm vào mọi VLM năm 2026 như tháp thị giác của nó. SigLIP 2 (2025) thay softmax bằng sigmoid và mở rộng quy mô vượt CLIP với chi phí thấp hơn. Bài này đi qua phần toán từ InfoNCE tới sigmoid pairwise loss và tự dựng bước huấn luyện bằng Python thuần.

**Type:** Build
**Languages:** Python (stdlib, InfoNCE + sigmoid loss implementations)
**Prerequisites:** Giai đoạn 12, Bài 01 (Patch ViT), Giai đoạn 7 (Transformer)
**Time:** ~180 phút

## Mục tiêu học tập

- Suy ra InfoNCE loss từ mutual information và cài đặt một phiên bản vector hoá ổn định về số học.
- Giải thích vì sao sigmoid pairwise loss (SigLIP) mở rộng tới batch (lô) 32768+ mà không cần overhead all-gather như softmax đòi hỏi.
- Chạy phân loại zero-shot trên ImageNet bằng cách dựng text template (`a photo of a {class}`) rồi lấy argmax trên cosine similarity.
- Kể tên bốn đòn bẩy mà tiền huấn luyện CLIP / SigLIP mang lại: kích thước batch, temperature, prompt template, chất lượng dữ liệu.

## Vấn đề

Thị giác trước CLIP là học có giám sát. Thu thập bộ dữ liệu gán nhãn (ImageNet: 1.2M ảnh, 1000 lớp), huấn luyện một CNN, đem đi dùng. Nhãn thì đắt, nhãn bị thiên lệch theo thứ mà người gán nhãn có thể đồng thuận, và nhãn không chuyển được sang tác vụ mới nếu không fine-tune.

Web ảnh-chú thích có hơn một tỷ cặp gắn nhãn lỏng lẻo miễn phí. Một bức ảnh chó golden retriever với alt text "my dog Max in the park" mang tín hiệu giám sát — đoạn text mô tả bức ảnh. Câu hỏi: bạn có biến nó thành dữ liệu huấn luyện hữu ích được không?

Câu trả lời của CLIP: coi cặp ảnh-chú thích như một bài toán khớp. Cho một batch gồm N ảnh và N chú thích, hãy học cách khớp mỗi ảnh với chú thích của chính nó trước N-1 phương án gây nhiễu. Tín hiệu giám sát là "hai thứ này thuộc về nhau; N-1 thứ kia thì không." Không nhãn lớp. Không chú thích thủ công. Chỉ một mất mát tương phản.

Không gian embedding thu được làm được nhiều hơn thứ CLIP được huấn luyện để làm. Zero-shot trên ImageNet hoạt động vì "a photo of a cat" nằm gần các bức ảnh mèo chưa từng được gán nhãn mèo một cách tường minh. Đây là cược đã sinh ra mọi VLM năm 2026.

## Khái niệm

### Bộ mã hoá kép

CLIP có hai tháp:

- Bộ mã hoá ảnh `f`: ViT hoặc ResNet, xuất ra một vector D chiều cho mỗi ảnh.
- Bộ mã hoá văn bản `g`: một transformer nhỏ, xuất ra một vector D chiều cho mỗi chú thích.

Cả hai tháp chuẩn hoá đầu ra về độ dài đơn vị. Độ tương đồng là `cos(f(x), g(y)) = f(x)^T g(y)` vì cả hai đều có chuẩn đơn vị.

Với một batch gồm N cặp (ảnh, chú thích), dựng ma trận tương đồng `S` có kích thước `(N, N)`:

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

trong đó `tau` là temperature (nhiệt độ) được học (CLIP khởi tạo 0.07; học trong log-space).

### Mất mát InfoNCE

CLIP dùng cross-entropy đối xứng trên hàng và cột:

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

Đây là InfoNCE. Phép softmax trong CE buộc mỗi ảnh khớp với chú thích của nó hơn mọi chú thích khác trong batch. Các negative pair (cặp âm) là mọi mục khác trong lô. Lô lớn hơn = nhiều cặp âm hơn = tín hiệu mạnh hơn. CLIP huấn luyện ở batch 32k; quy mô quan trọng.

### Temperature

`tau` điều khiển độ nhọn của softmax. tau thấp → phân phối nhọn, tạo hiệu ứng hard negative mining. tau cao → mềm, mọi mẫu đều đóng góp. CLIP học log(1/tau), có cắt để tránh sụp. SigLIP 2 cố định tau ban đầu và dùng một bias được học thay thế.

### Vì sao sigmoid mở rộng tốt hơn (SigLIP)

Softmax cần cả ma trận tương đồng đồng bộ. Trong huấn luyện phân tán, bạn phải all-gather mọi embedding tới mọi replica, rồi mới làm softmax. Chi phí truyền thông này bậc hai theo số lượng worker.

SigLIP thay softmax bằng sigmoid theo từng phần tử: với mỗi cặp `(i, j)`, mất mát là một bài toán phân loại nhị phân "hai thứ này có phải cặp khớp không?" nhãn của positive pair (cặp dương) là đường chéo, mọi thứ còn lại là âm. Mất mát là:

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` nếu `i == j`, ngược lại 0. Mất mát của mỗi cặp độc lập. Không cần all-gather. Mỗi GPU tính khối cục bộ của mình rồi cộng. SigLIP 2 mở rộng tới batch 32k-512k với chi phí thấp, trong khi CLIP sẽ cần truyền thông nhiều hơn tương ứng.

### Phân loại zero-shot

Cho N tên lớp, với mỗi lớp dựng một text template:

```
"a photo of a {class}"
```

Nhúng mỗi template bằng bộ mã hoá văn bản. Nhúng ảnh của bạn bằng bộ mã hoá ảnh. Argmax cosine similarity = lớp dự đoán. Không huấn luyện trên các lớp đích.

Prompt template quan trọng. Bài báo gốc của CLIP dùng 80 template mỗi lớp (thường, nghệ thuật, ảnh chụp, tranh vẽ, v.v.) và lấy trung bình các embedding. +3 điểm ImageNet. Cách dùng hiện đại thường chọn một hoặc hai template.

### Linear probe và fine-tuning

Zero-shot chỉ là mức nền. Một linear probe (huấn luyện một lớp tuyến tính trên đặc trưng CLIP đóng băng cho các lớp đích) đánh bại zero-shot trên tác vụ cùng miền. Fine-tuning toàn phần đánh bại linear probe trên cùng miền nhưng có thể làm hại khả năng chuyển zero-shot. Ba chế độ với ba đánh đổi.

### SigLIP 2: NaFlex và đặc trưng dày

SigLIP 2 (2025) bổ sung:
- NaFlex: một model duy nhất xử lý tỉ lệ khung hình và độ phân giải thay đổi.
- Đặc trưng dày tốt hơn cho segmentation và ước lượng độ sâu, nhắm tới việc dùng làm backbone đóng băng trong VLM.
- Đa ngôn ngữ: huấn luyện trên hơn 100 ngôn ngữ, trong khi CLIP chỉ tiếng Anh.
- Quy mô 1B tham số, trong khi CLIP dừng ở 400M.

Trong các VLM mở năm 2026, SigLIP 2 SO400m/14 là tháp thị giác mặc định. CLIP vẫn là mặc định cho truy hồi ảnh-văn bản thuần, nơi phân phối huấn luyện LAION-2B cụ thể khớp với kiểu truy vấn của bạn.

### ALIGN, BASIC, OpenCLIP, EVA-CLIP

ALIGN (Google, 2021): cùng ý tưởng với CLIP, quy mô 1.8B cặp, 90% nhiễu. Chứng minh dữ liệu nhiễu vẫn mở rộng được. OpenCLIP (LAION): bản tái lập mở của CLIP trên LAION-400M / 2B, nhiều quy mô, là checkpoint mở được dùng nhiều nhất. EVA-CLIP: khởi tạo từ masked image modeling; backbone mạnh cho VLM. BASIC: bản lai CLIP+ALIGN của Google. Tất cả cùng một họ, khác dữ liệu và cách tinh chỉnh.

### Trần của zero-shot

Các model họ CLIP dừng quanh 76% zero-shot trên ImageNet (CLIP-G, OpenCLIP-G). Vượt qua cần dữ liệu lớn hơn nhiều (SigLIP 2 đạt 80%+) hoặc thay đổi kiến trúc (head giám sát, nhiều tham số hơn). Benchmark đang bão hoà; giá trị thật nằm ở không gian embedding mà các VLM hạ nguồn tiêu thụ.

```figure
multimodal-fusion
```

## Áp dụng

`code/main.py` cài đặt:

1. Một bộ mã hoá kép đồ chơi (đặc trưng ảnh dựa trên hash, đặc trưng ký tự văn bản) để bạn thấy hình dạng của InfoNCE mà không cần numpy.
2. InfoNCE loss bằng Python thuần (ổn định số học nhờ log-sum-exp).
3. Sigmoid pairwise loss để so sánh.
4. Một quy trình phân loại zero-shot: tính cosine similarity với một tập text prompt, argmax để dự đoán.

Chạy nó và xem đường cong mất mát. Các con số tuyệt đối là đồ chơi; hình dạng thì khớp với thứ một trainer CLIP thật in ra.

## Đóng gói

Bài này tạo ra `outputs/skill-clip-zero-shot.md`. Cho một tập ảnh (qua đường dẫn) và danh sách lớp đích, nó dựng text prompt bằng template CLIP, nhúng cả hai phía bằng một checkpoint được nêu rõ (ví dụ `openai/clip-vit-large-patch14`), và trả về dự đoán top-1 / top-5 kèm điểm tương đồng. Skill từ chối đưa ra khẳng định về các lớp không có trong danh sách prompt.

## Bài tập

1. Tự tay cài InfoNCE cho một batch 4 cặp. Dựng ma trận tương đồng 4x4, chạy softmax, lấy đường chéo, tính cross-entropy. Đối chiếu bản cài Python của bạn với phép tính tay này.

2. SigLIP dùng thêm tham số bias `b` bên cạnh temperature: `S'[i,j] = S[i,j]/tau + b`. `b` đóng vai trò gì khi batch mất cân bằng lớp mạnh (mỗi hàng có nhiều cặp âm hơn cặp dương rất nhiều)? Đọc Mục 3 của SigLIP (arXiv:2303.15343).

3. Dựng bộ phân loại zero-shot cho mèo và chó. Thử hai prompt template: `a photo of a {class}` và `a picture of a {class}`. Đo độ chính xác trên 100 ảnh test. Ensemble của các template có thắng template đơn không?

4. Tính chi phí truyền thông của softmax InfoNCE so với sigmoid pairwise cho một lần chạy 512 GPU ở batch 32k. Cái nào mở rộng theo O(N), cái nào theo O(N^2)? Dẫn Mục 4 của SigLIP.

5. Đọc bài scaling laws của OpenCLIP (arXiv:2212.07143, Cherti và cộng sự). Tái lập kết luận của họ về mở rộng dữ liệu từ các hình: với kích thước model cố định, quan hệ tuyến tính-log giữa độ chính xác zero-shot ImageNet và kích thước dữ liệu huấn luyện là gì?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| InfoNCE | "Contrastive loss" | Cross-entropy trên ma trận tương đồng của một batch; cặp dương của mỗi mục là mục đi kèm nó, cặp âm là mọi thứ còn lại |
| Sigmoid loss | "SigLIP loss" | Binary cross-entropy theo từng cặp; không softmax, không all-gather, mở rộng rẻ trong huấn luyện phân tán |
| Temperature | "tau" | Vô hướng nhân vào logits trước softmax/sigmoid; điều khiển độ nhọn của phân phối |
| Zero-shot | "no-finetune classification" | Dùng text prompt để dựng embedding lớp và phân loại bằng cosine similarity; không huấn luyện trên lớp đích |
| Prompt template | "a photo of a ..." | Khung văn bản quanh tên lớp; ảnh hưởng 1-5 điểm độ chính xác zero-shot |
| Dual encoder | "Two-tower" | Một bộ mã hoá ảnh + một bộ mã hoá văn bản, đầu ra trong không gian D chiều chung |
| Hard negative | "Tough distractor" | Một cặp âm đủ giống cặp dương đến mức model phải nỗ lực để tách chúng |
| Linear probe | "Frozen + one layer" | Chỉ huấn luyện một bộ phân loại tuyến tính trên đặc trưng đóng băng; đo chất lượng đặc trưng |
| NaFlex | "Native flexible resolution" | Khả năng của SigLIP 2 nhận ảnh ở mọi tỉ lệ khung hình và độ phân giải mà không cần resize |
| Temperature scaling | "log-parametrized tau" | CLIP tham số hoá `log(1/tau)` để gradient hoạt động tốt; cắt để tránh sụp về tau gần 0 |

## Đọc thêm

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — bài báo CLIP.
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP.
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — đa ngôn ngữ + NaFlex.
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — mở rộng quy mô với dữ liệu web nhiễu.
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — scaling laws của OpenCLIP.

---

[← Bài trước: Vision Transformer (ViT) và primitive patch-token](../../01-vision-transformer-patch-tokens/docs/vi.md) · [Bài sau: Từ CLIP đến BLIP-2 →](../../03-blip2-qformer-bridge/docs/vi.md)
