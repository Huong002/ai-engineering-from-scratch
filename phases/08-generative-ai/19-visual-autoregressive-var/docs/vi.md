# Mô hình tự hồi quy thị giác (VAR): dự đoán thang kế tiếp

> Mô hình diffusion lấy mẫu lặp theo thời gian (các bước khử nhiễu). VAR lấy mẫu lặp theo thang — next-scale prediction (dự đoán thang kế tiếp): mô hình dự đoán một token 1x1, rồi 2x2, rồi 4x4, cho tới độ phân giải cuối, mỗi thang đều điều kiện hoá trên thang trước đó. Bài báo năm 2024 cho thấy VAR tuân theo quy luật scaling kiểu GPT khi sinh ảnh và vượt DiT với cùng ngân sách tính toán. Bài này dựng cơ chế cốt lõi đó.

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Giai đoạn 7, Bài 03 (Attention nhiều đầu), Giai đoạn 8, Bài 06 (DDPM)
**Time:** ~90 phút

## Vấn đề

Sinh tự hồi quy thống trị mô hình ngôn ngữ vì nó mở rộng một cách dễ dự đoán: thêm tính toán, thêm tham số, perplexity thấp hơn, đầu ra tốt hơn. Sinh ảnh có hai hướng AR chính trước 2024: PixelRNN/PixelCNN (từng điểm ảnh) và DALL-E 1 / Parti / MuseGAN (từng token trên mã VQ-VAE).

Cả hai đều mắc vấn đề thứ tự sinh. Điểm ảnh và token nằm trên lưới 2D, nhưng mô hình AR buộc phải duyệt chúng theo thứ tự quét 1D. Một điểm ảnh ở góc xuất hiện sớm không hề biết ảnh cuối cùng sẽ ra sao. Chất lượng sinh mở rộng kém hơn GPT trên văn bản và không bao giờ đạt chất lượng của mô hình diffusion với cùng mức tính toán.

VAR sửa vấn đề thứ tự sinh bằng cách đổi thứ được sinh ra. Thay vì dự đoán token ảnh lần lượt theo không gian, VAR dự đoán cả ảnh ở độ phân giải tăng dần. Bước 1: dự đoán một token 1x1 (bản "tóm tắt" toàn ảnh). Bước 2: dự đoán lưới token 2x2 (đặc trưng thô hơn). Bước 3: dự đoán lưới 4x4. Bước K: dự đoán lưới (H/8)x(W/8) cuối cùng.

Mỗi thang chú ý tới mọi thang trước đó (nhân quả theo "thứ tự thang") và song song trong nội bộ thang của nó. Vấn đề thứ tự biến mất: toàn bộ ảnh ở thang k được sinh ra trong một lượt truyền qua transformer.

## Khái niệm

### Tokenizer VQ-VAE đa thang (multi-scale)

VAR cần một **tokenizer rời rạc đa thang**. Với một ảnh x, nó tạo ra một chuỗi bản đồ token (token map) có độ phân giải tăng dần:

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

Mỗi z_k dùng chung một sổ mã (codebook), kích thước thường 4096-16384. Việc tokenize ở mỗi thang không độc lập — nó được huấn luyện sao cho cộng các phần dư ở từng thang lại thì tái dựng được f:

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

Đây là một biến thể **residual VQ** (lượng tử hoá vector phần dư). Thang k nắm bắt phần mà các thang 1..k-1 bỏ sót. Decoder lấy tổng embedding của mọi thang rồi tạo ra ảnh.

Tokenizer VQ đa thang được huấn luyện một lần (giống VQGAN) rồi đóng băng. Toàn bộ công việc sinh ảnh do mô hình tự hồi quy nằm phía trên đảm nhiệm.

### Dự đoán thang kế tiếp

Mô hình sinh là một transformer nhìn thấy token của mọi thang trước và dự đoán token ở thang kế tiếp.

Cấu trúc chuỗi đầu vào:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Position embedding mã hoá cả chỉ số thang lẫn vị trí không gian trong thang. Attention là nhân quả theo thứ tự thang: token ở thang k, vị trí (i, j) có thể chú ý tới mọi token ở các thang 1..k và tới các token ở chính thang k xuất hiện trước đó theo bất kỳ thứ tự nội thang nào đang dùng (VAR dùng attention vị trí cố định, không có nhân quả nội thang — mọi vị trí trong một thang được dự đoán song song).

Loss huấn luyện: ở mỗi thang k, dự đoán token z_k khi đã có mọi token của các thang trước. Dùng cross-entropy loss trên các mã VQ rời rạc. Cấu trúc giống GPT, chỉ khác là "chuỗi" giờ được tổ chức theo thang.

### Sinh ảnh

Khi suy luận:
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

Với K = 10 thang, sinh ảnh là 10 lượt truyền xuôi qua transformer. Mỗi lượt sinh trọn một thang song song — không có tự hồi quy từng token trong một thang. Với ảnh 256x256, đó là khoảng 10 lượt so với 28-50 lượt của DiT.

### Vì sao dự đoán thang thắng dự đoán token

Ba lợi thế mang tính cấu trúc:
1. **Thô tới mịn khớp với thống kê ảnh tự nhiên.** Cả tri giác thị giác của con người lẫn các bộ dữ liệu ảnh đều thể hiện quy luật phụ thuộc thang: cấu trúc tần số thấp ổn định và dễ dự đoán; chi tiết tần số cao phụ thuộc vào nội dung tần số thấp. Dự đoán thang kế tiếp khai thác đúng điều này.
2. **Sinh song song trong một thang.** Khác với AR theo token kiểu GPT, VAR sinh mọi token của một thang trong một bước. Độ dài sinh hiệu dụng co lại theo log của số thang thay vì tuyến tính.
3. **Không thiên lệch thứ tự sinh.** Token ở thang k thấy toàn bộ thang k-1; không có thiên lệch "bên trái" hay "phía trên" buộc các token đầu phải chốt trước khi ngữ cảnh phía sau xuất hiện.

### Quy luật scaling

Tian và cộng sự đã chứng minh VAR tuân theo đường cong scaling dạng luỹ thừa cho FID trên ImageNet — giống như GPT với perplexity. Tăng gấp đôi tham số hoặc tính toán giúp giảm một nửa sai số một cách ổn định. Đây là mô hình sinh ảnh đầu tiên thể hiện hành vi scaling sạch sẽ như các mô hình ngôn ngữ. Kết quả là hiệu năng của VAR trở nên suy ra được từ lượng tính toán, chứ không còn là phỏng đoán thực nghiệm cho từng kiến trúc.

### Quan hệ với diffusion

VAR và diffusion chia sẻ cùng một câu chuyện nén dữ liệu: cả hai chia bài toán sinh thành một chuỗi bài toán con dễ hơn.

- Diffusion: thêm nhiễu dần dần, học cách hoàn tác một bước.
- VAR: thêm độ phân giải dần dần, học cách dự đoán thang kế tiếp.

Chúng là hai trục khác nhau đi qua cùng một bài toán. Cả hai đều cho ra phân phối có điều kiện dễ xử lý. Trên thực nghiệm, VAR suy luận nhanh hơn (ít lượt hơn, tất cả song song trong một thang) và ngang hoặc vượt DiT trên ImageNet có điều kiện lớp. VAR có điều kiện văn bản (VARclip, HART) là một hướng nghiên cứu đang sôi động.

```figure
gx-var-next-scale
```

## Tự tay làm

Trong `code/main.py` bạn sẽ:
1. Dựng một **tokenizer VQ đa thang** nhỏ trên dữ liệu "ảnh" tổng hợp (các vòng Gaussian 2D).
2. Huấn luyện một **transformer kiểu VAR** để dự đoán token theo thang kế tiếp.
3. Lấy mẫu bằng cách gọi transformer 4 lần (4 thang) rồi decode.
4. Kiểm chứng rằng huấn luyện theo thứ tự thang khiến việc sinh song song trong một thang.

Đây là bản cài đặt đồ chơi. Mục đích là để thấy mặt nạ attention có cấu trúc theo thang và việc sinh song song trong thang thực sự hoạt động.

## Đóng gói

Bài này tạo ra `outputs/skill-var-tokenizer-designer.md` — một skill để thiết kế tokenizer đa thang: số thang, tỉ lệ thang, kích thước sổ mã, chia sẻ phần dư, kiến trúc decoder.

## Bài tập

1. **Thử bỏ số thang.** Huấn luyện VAR với 4, 6, 8, 10 thang. Đo chất lượng tái dựng so với số lượt tự hồi quy. Nhiều thang hơn = phần dư mịn hơn = chất lượng tốt hơn nhưng nhiều lượt hơn.

2. **Kích thước sổ mã.** Huấn luyện tokenizer với kích thước sổ mã 512, 4096, 16384. Sổ mã lớn hơn cho tái dựng tốt hơn nhưng dự đoán khó hơn. Tìm điểm gãy.

3. **Kiểm tra song song trong thang.** Với một VAR đã huấn luyện, đo trực tiếp mẫu attention. Trong thang k, mô hình có chú ý tới các vị trí khác thang mà không chú ý nội thang không? Kiểm chứng phần cài đặt mặt nạ.

4. **Scaling VAR so với DiT.** Với cùng bài toán ImageNet có điều kiện lớp, huấn luyện VAR và DiT ở ngân sách tham số tương đương (ví dụ 33M, 130M, 458M). Vẽ FID theo lượng tính toán. VAR phải vượt DiT ở mỗi kích thước — tái lập kết quả của bài báo ở quy mô nhỏ.

5. **Điều kiện hoá văn bản.** Mở rộng VAR để nhận embedding văn bản (CLIP pooled) làm đầu vào điều kiện bổ sung qua adaLN. Đây là công thức của HART. FID cải thiện bao nhiêu khi lấy mẫu bám theo văn bản?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|----------------------|
| VAR | "Visual AutoRegressive" | Sinh ảnh bằng dự đoán thang kế tiếp trên một kim tự tháp các bản đồ token VQ |
| Next-scale prediction | "Predict coarser, then finer" | Mô hình dự đoán token ở các thang độ phân giải tăng dần, điều kiện hoá trên mọi thang trước |
| Multi-scale VQ tokenizer | "Residual VQ" | VQ-VAE tạo ra K bản đồ token có độ phân giải tăng dần, decoder cộng tất cả các thang |
| Scale k | "Pyramid level k" | Một trong K mức độ phân giải, từ 1x1 khi k=1 đến (H/p)x(W/p) khi k=K |
| Parallel-within-scale | "One forward per scale" | Mọi token ở thang k được dự đoán trong một lượt truyền qua transformer, không tự hồi quy |
| Causal-across-scales | "Scale-ordered attention" | Token ở thang k có thể chú ý tới mọi thang 1..k nhưng không tới các thang k+1..K |
| Residual VQ | "Additive tokenization" | Token của mỗi thang mã hoá phần dư mà các thang thấp hơn để lại; decoder cộng mọi embedding thang |
| VAR scaling law | "Image GPT scaling" | FID tuân theo quy luật luỹ thừa dễ dự đoán theo lượng tính toán, như perplexity của mô hình ngôn ngữ |
| HART | "Hybrid VAR + text" | Biến thể VAR có điều kiện văn bản, kết hợp giải mã lặp kiểu MaskGIT với cấu trúc thang của VAR |
| Scale position embedding | "(scale, row, col) triple" | Mã hoá vị trí mang cả chỉ số thang lẫn toạ độ không gian trong thang |

## Đọc thêm

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — bài báo VAR, tài liệu tham chiếu chuẩn
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT, mốc so sánh với diffusion
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN, họ tokenizer mà tokenizer đa thang của VAR mở rộng
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE, nền tảng của tokenize ảnh rời rạc
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — VAR có điều kiện văn bản

---

[← Bài trước: Đánh giá](../../14-evaluation-fid-clip-score/docs/vi.md)
