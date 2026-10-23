# Native Sparse Attention (DeepSeek NSA)

> Ở 64k token, attention chiếm 70-80% độ trễ giải mã (decode). Lab nào làm mô hình mở cũng có kế hoạch khắc phục. NSA của DeepSeek (bài tốt nhất ACL 2025) là hướng đã trụ lại: ba nhánh attention song song — token thô đã nén, token chi tiết được giữ có chọn lọc, và cửa sổ trượt cho ngữ cảnh cục bộ — kết hợp qua một cổng có học. Nó căn theo phần cứng (thân thiện với kernel), huấn luyện được ngay từ gốc (chạy trong tiền huấn luyện, không phải gắn thêm lúc suy luận), và ở giải mã 64k nó chạy nhanh hơn FlashAttention trong khi chất lượng bằng hoặc hơn full attention. Bài này dựng ba nhánh từ đầu đến cuối và chỉ ra vì sao độ thưa này khả vi từ đầu đến cuối.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 7, Bài 12 (KV cache, flash-attention), Giai đoạn 7, Bài 15 (các biến thể attention), Giai đoạn 10, Bài 16 (differential attention)
**Time:** ~60 phút

## Mục tiêu học tập

- Nêu ba nhánh attention của NSA và mỗi nhánh nắm bắt điều gì.
- Giải thích vì sao NSA "huấn luyện được từ gốc" trong khi các phương pháp attention thưa trước đó chỉ dùng được lúc suy luận.
- Tính mức tiết kiệm compute của NSA so với full attention ở ngữ cảnh 64k như một hàm của kích thước khối nén và top-k chọn lọc.
- Cài đặt tổ hợp ba nhánh bằng Python thuần trên một chuỗi tổng hợp ngắn và kiểm chứng trọng số cổng hoạt động đúng.

## Vấn đề

Full attention ở độ dài chuỗi N tốn `O(N^2)` thời gian và `O(N)` KV cache mỗi lớp. Ở 64k token, các con số compute và băng thông bộ nhớ là thảm hoạ. Ước lượng lý thuyết đo được từ bài báo NSA: attention chiếm 70-80% tổng độ trễ giải mã ở 64k. Mọi thứ phía sau — TTFT, token/giây, chi phí trên một triệu token — đều bị chi phối bởi chi phí attention.

Attention thưa là câu trả lời hiển nhiên. Các nỗ lực trước rơi vào hai nhóm. Độ thưa mẫu cố định (cửa sổ trượt, bước nhảy, khối cục bộ) vứt bỏ thông tin và thất bại trên các bài truy hồi tầm xa. Độ thưa lúc suy luận (tỉa KV cache, H2O, StreamingLLM) được áp lên một mô hình đã tiền huấn luyện với attention dày đặc và chỉ lấy lại được một phần tốc độ tiềm năng, vì mô hình chưa bao giờ được yêu cầu định tuyến thông tin qua mẫu thưa.

Native Sparse Attention (attention thưa gốc) (Yuan et al., DeepSeek + PKU + UW, bài tốt nhất ACL 2025, arXiv:2502.11089) làm cả hai: một mẫu thưa mà mô hình học trong tiền huấn luyện, được cài như thuật toán căn theo kernel thực sự mang lại mức tiết kiệm compute lúc suy luận. Hai năm nữa, NSA hoặc hậu duệ trực tiếp của nó sẽ là attention mặc định trên mọi mô hình ngữ cảnh dài (long context) tiên phong.

## Khái niệm

### Ba nhánh song song

Với mỗi query, NSA chạy attention ba lần, trên ba góc nhìn khác nhau của KV cache:

1. **Nhánh nén** (compression branch). Các token được gom thành các khối (block) kích thước `l` (thường là 32 hoặc 64). Mỗi khối được nén thành một token tóm tắt duy nhất qua một MLP nhỏ có học. Query chú ý trên các token nén này, thu được góc nhìn thô trên toàn chuỗi.

2. **Nhánh chọn lọc.** Dùng điểm attention từ nhánh nén, phép chọn thưa (sparse selection) xác định top-k khối liên quan nhất tới query hiện tại. Các token chi tiết (không nén) từ những khối đó được đọc và query chú ý trên tất cả chúng. Hãy coi attention của nhánh nén như tín hiệu định tuyến cho bước chọn.

3. **Nhánh cửa sổ trượt.** Query chú ý tới `W` token gần nhất (thường 512) để lấy ngữ cảnh cục bộ. Nhánh này nắm các mẫu tầm ngắn nặng về cấu trúc (cú pháp, đồng tham chiếu cục bộ) mà hai nhánh kia có thể bỏ sót.

Đầu ra của ba nhánh được kết hợp qua một cổng có học theo từng vị trí:

```
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` là trọng số cổng lấy từ một MLP nhỏ áp trên query. Chúng không cần cộng lại bằng 1 — mỗi nhánh có thể được đánh trọng số độc lập.

### Vì sao đây là "huấn luyện được từ gốc"

Bước chọn (các khối top-k) là rời rạc. Các phép toán rời rạc làm đứt dòng gradient. Công trình attention thưa trước đó hoặc bỏ qua lan truyền ngược qua bước chọn (hạn chế huấn luyện), hoặc dùng nới lỏng liên tục nhưng không cho độ thưa thật lúc suy luận.

NSA né chuyện này: attention của nhánh nén CHÍNH LÀ một attention thô khả vi trên toàn chuỗi. Phép top-k chỉ tái dùng các điểm attention cao nhất từ nhánh nén để chọn khối chi tiết nào cần nạp. Gradient chảy qua điểm của nhánh nén (những điểm ảnh hưởng cả đầu ra nén VÀ logic chọn), và phần đóng góp của các khối được chọn vào đầu ra cuối cũng khả vi. Phép `top_k` không khả vi là no-op trên đồ thị tính toán xuôi — nó chỉ quyết định khối nào được nạp từ bộ nhớ.

Đó là lý do NSA dùng được trong tiền huấn luyện từ đầu đến cuối. Mô hình học định tuyến thông tin qua ba nhánh cùng lúc, tạo ra một mẫu thưa mà lúc suy luận thực sự mang lại tốc độ đã hứa.

### Kernel căn theo phần cứng

Kernel của NSA được thiết kế cho phân cấp bộ nhớ GPU hiện đại. Kernel nạp query theo nhóm GQA (grouped-query attention) ở vòng ngoài, lấy các khối KV thưa tương ứng cho từng nhóm ở vòng trong, và chạy attention trên SRAM. Vì mỗi nhóm query thấy cùng các khối được chọn (chọn theo nhóm query, không theo từng head), việc nạp KV được chia đều trong nhóm. Cường độ số học vẫn cao.

Bài báo báo cáo kernel Triton chạy nhanh hơn FlashAttention 9 lần ở giải mã 64k, tỉ lệ tăng tốc lớn dần theo độ dài chuỗi. Cả kernel xuôi và ngược đều được cung cấp.

### Ngân sách compute

Cho `N` là độ dài chuỗi, `l` là kích thước khối nén, `k` là số lượng chọn top-k, `w` là cửa sổ trượt, `b` là kích thước khối được chọn (thường bằng `l`).

- Nhánh nén: `O(N/l)` khoá cho mỗi query, nên tổng `O(N * N / l)`.
- Nhánh chọn lọc: `O(k * b)` khoá cho mỗi query, nên `O(N * k * b)`.
- Nhánh cửa sổ: `O(w)` khoá cho mỗi query, nên `O(N * w)`.

Tổng độ phức tạp (complexity): `O(N * (N/l + k*b + w))`.

Với `N = 64k, l = 64, k = 16, b = 64, w = 512`: chi phí mỗi query là `1000 + 1024 + 512 = 2536 keys`. Full attention là `64000 keys`. Giảm 25 lần compute.

Với `N = 128k, l = 64, k = 16, b = 64, w = 512`: chi phí mỗi query là `2000 + 1024 + 512 = 3536 keys`. Full attention là `128000 keys`. Giảm 36 lần. Lợi ích tăng theo độ dài chuỗi, và đó chính là mục đích.

### So sánh thế nào

| Phương pháp | Khả vi | Tăng tốc suy luận thật | Truy hồi tầm xa |
|--------|---------------|----------------------|-------------------|
| Chỉ cửa sổ trượt | có | có | thất bại |
| Bước nhảy / khối thưa | có | có | một phần |
| Tỉa KV (H2O, StreamingLLM) | N/A (lúc suy luận) | có | một phần |
| MoBA (Moonshot) | một phần | có | tốt |
| NSA | có (từ gốc) | có (9 lần ở 64k) | ngang full attention |

MoBA (Moonshot, arXiv:2502.13189) được công bố cùng thời và theo hướng tiếp cận tương tự "ba nhánh tốt hơn một", áp nguyên lý MoE lên các khối attention. NSA và MoBA là hai kiến trúc cần biết cho tiền huấn luyện ngữ cảnh dài năm 2026.

```figure
sliding-window-attention
```

## Tự tay làm

`code/main.py` cài đặt ba nhánh trên một chuỗi tổng hợp ngắn và cho thấy:

- MLP nén (bản baseline mean-pool đơn giản được dùng cho mục đích sư phạm; NSA thật dùng MLP có học).
- Bước chọn khối top-k dựa trên điểm của nhánh nén.
- Attention cửa sổ trượt trên `w` token cuối.
- Kết hợp có cổng.
- Bản in đếm compute so với full attention.

### Bước 1: nén token thành khối

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### Bước 2: attention nhánh nén

Chạy softmax attention của query với các khoá đã nén. Điểm của nhánh nén đóng luôn vai tín hiệu cho bước chọn top-k.

### Bước 3: chọn khối top-k

Chọn chỉ số của `k` khối nén có điểm cao nhất (top-k). Nạp các token gốc chưa nén từ những khối đó và chạy attention trên chúng.

### Bước 4: attention cửa sổ trượt

Lấy `w` token cuối và chạy attention chuẩn với chúng.

### Bước 5: cổng + kết hợp

Một MLP nhỏ trên query tạo ra ba trọng số cổng. Đầu ra cuối là tổng có trọng số của đầu ra ba nhánh.

### Bước 6: đếm compute

In số khoá được chú ý cho mỗi query ở từng nhánh và tổng. So với `N` (full attention). Trên chuỗi tổng hợp 1024 token với `l = 32, k = 4, w = 128`, NSA thấy `32 + 128 + 128 = 288` khoá mỗi query so với 1024 của full attention — ít hơn 3.5 lần.

## Áp dụng

NSA đang được đưa vào pipeline tiền huấn luyện ngữ cảnh dài của chính DeepSeek. Tình trạng tích hợp trong các stack suy luận công khai tính đến tháng 4/2026:

- **DeepSeek nội bộ**: gốc, trọng số công bố dùng NSA hoặc hậu duệ DSA (Deepseek Sparse Attention).
- **vLLM**: hỗ trợ NSA thử nghiệm đang phát triển cho trọng số DeepSeek-V3.x.
- **SGLang**: đã công bố benchmark NSA; đường sản phẩm đi sau vLLM.
- **llama.cpp / CPU**: không hỗ trợ; chi phí phân rã kernel không đáng ở thông lượng CPU.

Khi nào nên dùng NSA:

- Lượt tiền huấn luyện hoặc huấn luyện tiếp nhắm tới ngữ cảnh 64k trở lên với ngân sách compute nghiêm túc.
- Suy luận các checkpoint ngữ cảnh dài của chính DeepSeek. Trọng số vốn là NSA gốc.

Khi nào không nên:

- Phục vụ một mô hình đã tiền huấn luyện với attention dày đặc. Bạn không thể gắn thêm NSA nếu không huấn luyện tiếp.
- Ngữ cảnh dưới 16k. Chi phí ba nhánh lớn hơn phần tiết kiệm.
- Chat tương tác batch-1. Giải mã nhạy độ trễ có lợi, nhưng chỉ ở ngữ cảnh dài.

## Đóng gói

Bài này tạo ra `outputs/skill-nsa-integrator.md`. Cho một đặc tả lượt tiền huấn luyện ngữ cảnh dài, nó tạo ra kế hoạch tích hợp NSA: kích thước khối nén, top-k, cửa sổ trượt, độ rộng MLP cổng, lựa chọn kernel, và các eval ngữ cảnh dài cụ thể để biện minh cho việc đổi kiến trúc.

## Bài tập

1. Chạy `code/main.py` trên một chuỗi tổng hợp 1024 token. Quét `(l, k, w)` qua ba preset và in số compute. Xác định preset đạt số khoá mỗi query thấp nhất mà vẫn giữ 95% truy hồi so với full attention trong bài kiểm tra kim trong đống cỏ (needle-in-haystack).

2. Thay bộ nén mean-pool bằng một MLP nhỏ có học (2 lớp, ẩn 32). Huấn luyện nó trên một bài tổng hợp mà tín hiệu là trung bình của một khối. Đo khoảng cách perplexity so với baseline mean-pool trên dữ liệu giữ riêng.

3. Cài đặt MLP cổng. Nó nhận query làm đầu vào và xuất ba số vô hướng. Chứng minh cổng hành xử hợp lý: trọng số gần đều trên query ngẫu nhiên, trọng số lớn cho nhánh chọn lọc khi query trúng một khối ở xa.

4. Tính ngân sách bộ nhớ KV cache cho một mô hình 70B bật NSA ở ngữ cảnh 128k. KV head là 8, head dim 128, BF16. So với full attention và với MLA (Giai đoạn 10 · 14 đã cho số của MLA). Xác định độ dài chuỗi mà tại đó KV cache của nhánh chi tiết trong NSA bằng full attention.

5. Đọc Mục 4 của bài báo NSA (arXiv:2502.11089) và giải thích trong ba câu vì sao điểm attention của nhánh nén được tái dùng cho bước chọn top-k thay vì tính một điểm định tuyến riêng. Gắn câu trả lời với dòng gradient.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Compressed branch | "Góc nhìn thô" | Attention trên các khoá đã lấy trung bình theo khối, cung cấp ngữ cảnh toàn cục với O(N/l) khoá mỗi query |
| Selected branch | "Các khối top-k" | Attention chi tiết trên `k` khối có điểm nhánh nén cao nhất |
| Sliding window | "Ngữ cảnh cục bộ" | Attention trên `W` token cuối cho các mẫu tầm ngắn |
| Native trainability | "Tiền huấn luyện kèm độ thưa" | Mẫu thưa được học trong tiền huấn luyện, không gắn thêm lúc suy luận |
| Compression block size l | "Kích thước nhóm cho góc nhìn thô" | Bao nhiêu token được gộp vào một bản tóm tắt; thường 32-64 |
| Top-k | "Số khối giữ lại" | Số khối nén mà token chưa nén của chúng được đọc; thường 16 |
| Sliding window W | "Bán kính attention cục bộ" | Thường 512; ngắn hơn làm giảm mạch lạc cục bộ, dài hơn thì phí compute |
| Branch gate | "Cách trộn ba nhánh" | Đầu ra MLP theo từng vị trí, đánh trọng số đóng góp của ba nhánh |
| Hardware alignment | "Độ thưa thân thiện kernel" | Mẫu thưa được chọn sao cho kernel GPU thật đạt tốc độ lý thuyết |
| DSA | "Hậu duệ của NSA" | Deepseek Sparse Attention, kiến trúc nối tiếp NSA trong dòng phát triển của DeepSeek |

## Đọc thêm

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — bài báo
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — họ kiến trúc mà NSA nhắm tới
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) — công trình cùng thời, attention kiểu MoE trên các khối
- [Beltagy et al. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — nguồn gốc của cửa sổ trượt
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) — baseline độ thưa lúc suy luận mà NSA cải thiện
- [Dao et al. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — baseline full attention mà kernel NSA đánh bại ở 64k

---

[← Bài trước: Attention sai phân (V2)](../../16-differential-attention-v2/docs/vi.md) · [Bài sau: Dự đoán nhiều token (MTP) →](../../18-multi-token-prediction/docs/vi.md)
