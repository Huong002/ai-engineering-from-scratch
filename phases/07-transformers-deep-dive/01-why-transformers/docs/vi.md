# Vì sao có Transformer — Những vấn đề của RNN

> RNN xử lý token lần lượt từng cái một. Transformer xử lý tất cả token cùng lúc. Đúng một canh bạc kiến trúc đó đã thay đổi mọi đường cong scaling của deep learning sau năm 2017.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 3 (Deep Learning Core), Giai đoạn 5, Bài 09 (Mô hình chuỗi sang chuỗi), Giai đoạn 5, Bài 10 (Cơ chế attention)
**Time:** ~45 phút

## Vấn đề

Trước 2017, mọi mô hình chuỗi (sequence) đạt hiện đại nhất hành tinh — ngôn ngữ, dịch, tiếng nói — đều là mạng nơ-ron hồi quy. LSTM và GRU thắng các benchmark dịch tương đương ImageNet suốt nửa thập kỷ. Chúng là công cụ duy nhất người ta có.

Chúng có ba điểm yếu chí mạng. Tính toán tuần tự nghĩa là bạn không thể song song hoá (parallelization) theo trục thời gian: token `t+1` cần trạng thái ẩn từ token `t`. Một chuỗi 1,024 token nghĩa là 1,024 bước nối tiếp trên một GPU có thể làm 1,000,000 phép floating-point mỗi chu kỳ. Thời gian train tính theo đồng hồ treo tường tăng tuyến tính với độ dài chuỗi trên phần cứng vốn được thiết kế cho song song.

Gradient tiêu biến nghĩa là thông tin lùi 50 token đã bị nén qua 50 phép phi tuyến. Các đơn vị hồi quy có cổng (LSTM, GRU) làm dịu sự nghiền nát đó nhưng không bao giờ xoá bỏ được. Những phụ thuộc xa (long-range dependency) — "cuốn sách tôi đọc mùa hè trước trên một chuyến bay sang Kyoto là…" — thường xuyên thất bại.

Trạng thái ẩn có độ rộng cố định nghĩa là encoder ép toàn bộ chuỗi nguồn vào một vector duy nhất trước khi decoder thấy bất cứ thứ gì. Không quan trọng nguồn là 5 hay 500 token; nút cổ chai vẫn cùng một hình dạng.

Bài báo 2017 "Attention Is All You Need" đề xuất một điều triệt để: bỏ hoàn toàn recurrence (hồi quy). Cho mọi vị trí attention tới mọi vị trí khác, song song. Train trong một phép nhân ma trận lớn thay vì 1,024 phép tuần tự.

Kết quả thống trị mọi phương thức cho tới năm 2026. Ngôn ngữ (GPT-5, Claude 4, Llama 4), thị giác (ViT, DINOv2, SAM 3), âm thanh (Whisper), sinh học (AlphaFold 3), robot (RT-2). Cùng một khối, đầu vào khác nhau.

## Khái niệm

![Tính toán tuần tự của RNN so với attention song song của Transformer](../assets/rnn-vs-transformer.svg)

**Recurrence như một nút cổ chai.** Một RNN tính `h_t = f(h_{t-1}, x_t)`. Mỗi bước phụ thuộc vào bước trước. Bạn không thể tính `h_5` trước `h_4`. Trên GPU hiện đại với hơn 10,000 lõi song song, điều này lãng phí 99% silicon trên một chuỗi dài.

**Attention như một phép phát sóng.** Self-attention tính `output_i = sum_j(a_ij * v_j)` cho mọi cặp `(i, j)` cùng lúc. Toàn bộ ma trận attention N×N được điền trong một phép matmul gộp lô. Không bước nào phụ thuộc bước khác. GPU rất thích điều này.

**Tốc độ tăng không phải một hằng số.** Đó là khác biệt giữa độ sâu tuần tự `O(N)` và độ sâu tuần tự `O(1)`. Trong thực tế, transformer train nhanh hơn 5–10× mỗi epoch trên phần cứng tương đương ở N=512, và khoảng cách nới rộng theo độ dài chuỗi cho tới khi bạn chạm bức tường bộ nhớ `O(N²)` của attention (mà Flash Attention sau này đã xử lý — xem Bài 12).

**Cái giá của transformer.** Bộ nhớ attention tăng theo `O(N²)`. Với context 2K thì ổn. Với context 128K, bạn cần cửa sổ trượt, ngoại suy RoPE, chia khối Flash Attention, hoặc các biến thể linear attention. Recurrence từng là `O(N)` cả về thời gian lẫn bộ nhớ; transformer đổi thời gian lấy bộ nhớ rồi thắng lại thời gian nhờ song song hoá.

**Sự dịch chuyển thiên lệch quy nạp (inductive bias).** RNN giả định tính cục bộ và tính gần đây. Transformer không giả định gì — mọi cặp đều là ứng viên cho attention. Đó là lý do transformer cần nhiều dữ liệu hơn để train tốt nhưng mở rộng xa hơn khi đã có dữ liệu. Chinchilla (2022) hình thức hoá điều này: với đủ token, transformer luôn thắng một RNN có cùng số tham số.

```figure
rnn-vs-parallel
```

## Tự tay làm

Không có mạng nơ-ron ở đây — chúng ta mô phỏng nút cổ chai cốt lõi bằng số để bạn cảm nhận khoảng cách ngay trên laptop.

### Bước 1: đo độ sâu tuần tự

Xem `code/main.py`. Chúng ta xây hai hàm. Một hàm mã hoá chuỗi thành một chuỗi phép cộng (tuần tự, như RNN). Một hàm mã hoá nó thành phép rút gọn song song (phát sóng, như attention). Cùng phép toán, khác đồ thị phụ thuộc.

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

Chúng ta đo thời gian cả hai trên các chuỗi tới 100,000 phần tử. Bản RNN là O(N) và chỉ chạy trên một đường ống CPU. Ngay cả trong Python thuần, phép rút gọn kiểu attention vẫn thắng ở độ dài ≥ 1,000 vì `sum()` của Python được cài đặt bằng C và lặp mà không tốn chi phí interpreter mỗi bước.

### Bước 2: đếm số phép toán lý thuyết

Cả hai thuật toán đều làm N phép cộng. Khác biệt là *độ sâu phụ thuộc*: bao nhiêu phép phải xảy ra tuần tự trước khi phép tiếp theo có thể bắt đầu. Độ sâu RNN = N. Độ sâu attention = log(N) với rút gọn dạng cây, hoặc 1 với parallel scan. Độ sâu, không phải số phép, mới quyết định thời gian GPU.

### Bước 3: scaling thực nghiệm trên chuỗi dài

Chúng ta in một bảng thời gian làm khoảng cách O(N) hiện rõ. Trên một laptop Mac 2026, chuỗi dưới 1,000 phần tử quá nhanh để đo. Chuỗi 100,000 cho thấy một đường quét tuyến tính rõ ràng. Nhân lên thành một transformer 16,384 token với tương đương LSTM 12 lớp và bạn thấy vì sao thời gian train theo đồng hồ là rào cản năm 2016.

## Áp dụng

Khi nào vẫn nên chọn RNN vào năm 2026:

| Tình huống | Chọn |
|-----------|------|
| Suy luận streaming, từng token một, bộ nhớ không đổi | RNN hoặc state-space model (Mamba, RWKV) |
| Chuỗi rất dài (>1M token) nơi bộ nhớ attention bùng nổ | Linear attention, Mamba 2, Hyena |
| Thiết bị biên không có bộ tăng tốc matmul | RNN tách độ sâu vẫn thắng về FLOPs/watt |
| Mọi thứ khác (training, suy luận theo lô, context tới 128K) | Transformer |

State-space model (SSM) như Mamba thực chất là RNN với cách tham số hoá có cấu trúc, cho chúng điều tốt nhất của cả hai: bộ nhớ quét `O(N)`, train song song nhờ selective scan. Chúng lấy lại 90% chất lượng transformer với khả năng mở rộng long-context tốt hơn. Năm 2026, hầu hết phòng lab tiên phong train các mô hình lai SSM+transformer (ví dụ Jamba, Samba) — recurrence chưa chết, nó là một thành phần.

## Đóng gói

Xem `outputs/skill-architecture-picker.md`. Skill chọn kiến trúc cho một bài toán chuỗi mới dựa trên độ dài, thông lượng (throughput), và các ràng buộc ngân sách train. Nó luôn phải từ chối đề xuất RNN thuần cho các lượt train trên 1B token mà không nêu rõ đánh đổi.

## Bài tập

1. **Dễ.** Lấy `rnn_style` trong `code/main.py` và thay trạng thái ẩn vô hướng bằng một vector 64 trạng thái ẩn. Đo lại. Chi phí tuần tự tăng bao nhiêu khi số chiều trạng thái ẩn tăng?
2. **Trung bình.** Cài đặt parallel prefix-sum (Hillis-Steele scan) bằng Python thuần. Kiểm chứng nó cho cùng đầu ra số như scan tuần tự trên độ dài 1024. Đếm độ sâu.
3. **Khó.** Chuyển phép rút gọn kiểu attention sang PyTorch trên GPU. Đo thời gian cả hai khi quét độ dài chuỗi từ 64 đến 65,536. Vẽ và giải thích hình dạng đường cong.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Recurrence | "RNN là tuần tự" | Phép tính trong đó bước `t` phụ thuộc bước `t-1`, buộc thực thi tuần tự theo trục thời gian. |
| Serial depth | "Đồ thị sâu bao nhiêu" | Chuỗi phép phụ thuộc dài nhất; nó chặn thời gian thực ngay cả trên phần cứng vô hạn. |
| Attention | "Cho các token nhìn nhau" | Tổng có trọng số `sum_j a_ij v_j` trong đó `a_ij` đến từ điểm tương đồng giữa vị trí i và j. |
| Context window | "Mô hình thấy được bao nhiêu" | Số vị trí mà một lớp attention nhận làm đầu vào; chi phí bộ nhớ bậc hai nằm ở đây. |
| Inductive bias | "Giả định gắn sẵn trong kiến trúc" | Tiên nghiệm về hình dạng của dữ liệu; CNN giả định bất biến tịnh tiến, RNN giả định tính gần đây. |
| State-space model | "RNN có đại số đằng sau" | Recurrence được tham số hoá để train song song qua các ma trận state-space có cấu trúc. |
| Quadratic bottleneck | "Vì sao context đắt thế" | Bộ nhớ attention = `O(N²)` theo độ dài chuỗi; Flash Attention giấu hằng số, không giấu bậc tăng. |

## Đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — bài báo đã giết recurrence trong NLP chính thống.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — nơi attention ra đời, được gắn lên một RNN.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — bài báo LSTM gốc, ghi lại để tham chiếu.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — câu trả lời hồi quy hiện đại cho transformer.

---

[Bài sau: Self-attention từ đầu →](../../02-self-attention-from-scratch/docs/vi.md)
