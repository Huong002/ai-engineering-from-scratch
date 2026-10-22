# CNNs và RNN cho Văn bản

> Tích chập học n-gram. Hồi quy ghi nhớ. Cả hai đều bị attention vượt qua. Cả hai vẫn còn quan trọng trên phần cứng hạn chế.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 3, Bài 11 (Giới thiệu PyTorch), Giai đoạn 5, Bài 03 (Word Embeddings), Giai đoạn 4, Bài 02 (Tích chập from scratch)
**Time:** ~75 phút

## Vấn đề

TF-IDF và Word2Vec tạo ra các vector phẳng, bỏ qua thứ tự từ. Một bộ phân loại dựng trên chúng không phân biệt được `dog bites man` với `man bites dog`. Thứ tự từ đôi khi mang tín hiệu.

Hai họ kiến trúc đã lấp khoảng trống đó trước khi transformer xuất hiện.

**Mạng tích chập cho văn bản (TextCNN).** Áp tích chập 1D lên chuỗi embedding từ. Một convolutional filter (bộ lọc tích chập) rộng 3 là bộ dò trigram học được: nó trải qua ba từ và cho ra một điểm. Xếp chồng nhiều độ rộng khác nhau (2, 3, 4, 5) để bắt các mẫu đa tỉ lệ. Max-pool (gộp cực đại) về một biểu diễn kích thước cố định. Phẳng, song song, nhanh.

**Mạng recurrent (hồi quy) — RNN, LSTM, GRU.** Xử lý từng token một, duy trì một hidden state (trạng thái ẩn) mang thông tin về phía trước. Tuần tự, có bộ nhớ, linh hoạt về độ dài chuỗi (sequence length). Thống trị mô hình hóa chuỗi từ 2014 đến 2017, rồi attention xuất hiện.

Bài này tự tay xây cả hai, rồi gọi tên thất bại đã thúc đẩy attention.

## Khái niệm

**TextCNN** (Kim, 2014). Token được embedding. Một tích chập 1D có độ rộng hạt nhân (kernel) `k` trượt bộ lọc qua các `k`-gram embedding liên tiếp, tạo ra một feature map. Global max-pooling trên map đó chọn activation mạnh nhất. Nối các output đã max-pool từ vài độ rộng bộ lọc. Đưa vào đầu phân loại.

Vì sao nó hiệu quả. Một bộ lọc là n-gram học được. Max-pooling bất biến vị trí, nên "not good" kích hoạt cùng một đặc trưng ở đầu hay giữa bài đánh giá. Ba độ rộng bộ lọc, mỗi độ rộng 100 bộ lọc, cho bạn 300 bộ dò n-gram học được. Huấn luyện song song; không có phụ thuộc tuần tự.

**RNN.** Ở mỗi bước thời gian `t`, hidden state `h_t = f(W * x_t + U * h_{t-1} + b)`. Chia sẻ `W`, `U`, `b` theo thời gian. Hidden state ở thời điểm `T` là bản tóm tắt toàn bộ tiền tố. Với phân loại, gộp qua `h_1 ... h_T` (max, mean, hoặc last).

RNN thuần chịu vanishing gradient (tiêu biến gradient). **LSTM** thêm các cổng quyết định quên gì, lưu gì, và xuất gì, nhờ đó ổn định gradient qua chuỗi dài. **GRU** đơn giản hóa LSTM thành hai cổng; hiệu năng tương đương với ít tham số hơn.

**RNN hai chiều** chạy một RNN xuôi và một RNN ngược, nối các hidden state lại. Biểu diễn của mỗi token thấy cả ngữ cảnh bên trái và bên phải. Thiết yếu cho các bài toán gán nhãn.

```figure
rnn-unroll
```

## Tự tay làm

### Bước 1: TextCNN trong PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` đổi hình dạng `[batch, seq_len, embed_dim]` thành `[batch, embed_dim, seq_len]` vì `nn.Conv1d` coi trục giữa là các kênh. Output sau gộp có kích thước cố định bất kể độ dài đầu vào.

### Bước 2: Bộ phân loại LSTM

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

Max-pool trên cả chuỗi, không phải pool trạng thái cuối. Với phân loại, max-pooling thường thắng việc lấy hidden state cuối, vì thông tin ở cuối một chuỗi dài có xu hướng lấn át trạng thái cuối.

### Bước 3: bản demo vanishing gradient (trực giác)

Một RNN thuần không có cổng không học được các phụ thuộc xa. Xét một bài đồ chơi: dự đoán xem token `A` có xuất hiện ở bất kỳ đâu trong chuỗi hay không. Nếu `A` ở vị trí 1 và chuỗi dài 100 token, gradient từ loss phải chảy ngược qua 99 lần nhân trọng số hồi quy. Nếu trọng số nhỏ hơn 1, gradient tiêu biến. Nếu lớn hơn 1, nó bùng nổ.

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM khắc phục điều này bằng một **cell state** chạy xuyên mạng chỉ với các tương tác cộng (cổng quên nhân tỉ lệ nó, nhưng gradient vẫn chảy theo "đường cao tốc"). GRU làm điều tương tự với ít tham số hơn. Cả hai cho bạn huấn luyện ổn định qua chuỗi 100+ bước.

### Bước 4: vì sao vẫn chưa đủ

Ba vấn đề vẫn tồn tại kể cả khi đã có LSTM.

1. **Nút cổ chai tuần tự.** Huấn luyện một RNN trên chuỗi dài 1000 cần 1000 bước xuôi/ngược tuần tự. Không thể song song hóa theo thời gian.
2. **Vector ngữ cảnh kích thước cố định trong thiết lập encoder-decoder.** Decoder chỉ thấy hidden state cuối cùng của encoder, đã nén trên toàn bộ đầu vào. Đầu vào dài mất chi tiết. Bài 09 nói trực tiếp về điều này.
3. **Trần độ chính xác với phụ thuộc xa.** LSTM vượt RNN thuần nhưng vẫn chật vật truyền thông tin cụ thể qua 200+ bước.

Attention giải quyết cả ba. Transformer bỏ hẳn recurrence. Bài 10 là bước ngoặt.

## Áp dụng

`nn.LSTM`, `nn.GRU`, và `nn.Conv1d` của PyTorch đã sẵn sàng cho production. Code huấn luyện là chuẩn mực.

Hugging Face cung cấp embedding tiền huấn luyện để bạn cắm vào làm lớp đầu vào:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

Danh sách kiểm tra dùng-khi-phù-hợp-ràng-buộc.

- **Suy luận trên thiết bị biên.** TextCNN với embedding GloVe nhỏ hơn transformer 10-100 lần. Nếu đích triển khai là điện thoại, đây là stack.
- **Phân loại streaming / trực tuyến.** RNN xử lý từng token một; transformer cần toàn bộ chuỗi. Với văn bản đến theo thời gian thực, LSTM vẫn thắng.
- **Mô hình tí hon làm baseline.** Lặp nhanh trên một bài mới. Huấn luyện một TextCNN trong 5 phút trên CPU.
- **Gán nhãn chuỗi với ít dữ liệu.** BiLSTM-CRF (bài 06) vẫn là kiến trúc NER chuẩn production cho 1k-10k câu gán nhãn.

Mọi thứ còn lại đều đi về transformer.

## Đóng gói

Lưu thành `outputs/prompt-text-encoder-picker.md`:

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## Bài tập

1. **Dễ.** Huấn luyện một TextCNN trên tập dữ liệu đồ chơi 3 lớp (bạn tự tạo dữ liệu). Kiểm chứng rằng các độ rộng bộ lọc (2, 3, 4) cho F1 trung bình tốt hơn một độ rộng duy nhất (3).
2. **Trung bình.** Cài đặt max-pool, mean-pool, và last-state pooling cho bộ phân loại LSTM. So sánh trên một tập nhỏ; ghi lại cách gộp nào thắng và đặt giả thuyết vì sao.
3. **Khó.** Xây một bộ gán nhãn NER BiLSTM-CRF (kết hợp bài 06 và bài này). Huấn luyện trên CoNLL-2003. So sánh với baseline chỉ dùng CRF ở bài 06 và với một BERT fine-tune. Báo cáo thời gian huấn luyện, bộ nhớ, và F1.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| TextCNN | CNN for text | Chồng các tích chập 1D trên embedding từ với global max-pool. Kim (2014). |
| RNN | Recurrent net | Hidden state được cập nhật ở mỗi bước thời gian: `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | Gated RNN | Thêm các cổng input / forget / output + một cell state. Huấn luyện ổn định qua chuỗi dài. |
| GRU | Simpler LSTM | Hai cổng thay vì ba. Độ chính xác tương đương, ít tham số hơn. |
| Bidirectional | Both directions | RNN xuôi + ngược nối lại. Mỗi token thấy cả hai phía ngữ cảnh của nó. |
| Vanishing gradient | Training signal dies | Nhân lặp với trọng số <1 trong RNN thuần khiến gradient ở bước đầu gần như bằng không. |

## Đọc thêm

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — bài báo TextCNN. Tám trang. Dễ đọc.
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — bài báo LSTM. Rõ ràng đến bất ngờ.
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — những hình vẽ đã giúp LSTM trở nên dễ tiếp cận với mọi người.

---

[← Bài trước: Gán nhãn từ loại và phân tích cú pháp](../../07-pos-tagging-parsing/docs/vi.md) · [Bài sau: Mô hình sequence-to-sequence (chuỗi sang chuỗi) →](../../09-sequence-to-sequence/docs/vi.md)
