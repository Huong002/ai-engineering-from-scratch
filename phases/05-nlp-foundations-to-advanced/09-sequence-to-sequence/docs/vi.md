# Mô hình sequence-to-sequence (chuỗi sang chuỗi)

> Hai mạng RNN đóng vai một cỗ máy dịch. Nút thắt cổ chai mà chúng gặp phải chính là lý do attention ra đời.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 08 (CNN + RNN cho văn bản), Giai đoạn 3, Bài 11 (Giới thiệu PyTorch)
**Time:** ~75 phút

## Vấn đề

Phân loại ánh xạ một chuỗi có độ dài thay đổi thành một nhãn duy nhất. Dịch thuật ánh xạ một chuỗi có độ dài thay đổi thành một chuỗi có độ dài thay đổi khác. Đầu vào và đầu ra nằm trong hai từ vựng khác nhau, có thể là hai ngôn ngữ khác nhau, và không có gì bảo đảm độ dài tương ứng.

Kiến trúc seq2seq (Sutskever, Vinyals, Le, 2014) giải quyết bài toán này bằng một công thức cố tình đơn giản. Hai mạng RNN. Một mạng đọc câu nguồn và tạo ra một vector ngữ cảnh có kích thước cố định. Mạng còn lại đọc vector đó và sinh câu đích theo từng token. Vẫn là đoạn code bạn viết ở bài 08, chỉ ghép lại theo cách khác.

Bài này đáng học vì hai lý do. Thứ nhất, nút thắt cổ chai của vector ngữ cảnh là thất bại có giá trị sư phạm bậc nhất trong NLP: nó là động lực cho mọi thứ mà attention và transformer làm tốt. Thứ hai, công thức huấn luyện — teacher forcing (buộc dùng token đúng), scheduled sampling, beam search (tìm kiếm chùm) khi suy luận — vẫn còn đúng với mọi hệ sinh văn bản hiện đại, kể cả LLM.

## Khái niệm

**Encoder (bộ mã hoá).** Một mạng RNN đọc câu nguồn. Trạng thái ẩn cuối cùng của nó là **vector ngữ cảnh** — một bản tóm tắt kích thước cố định của toàn bộ đầu vào. Người ta cho rằng nó không làm mất mát gì của câu nguồn.

**Decoder (bộ giải mã).** Một mạng RNN khác, được khởi tạo từ vector ngữ cảnh. Ở mỗi bước, nó nhận token vừa sinh ra trước đó làm đầu vào và tạo ra một phân phối trên từ vựng đích. Lấy mẫu hoặc argmax để chọn token kế tiếp. Đưa token đó trở lại đầu vào. Lặp lại cho tới khi sinh ra token `<EOS>` hoặc chạm độ dài tối đa.

**Training:** Hàm mất mát cross-entropy ở mỗi bước của decoder, rồi cộng dồn trên toàn chuỗi. Lan truyền ngược qua thời gian tiêu chuẩn qua cả hai mạng.

**Teacher forcing.** Khi huấn luyện, đầu vào của decoder ở bước `t` là token *đúng thực tế* ở vị trí `t-1`, không phải dự đoán trước đó của chính decoder. Điều này giúp huấn luyện ổn định; nếu không có nó, những sai lầm ban đầu sẽ lan thành hiệu ứng domino và mô hình không bao giờ học được. Khi suy luận, bạn buộc phải dùng dự đoán của chính mô hình, nên luôn tồn tại một khoảng cách phân phối giữa huấn luyện và suy luận. Khoảng cách đó gọi là **exposure bias** (độ lệch do mô hình chỉ quen với dữ liệu đúng).

**Nút thắt cổ chai.** Mọi thứ encoder học được về câu nguồn đều phải nhồi vào đúng một vector ngữ cảnh đó. Câu dài mất chi tiết. Từ hiếm bị nhoè. Việc đảo trật tự (chat noir so với black cat) phải được ghi nhớ chứ không thể tính ra.

Attention (bài 10) khắc phục điều này bằng cách cho decoder nhìn vào *mọi* trạng thái ẩn của encoder, không chỉ trạng thái cuối. Toàn bộ ý tưởng chỉ có vậy.

```figure
lstm-gates
```

## Tự tay làm

### Bước 1: bộ mã hoá

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` có kích thước `[batch, seq_len, hidden_dim]` — một trạng thái ẩn cho mỗi vị trí đầu vào. `hidden` có kích thước `[1, batch, hidden_dim]` — bước cuối cùng. Bài 08 nói "pool trên outputs để phân loại". Ở đây ta giữ trạng thái ẩn cuối cùng làm vector ngữ cảnh và bỏ qua các output từng bước.

### Bước 2: bộ giải mã

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Decoder được gọi từng bước một. Đầu vào: một batch gồm các token đơn và trạng thái ẩn hiện tại. Đầu ra: logits trên từ vựng cho token kế tiếp và trạng thái ẩn đã cập nhật.

### Bước 3: vòng lặp huấn luyện với teacher forcing

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

Có hai nút cần gọi tên. `ignore_index=0` bỏ qua mất mát trên các token đệm. `teacher_forcing_ratio` là xác suất dùng token đúng thay vì dự đoán của mô hình ở mỗi bước. Hãy bắt đầu ở 1.0 (teacher forcing hoàn toàn) rồi giảm dần về khoảng 0.5 trong quá trình huấn luyện để thu hẹp khoảng cách do exposure bias.

### Bước 4: vòng lặp suy luận (greedy)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

Giải mã greedy chọn token có xác suất cao nhất ở mọi bước. Nó có thể đi lạc: một khi đã chốt một token, bạn không thể rút lại. **Beam search** giữ lại top-`k` chuỗi chưa hoàn chỉnh và cuối cùng chọn chuỗi hoàn chỉnh có điểm cao nhất. Độ rộng chùm 3-5 là tiêu chuẩn.

### Bước 5: nút thắt cổ chai, minh hoạ cụ thể

Huấn luyện mô hình trên một bài tập chép đồ chơi: nguồn `[a, b, c, d, e]`, đích `[a, b, c, d, e]`. Tăng dần độ dài chuỗi. Quan sát độ chính xác.

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

Một trạng thái ẩn GRU duy nhất không thể ghi nhớ nguyên vẹn một đầu vào 40 token. Thông tin vẫn có ở mọi bước của encoder, nhưng decoder chỉ thấy trạng thái cuối. Attention khắc phục trực tiếp điểm này.

## Áp dụng

PyTorch có các mẫu seq2seq dựa trên `nn.Transformer` và `nn.LSTM`. Thư viện `transformers` của Hugging Face cung cấp sẵn các mô hình encoder-decoder đầy đủ (BART, T5, mBART, NLLB) đã được huấn luyện trên hàng tỷ token.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

Các mô hình encoder-decoder hiện đại đã bỏ RNN để dùng transformer. Hình dạng tổng thể (encoder, decoder, sinh token từng bước) vẫn giống hệt bài báo seq2seq năm 2014. Chỉ cơ chế bên trong mỗi khối là khác.

### Khi nào vẫn nên dùng seq2seq dựa trên RNN

Gần như không bao giờ, với dự án mới. Vài ngoại lệ cụ thể:

- Dịch luồng (streaming) khi bạn tiêu thụ đầu vào từng token một với bộ nhớ có hạn.
- Sinh văn bản trên thiết bị, nơi chi phí bộ nhớ của transformer là quá cao.
- Mục đích sư phạm. Hiểu nút thắt cổ chai của encoder-decoder là con đường nhanh nhất để hiểu vì sao transformer thắng.

### Exposure bias và các cách giảm thiểu

- **Scheduled sampling.** Giảm dần tỉ lệ teacher forcing trong quá trình huấn luyện để mô hình học cách tự sửa sai lầm của chính nó.
- **Minimum risk training.** Huấn luyện trên điểm BLEU cấp câu thay vì cross-entropy cấp token. Gần với điều bạn thực sự muốn hơn.
- **Tinh chỉnh bằng học tăng cường.** Thưởng cho bộ sinh chuỗi bằng một độ đo. Được dùng trong RLHF của LLM hiện đại.

Cả ba cách vẫn áp dụng được cho sinh văn bản dựa trên transformer.

## Đóng gói

Lưu thành `outputs/prompt-seq2seq-design.md`:

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## Bài tập

1. **Dễ.** Cài đặt bài tập chép đồ chơi. Huấn luyện một seq2seq GRU trên các cặp đầu vào - đầu ra trong đó đích bằng nguồn. Đo độ chính xác ở độ dài 5, 10, 20. Tái tạo lại nút thắt cổ chai.
2. **Trung bình.** Thêm giải mã beam search với độ rộng chùm 3. Đo BLEU trên một kho ngữ liệu song song nhỏ, so với greedy. Ghi lại chỗ beam search thắng (thường là những token cuối) và chỗ nó không tạo khác biệt.
3. **Khó.** Tinh chỉnh `facebook/bart-base` trên một tập dữ liệu diễn giải 10k cặp. So sánh đầu ra beam-4 của mô hình đã tinh chỉnh với mô hình gốc trên các đầu vào giữ lại để kiểm tra. Báo cáo BLEU và chọn ra 10 ví dụ định tính.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Encoder | RNN đầu vào | Đọc câu nguồn. Tạo ra trạng thái ẩn từng bước và một vector ngữ cảnh cuối cùng. |
| Decoder | RNN đầu ra | Được khởi tạo từ vector ngữ cảnh. Sinh token đích lần lượt từng cái một. |
| Context vector | Bản tóm tắt | Trạng thái ẩn cuối cùng của encoder. Kích thước cố định. Chính là nút thắt cổ chai mà attention giải quyết. |
| Teacher forcing | Dùng token đúng | Đưa token đúng thực tế của bước trước vào khi huấn luyện. Giúp việc học ổn định. |
| Exposure bias | Khoảng cách train/test | Mô hình huấn luyện trên token đúng nên chưa từng luyện cách gượng dậy sau sai lầm của chính mình. |
| Beam search | Giải mã tốt hơn | Giữ top-k chuỗi chưa hoàn chỉnh ở mỗi bước thay vì chốt ngay theo kiểu greedy. |

## Đọc thêm

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — bài báo seq2seq gốc. Bốn trang.
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — giới thiệu GRU và cách đóng khung encoder-decoder.
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — bài báo attention. Đọc ngay sau bài này.
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — code seq2seq + attention chạy được.

---

[← Bài trước: CNNs và RNN cho Văn bản](../../08-cnns-rnns-for-text/docs/vi.md) · [Bài sau: Cơ chế chú ý (attention) →](../../10-attention-mechanism/docs/vi.md)
