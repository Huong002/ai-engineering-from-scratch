# Song song DualPipe

> DeepSeek-V3 được huấn luyện trên 2,048 GPU H800 với các chuyên gia MoE rải khắp các node. Truyền thông all-to-all chuyên gia xuyên node tốn 1 GPU-giờ truyền thông cho mỗi 1 GPU-giờ tính toán. GPU nhàn rỗi một nửa thời gian. DualPipe (DeepSeek, tháng 12/2024) là một pipeline hai chiều, chồng lấn phần tính toán forward và backward với chính các trao đổi all-to-all mà chúng kích hoạt. Bubble (khoảng trống chờ) giảm, throughput (thông lượng) tăng, và việc giữ hai bản sao tham số mô hình ("dual" trong tên gọi) là rẻ một khi Expert Parallelism (song song chuyên gia) vốn đã rải chuyên gia khắp các rank. Bài này là một walkthrough kiểu Learn về việc DualPipe thực sự làm gì và vì sao bản cải tiến DualPipeV của Sea AI Lab bỏ được chi phí tham số gấp 2, đổi lại khoảng trống chờ chỉ chặt hơn không đáng kể.

**Type:** Learn
**Languages:** Python (stdlib, schedule simulator)
**Prerequisites:** Giai đoạn 10, Bài 05 (huấn luyện phân tán, FSDP, DeepSpeed), Giai đoạn 10, Bài 14 (kiến trúc mô hình mở và MoE)
**Time:** ~60 phút

## Mục tiêu học tập

- Kể tên bốn thành phần của một chunk (khối) forward-backward trong DualPipe và giải thích vì sao mỗi thành phần có cửa sổ overlap (chồng lấn) riêng.
- Giải thích vấn đề bubble ở quy mô lớn, và "bubble-free" nghĩa là gì trên thực tế so với trên marketing.
- Lần theo một lịch DualPipe bằng tay với 8 rank PP và 16 micro-batch, rồi xác nhận luồng forward và luồng ngược lấp đầy các khe nhàn rỗi của nhau.
- Nêu đánh đổi của DualPipeV (Sea AI Lab, 2025): bỏ được việc nhân đôi tham số, đổi lại bubble lớn hơn một chút khi Expert Parallelism không hoạt động.

## Vấn đề

Huấn luyện một mô hình MoE 671B trên 2k GPU H800 gặp ba nút thắt cộng dồn:

1. **Áp lực bộ nhớ.** Mỗi GPU giữ một phần của mô hình. Bộ nhớ activation với độ dài chuỗi 8k, qua 61 lớp, trên 128 head là rất lớn.
2. **Pipeline bubble.** Pipeline parallelism (song song pipeline) truyền thống (GPipe, 1F1B) để GPU nhàn rỗi trong lúc chờ đầu vào hoặc gradient của stage mình. Với 8 stage, khoảng 12% thời gian GPU có thể là bubble ngay cả khi lập lịch 1F1B.
3. **All-to-all xuyên node.** MoE với Expert Parallelism rải chuyên gia khắp các node. Mỗi lượt forward kích hoạt một all-to-all để dispatch token tới chuyên gia của chúng, và một all-to-all nữa để combine. Với 2k GPU, tỷ lệ tính toán trên truyền thông dễ dàng thành 1:1.

Mỗi vấn đề đều có giải pháp riêng: gradient checkpointing cho bộ nhớ, Zero Bubble (Sea AI Lab, 2023) cho pipeline bubble, kernel truyền thông expert-parallel cho all-to-all. DualPipe làm việc khác: khiến chúng phối hợp với nhau. Lịch của nó chồng lấn tính toán và truyền thông trong cùng một chunk forward-backward, bơm micro-batch từ cả hai đầu pipeline cùng lúc, và dùng lịch thu được để giấu all-to-all bên trong các cửa sổ tính toán.

Kết quả được báo cáo: gần như xoá sạch pipeline bubble, hiệu suất GPU trên 95% trong lượt huấn luyện 14.8T token của DeepSeek-V3.

## Khái niệm

### Nhắc lại về pipeline parallelism

Chia một mô hình N lớp ra P thiết bị. Thiết bị `i` giữ các lớp `i * N/P .. (i+1) * N/P - 1`. Một micro-batch chảy forward qua các thiết bị 0 đến P-1, rồi backward từ P-1 về 0. Mỗi thiết bị chỉ có thể bắt đầu stage forward khi thiết bị trước gửi đầu ra, và chỉ có thể bắt đầu backward khi thiết bị phía sau gửi gradient ngược lên.

GPipe (Huang et al., 2019) lập lịch từng micro-batch một, cách này lãng phí phần lớn thời gian GPU. 1F1B (Narayanan et al., 2021) xen kẽ các lượt forward và backward cho nhiều micro-batch. Zero Bubble (Qi et al., 2023) tách lượt backward thành hai phần — backward-for-input (B) và backward-for-weights (W) — rồi lập lịch chúng để lấp bubble. Sau Zero Bubble, pipeline gần như kín.

DualPipe là bước tiếp theo. Nó thêm hai ý tưởng:

### Ý tưởng 1: phân rã chunk

Mỗi chunk forward được tách thành bốn thành phần:

- **Attention.** Chiếu Q/K/V, attention, chiếu đầu ra.
- **All-to-all dispatch.** Truyền thông xuyên node gửi token tới các chuyên gia tương ứng.
- **MLP.** Phần tính toán chuyên gia MoE.
- **All-to-all combine.** Truyền thông xuyên node mang đầu ra của chuyên gia về.

Một chunk backward thêm phiên bản gradient của từng thành phần trên. DualPipe lập lịch để all-to-all dispatch chạy song song với tính toán attention của chunk kế tiếp, và all-to-all combine chạy song song với tính toán MLP của chunk sau đó.

### Ý tưởng 2: lập lịch hai chiều

Hầu hết lịch pipeline bơm micro-batch từ stage 0 và chảy về phía stage P-1. DualPipe bơm micro-batch từ CẢ HAI đầu. Stage 0 thấy các micro-batch forward bắt đầu từ đó; stage P-1 cũng thấy các micro-batch forward bắt đầu từ đó. Hai luồng gặp nhau ở giữa.

Để làm được vậy, thiết bị `i` phải giữ CẢ lớp `i` ở đầu pipeline VÀ lớp `P - 1 - i` ở cuối pipeline. Đó chính là phần "dual" của DualPipe: mỗi thiết bị giữ hai bản sao của các lớp mô hình mà nó phục vụ (một cho mỗi chiều). Ở quy mô DeepSeek-V3, đây là chi phí nhân đôi tham số. Nó chấp nhận được vì Expert Parallelism vốn đã rải chuyên gia MoE mỏng đến mức nhân đôi các lớp không phải chuyên gia chỉ là chuyện nhỏ.

Mấu chốt là luồng forward ở chiều này và luồng backward ở chiều kia chồng lấn đúng vào chỗ mà lịch một chiều sẽ có bubble. Bubble biến mất.

### Một lịch được lần bằng tay

Xét P = 4 rank, 8 micro-batch, chia 4 forward / 4 ngược. Thời gian chạy từ trái sang phải; mỗi hàng là một rank thiết bị.

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

Đọc ký hiệu "F4/F5R": ở cùng một khe thời gian, rank 1 đang chạy forward của micro-batch 4 (theo chiều trái sang phải trong pipeline) VÀ forward của micro-batch 5 (theo chiều phải sang trái). Đó là ý nghĩa vận hành của "bidirectional".

Ở rank 2 các luồng giao nhau sớm hơn, ở rank 0 và P-1 chúng giao muộn nhất. Trong pha giữa ổn định của lịch, mỗi rank chạy forward theo chiều X chồng lấn với backward theo chiều Y. Tính toán luôn bận. Các all-to-all dispatch cho lượt forward ẩn bên trong tính toán backward. Các all-to-all combine ẩn bên trong tính toán forward. Bubble bị ép ra ngoài.

### Hạch toán bubble

Bubble của pipeline 1F1B tiêu chuẩn (thời gian lãng phí trên mỗi rank):

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Bản cải tiến Zero Bubble kéo nó xuống nhưng chưa về 0. Ở pha ổn định, DualPipe có bubble bằng 0 nếu số micro-batch chia hết cho 2 lần độ sâu pipeline. Ngoài pha ổn định (warmup và cooldown), vẫn có bubble nhưng nó không tăng theo số micro-batch — một tính chất then chốt mà bài báo nhấn mạnh.

Theo ngôn ngữ marketing: "bubble-free". Theo ngôn ngữ kỹ thuật: bubble không tăng theo số micro-batch. Phân tích tiếp theo của Sea AI Lab (DualPipeV / Cut-in-half) cho thấy chỉ có zero-bubble trọn vẹn khi Expert Parallelism không phải nút thắt; với all-to-all do EP dẫn dắt, luôn có một thoả hiệp nào đó trong lập lịch.

### DualPipeV — bản cải tiến

Sea AI Lab (2025) nhận thấy việc nhân đôi tham số là lãng phí khi mục tiêu không phải là chồng lấn truyền thông EP. Lịch DualPipeV của họ gói việc bơm hai chiều vào một lịch "hình chữ V" chỉ chạy trên một bản sao tham số. Bubble lớn hơn của DualPipe một chút, nhưng tiết kiệm bộ nhớ đáng kể. DeepSeek đã dùng DualPipeV trong bản DualPipe mã nguồn mở của mình như một chế độ EP-off.

Đánh đổi:

| Tính năng | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| Số bản sao tham số mỗi thiết bị | 2 | 1 | 1 | 1 |
| Bubble theo số micro-batch | không đổi | tăng nhẹ | tăng | tăng |
| Chồng lấn tính toán - truyền thông | đầy đủ | một phần | tối thiểu | một phần |
| Dùng khi | MoE nặng EP | mô hình dense hoặc EP nhẹ | mức cơ sở | mọi pipeline |

### Ý nghĩa với một lượt chạy 14.8T token

Tiền huấn luyện DeepSeek-V3 tiêu tốn 14.8T token trên 2,048 GPU H800, khoảng 2.8M GPU-giờ. Với 1F1B ngây thơ, họ sẽ mất 12-15% trong đó cho pipeline bubble — 340-420K GPU-giờ, đủ để huấn luyện một mô hình 70B đầy đủ. DualPipe lấy lại phần lớn khoản đó. Rất khó định lượng trực tiếp đóng góp này nếu không có log nội bộ, nhưng bài báo khẳng định hiệu suất GPU trung bình trên toàn quá trình huấn luyện là trên 95%.

Với các lượt chạy nhỏ hơn (dưới 1k GPU), DualPipe là thừa thãi — bubble nhỏ hơn tương đối so với tổng chi phí, và huấn luyện mô hình dense hiếm khi chạm nút thắt all-to-all. Với huấn luyện MoE tiên phong ở quy mô vài nghìn GPU, nó gần như bắt buộc.

### Vị trí trong chồng công nghệ

- Bổ trợ cho **FSDP** (Giai đoạn 10 · 05). FSDP chia nhỏ tham số mô hình ra các rank; DualPipe lập lịch tính toán trên các rank. Hai thứ kết hợp được.
- Tương thích với **ZeRO-3** chia nhỏ gradient. Việc quản lý sổ sách cho bản sao hai lớp phải phối hợp với gradient đã chia nhỏ của ZeRO.
- Cần **kernel all-to-all tuỳ biến** được tinh chỉnh cho topology cụm cụ thể. Các kernel mã nguồn mở của DeepSeek là bản tham chiếu.

```figure
expert-capacity
```

## Áp dụng

`code/main.py` là một trình mô phỏng lịch pipeline. Nó nhận `(P, n_micro_batches, schedule)` và in hiệu suất ở pha ổn định cho từng lịch 1F1B, Zero Bubble, DualPipe và DualPipeV. Đây là công cụ để dạy — các con số khớp với những khẳng định định tính trong bài báo, chúng không phải khẳng định về tốc độ đo được trên production.

Giá trị của trình mô phỏng: chạy nó với các giá trị P và số micro-batch khác nhau rồi xem phần bubble tăng lên thế nào với 1F1B mà không tăng với DualPipe.

Những điểm cần cân nhắc khi tích hợp vào một lượt huấn luyện thật:

- Chọn độ sâu pipeline-parallel chia hết cho số micro-batch của bạn.
- Đảm bảo lưới expert-parallel của bạn hỗ trợ all-to-all hai chiều. Kernel của DeepSeek là bản tham chiếu.
- Lần đầu, hãy tính trước một tuần gỡ lỗi cho chính phần lập lịch. Việc quản lý sổ sách khá rắc rối.
- Theo dõi hiệu suất GPU theo từng rank, không chỉ tổng hợp. Lợi ích của DualPipe đến từ việc siết các rank tụt lại.

## Đóng gói

Bài này tạo ra `outputs/skill-dualpipe-planner.md`. Cho trước đặc tả một cụm huấn luyện (số GPU, topology, mạng liên kết, hình dạng mô hình), nó đề xuất chiến lược pipeline parallelism, thuật toán lập lịch nên dùng, và phần bubble kỳ vọng ở quy mô mục tiêu.

## Bài tập

1. Chạy `code/main.py` với `(P=8, micro_batches=16, schedule=dualpipe)` và `(P=8, micro_batches=16, schedule=1f1b)`. Tính chênh lệch hiệu suất GPU và biểu diễn nó thành số GPU-giờ thu hồi được trên mỗi triệu token huấn luyện.

2. Phác thảo bảng lịch cho `(P=4, micro_batches=8, schedule=dualpipe)` bằng tay. Ghi vào mỗi khe thời gian ID micro-batch và chiều. Xác định khe thời gian đầu tiên không còn bubble.

3. Đọc Hình 5 trong báo cáo kỹ thuật DeepSeek-V3 (arXiv:2412.19437). Xác định cửa sổ chồng lấn cho all-to-all dispatch bên trong một chunk forward của DualPipe. Giải thích cách lịch tính toán giấu nó đi.

4. Tính chi phí phụ trội tham số gấp 2 của DualPipe cho một mô hình dense 70B với P=8 stage pipeline và một mô hình MoE 671B với P=16 stage pipeline. Chỉ ra vì sao chi phí phụ trội ở trường hợp MoE nhỏ hơn theo tỷ lệ (phần lớn tham số là chuyên gia, được chia nhỏ trên một nhóm EP lớn).

5. So sánh DualPipe với Chimera (một bộ lập lịch hai chiều cạnh tranh từ 2021). Dựa vào Mục 3.4 của bài báo, chỉ ra hai tính chất cụ thể mà DualPipe thêm vào còn Chimera không có.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Pipeline bubble | "Thời gian nhàn rỗi mỗi rank" | Chu kỳ GPU bị lãng phí vì một stage pipeline đang chờ đầu vào hoặc gradient |
| 1F1B | "Lịch pipeline mặc định" | Lập lịch xen kẽ một forward / một backward; mức cơ sở mà DualPipe vượt qua |
| Zero Bubble | "Sea AI Lab 2023" | Tách backward thành B (gradient đầu vào) và W (gradient trọng số); gần như làm kín hoàn toàn pipeline |
| DualPipe | "Lịch của DeepSeek-V3" | Pipeline hai chiều + chồng lấn tính toán - truyền thông; bubble không tăng theo số micro-batch |
| DualPipeV | "Cut-in-half" | Bản cải tiến hình chữ V bỏ việc nhân đôi tham số, đổi lại bubble lớn hơn một chút |
| Chunk | "Đơn vị công việc của pipeline" | Một lượt forward hoặc backward của một micro-batch qua một stage pipeline |
| All-to-all dispatch | "Gửi token tới chuyên gia" | Truyền thông xuyên node định tuyến token tới các chuyên gia MoE được gán |
| All-to-all combine | "Mang đầu ra chuyên gia về" | Truyền thông xuyên node gom đầu ra của chuyên gia sau MLP |
| Expert Parallelism (EP) | "Chuyên gia rải trên các GPU" | Chia nhỏ chuyên gia MoE ra các rank để các GPU khác nhau giữ chuyên gia khác nhau |
| Pipeline Parallelism (PP) | "Các lớp rải trên các GPU" | Chia nhỏ các lớp mô hình ra các rank; đây là chiều mà DualPipe lập lịch |
| Bubble fraction | "Thời gian GPU lãng phí" | (bubble_time / total_time); phần mà DualPipe đẩy về 0 |

## Đọc thêm

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Mục 3.3.2 và Hình 5](https://arxiv.org/abs/2412.19437) — tài liệu tham chiếu chính về DualPipe
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) — bản tham chiếu mã nguồn mở, gồm cả chế độ DualPipeV (Cut-in-half)
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — tiền thân Zero Bubble
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) — phân tích DualPipeV, cơ sở cho chế độ EP-off của DeepSeek
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — lịch 1F1B mà DualPipe so sánh với
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — bài báo gốc về pipeline parallelism và bài toán bubble

---

[← Bài trước: Dự đoán nhiều token (MTP)](../../18-multi-token-prediction/docs/vi.md) · [Bài sau: Điểm qua kiến trúc DeepSeek-V3 →](../../20-deepseek-v3-walkthrough/docs/vi.md)
