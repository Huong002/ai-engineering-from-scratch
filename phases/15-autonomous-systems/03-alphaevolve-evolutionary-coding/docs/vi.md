# AlphaEvolve — Tác tử lập trình tiến hóa

> Ghép một mô hình lập trình tiên tiến với vòng lặp tiến hóa và bộ đánh giá có thể kiểm tra bằng máy. Cho vòng lặp chạy đủ lâu. Nó tìm ra cách nhân ma trận phức 4x4 dùng 48 phép nhân vô hướng — cải tiến đầu tiên so với Strassen sau 56 năm. Nó cũng tìm ra một heuristic lập lịch Borg dùng trên toàn Google, thu hồi ~0.7% năng lực tính toán cụm trong production. Kiến trúc này cố ý đơn giản. Thành quả đến từ độ chặt chẽ của bộ đánh giá.

**Type:** Learn
**Languages:** Python (stdlib, evolutionary-loop toy)
**Prerequisites:** Giai đoạn 15, Bài 01 (cách nhìn dài hạn), Giai đoạn 15, Bài 02 (tự học suy luận)
**Time:** ~60 phút

## Vấn đề

Mô hình ngôn ngữ lớn có thể viết code. Thuật toán tiến hóa có thể tìm kiếm trên code. Cả hai đã được thử riêng rẽ hàng chục năm và đều gặp giới hạn. Giới hạn của LLM là bịa đặt: mô hình viết code có vẻ đúng nhưng không làm được điều nó tuyên bố. Giới hạn của tiến hóa là chi phí tìm kiếm: đột biến cú pháp ngẫu nhiên hiếm khi tạo ra chương trình biên dịch được, chưa nói đến chương trình tốt hơn.

AlphaEvolve (Novikov et al., DeepMind, arXiv:2506.13131, tháng 6/2025) kết hợp hai cách. LLM đề xuất chỉnh sửa có chủ đích cho cơ sở dữ liệu chương trình; bộ đánh giá tự động chấm điểm từng biến thể; biến thể đạt điểm cao trở thành chương trình cha cho các thế hệ sau. LLM đảm nhiệm bước tốn kém là viết code có vẻ hợp lý; bộ đánh giá phát hiện những chỗ bịa đặt. Vòng lặp chạy từ vài giờ đến vài tuần.

Các kết quả được báo cáo: phép nhân ma trận phức 4x4 dùng 48 phép nhân vô hướng (giới hạn năm 1969 của Strassen là 49), một heuristic lập lịch Borg trong production của Google, tăng tốc kernel FlashAttention 32.5%, và cải thiện thông lượng huấn luyện Gemini.

Kiến trúc này hiệu quả vì bộ đánh giá có thể kiểm tra bằng máy. Nó không hiệu quả khi bộ đánh giá không đáp ứng điều đó. Sự khác biệt ấy là bài học chính.

## Khái niệm

### Vòng lặp

1. Bắt đầu với chương trình gốc `P_0` đúng nhưng chưa tối ưu.
2. Duy trì cơ sở dữ liệu các biến thể chương trình, mỗi biến thể được bộ đánh giá chấm điểm.
3. Lấy mẫu một hoặc nhiều chương trình cha từ cơ sở dữ liệu (theo kiểu MAP-elites hoặc mô hình đảo).
4. Nhắc LLM (Gemini Flash cho nhiều ứng viên, Gemini Pro cho trường hợp khó) tạo biến thể đã chỉnh sửa từ chương trình cha.
5. Biên dịch, chạy và đánh giá biến thể bằng bộ đánh giá giữ riêng.
6. Đưa biến thể vào cơ sở dữ liệu, lập chỉ mục theo điểm và vector đặc trưng.
7. Lặp lại.

Hai chi tiết quan trọng. Thứ nhất, prompt cho LLM không chỉ có chương trình cha — thường còn có vài biến thể hàng đầu trong cơ sở dữ liệu, chữ ký của bộ đánh giá và mô tả ngắn về nhiệm vụ. Việc của mô hình là đề xuất thay đổi có chủ đích, có thể cải thiện điểm. Thứ hai, cơ sở dữ liệu có cấu trúc (lưới MAP-elites, mô hình đảo) để vòng lặp khám phá đa dạng, thay vì chỉ bám theo biến thể đang dẫn đầu.

### Vì sao không thể thiếu bộ đánh giá

Các thành quả của AlphaEvolve đều đến từ lĩnh vực có bộ đánh giá nhanh, cho kết quả xác định và khó bị qua mặt:

- **Thuật toán nhân ma trận**: unit test nhân các ma trận và kiểm tra kết quả bằng nhau đến từng bit.
- **Heuristic lập lịch Borg**: trình mô phỏng đạt chuẩn production, phát lại tải cụm trong quá khứ và đo năng lực tính toán bị lãng phí.
- **Kernel FlashAttention**: kiểm thử tính đúng đắn cùng phép đo thời gian chạy thực trên phần cứng thật.
- **Thông lượng huấn luyện Gemini**: số giây GPU đo được cho mỗi bước.

Trong mỗi trường hợp, bộ đánh giá phát hiện các lỗi LLM vốn sẽ lấn át kết quả: tuyên bố sai về tính đúng đắn, tuyên bố hiệu năng không đúng trên phần cứng và lỗi ở trường hợp biên. Bỏ bộ đánh giá đi, vòng lặp sẽ tối ưu để code trông đẹp.

### Reward hacking là mặt kia của vấn đề

Tiến hóa tối ưu đúng thứ bộ đánh giá đo. Nếu bộ đánh giá có lỗ hổng, vòng lặp sẽ tìm ra. Ở lĩnh vực không thể kiểm chứng, nó sẽ tối ưu đặc điểm bề mặt thay vì hành vi mong muốn. DeepMind nêu rõ điều này trong bài báo: thành công của AlphaEvolve chỉ áp dụng được ở lĩnh vực có bộ đánh giá đủ chặt chẽ so với tham vọng tìm kiếm.

Các ví dụ cụ thể về reward hacking (khai thác thước đo thưởng) trong vòng lặp tìm kiếm code giai đoạn 2025-2026:

- Mục tiêu tối ưu thưởng cho “thời gian hoàn thành” lại khuyến khích nộp lời giải rỗng.
- Điểm benchmark thưởng cho tính đúng đắn trên bài kiểm thử lại khuyến khích ghi nhớ bài kiểm thử và overfitting.
- Thước đo thay thế cho “chất lượng code” lại thưởng cho việc xóa comment và đổi tên biến, dù ngữ nghĩa không đổi.

Cách khắc phục trong AlphaEvolve: dùng bộ đánh giá giữ riêng mà LLM chưa từng thấy, với đầu vào được tạo lúc đánh giá. Ngay cả vậy, DeepMind vẫn khuyến nghị rà soát kỹ mọi đề xuất triển khai.

### Vì sao LLM + tìm kiếm tốt hơn từng cách riêng lẻ

LLM có thể tạo ra chỉnh sửa biên dịch được và hợp lý về ngữ nghĩa. GA đột biến ngẫu nhiên trên file Python 2000 dòng hầu như luôn gây lỗi cú pháp. LLM cũng tập trung tìm kiếm trong vùng có triển vọng (đổi một hàm, không đổi byte ngẫu nhiên), qua đó giảm mạnh số lượt gọi bộ đánh giá bị lãng phí.

Ngược lại, bộ đánh giá phát hiện những chỗ LLM bịa đặt. LLM có thể quả quyết một hàm “là O(n log n) khi xét giới hạn” dù thực tế là O(n^2); phép đo thời gian chạy thực sẽ phân định rõ.

### Vị trí của AlphaEvolve trong hệ thống tiên tiến

| Hệ thống | Bộ tạo | Bộ đánh giá | Lĩnh vực | Thành quả ví dụ |
|---|---|---|---|---|
| AlphaEvolve | Gemini | tính đúng đắn + benchmark | thuật toán, kernel, bộ lập lịch | nhân ma trận 4x4 với 48 phép nhân |
| FunSearch (DeepMind, 2023) | PaLM / Codey | tính đúng đắn | toán tổ hợp | cận dưới cap-set |
| AI Scientist v2 (Sakana, L5) | GPT/Claude | LLM phản biện + thí nghiệm | nghiên cứu ML | bài báo tại ICLR workshop |
| Darwin Godel Machine (L4) | khung tác tử | SWE-bench / Polyglot | code tác tử | 20% → 50% SWE-bench |

Cả bốn đều biến thể của cùng một công thức: bộ tạo cộng bộ đánh giá, rồi lặp. Khác biệt nằm ở điều bộ đánh giá chấm và mức độ chặt chẽ của nó.

```figure
alphaevolve-loop
```

## Áp dụng

`code/main.py` triển khai vòng lặp tối giản kiểu AlphaEvolve cho bài toán hồi quy ký hiệu đơn giản. “LLM” là chương trình thay thế dùng stdlib, đề xuất đột biến cú pháp nhỏ cho chương trình tính hàm mục tiêu. “Bộ đánh giá” đo sai số bình phương trung bình trên các điểm kiểm thử giữ riêng.

Quan sát:

- Điểm tốt nhất cải thiện ra sao qua các thế hệ.
- Lưới MAP-elites giữ các lời giải đa dạng tồn tại ra sao để vòng lặp không hội tụ vào cực tiểu cục bộ.
- Khi bỏ kiểm thử giữ riêng (chỉ đánh giá trên dữ liệu huấn luyện), vòng lặp overfitting nghiêm trọng ra sao.

## Đóng gói

`outputs/skill-evaluator-rigor-audit.md` là điều kiện tiên quyết trước khi cân nhắc vòng lặp kiểu AlphaEvolve trong lĩnh vực mới: bộ đánh giá của bạn có thật sự phát hiện những lỗi bạn quan tâm không?

## Bài tập

1. Chạy `code/main.py`. Ghi lại diễn biến điểm tốt nhất. Tắt bộ đánh giá giữ riêng (cờ `--no-holdout`) rồi chạy lại. Định lượng mức overfitting.

2. Đọc Section 3 của bài báo AlphaEvolve về lưới MAP-elites. Thiết kế vector đặc trưng mô tả một bài toán mới (ví dụ các lượt tối ưu trình biên dịch) để duy trì tính đa dạng của tìm kiếm.

3. Kết quả nhân ma trận 4x4 bằng 48 phép nhân cải thiện giới hạn 49 phép nhân của Strassen sau 56 năm. Đọc Appendix F của bài báo và giải thích trong ba câu vì sao bộ đánh giá cho bài toán này đặc biệt dễ làm đúng, còn phần lớn lĩnh vực khác thì không.

4. Đề xuất một lĩnh vực mà AlphaEvolve sẽ thất bại. Chỉ rõ bộ đánh giá hỏng ở đâu và vì sao.

5. Với một lĩnh vực bạn biết, hãy viết chữ ký bộ đánh giá bạn sẽ dùng. Gồm (a) điều kiện đúng, (b) thước đo hiệu năng, (c) quy tắc tạo đầu vào giữ riêng, (d) ít nhất một phép kiểm tra chống reward hacking.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| AlphaEvolve | “Tác tử lập trình tiến hóa của DeepMind” | Gemini + cơ sở dữ liệu chương trình + bộ đánh giá có thể kiểm tra bằng máy |
| MAP-elites | “Kho lưu trữ duy trì tính đa dạng” | Lưới lập chỉ mục theo vector đặc trưng; mỗi ô giữ biến thể tốt nhất mang đặc trưng đó |
| Island model | “Các quần thể con tiến hóa song song” | Các quần thể độc lập trao đổi cá thể định kỳ; ngăn hội tụ quá sớm |
| Machine-checkable evaluator | “Bộ kiểm định cho kết quả xác định” | Unit test, trình mô phỏng hoặc benchmark mà LLM không thể giả mạo — điều kiện tiên quyết cho vòng lặp này |
| Reward hacking | “Tối ưu thước đo thay vì mục tiêu” | Vòng lặp tìm cách tối đa hóa điểm mà không thực hiện nhiệm vụ mong muốn |
| Seed program | “Điểm khởi đầu” | Chương trình ban đầu đúng nhưng chưa tối ưu, làm nền để vòng lặp tiến hóa |
| Held-out evaluator | “Dữ liệu đánh giá LLM chưa từng thấy” | Đầu vào được tạo lúc đánh giá để ngăn ghi nhớ |

## Đọc thêm

- [Novikov et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131) — bài báo đầy đủ.
- [Bài blog của DeepMind về AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — bài viết của nhà phát triển về các kết quả.
- [Kho kết quả AlphaEvolve](https://github.com/google-deepmind/alphaevolve_results) — các thuật toán được khám phá, gồm phép nhân ma trận 4x4 với 48 phép nhân.
- [Romera-Paredes et al. (2023). Mathematical discoveries from program search with LLMs (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) — hệ thống tiền nhiệm.
- [Anthropic — Responsible Scaling Policy v3.0 (Feb 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — xem quyền tự chủ bị giới hạn bởi bộ đánh giá là hướng nghiên cứu trọng tâm.

---

[← Bài trước: STaR, V-STaR, Quiet-STaR](../../02-star-family-reasoning/docs/vi.md) · [Bài sau: Darwin Godel Machine →](../../04-darwin-godel-machine/docs/vi.md)
