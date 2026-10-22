# Norms và Khoảng cách

> Hàm distance (khoảng cách) của bạn định nghĩa "giống nhau" nghĩa là gì. Chọn sai là mọi thứ phía sau đều đổ vỡ.

**Type:** Build
**Language:** Python
**Prerequisites:** Giai đoạn 1, Bài 01 (Trực giác Đại số tuyến tính), 02 (Vector, Ma trận & Phép toán)
**Time:** ~90 phút

## Mục tiêu học tập

- Tự tay triển khai các hàm khoảng cách L1, L2, cosin, Mahalanobis, Jaccard và khoảng cách chỉnh sửa
- Chọn metric (metric) khoảng cách phù hợp cho một bài toán ML cụ thể và giải thích vì sao các phương án khác thất bại
- Nối chuẩn L1 và L2 với chuẩn hoá LASSO và Ridge cùng các miền ràng buộc hình học của chúng
- Chứng minh rằng cùng một tập dữ liệu cho ra những láng giềng gần nhất khác nhau dưới các metric khác nhau

## Vấn đề

Bạn có hai vector. Có thể chúng là embedding từ ngữ. Có thể chúng là hồ sơ người dùng. Có thể chúng là mảng điểm ảnh. Bạn cần biết: chúng gần nhau đến mức nào?

Câu trả lời phụ thuộc hoàn toàn vào việc bạn chọn hàm khoảng cách nào. Hai điểm dữ liệu có thể là láng giềng gần nhất dưới metric này và lại cách xa nhau dưới metric khác. Bộ phân loại KNN của bạn, engine gợi ý của bạn, vector database của bạn, thuật toán phân cụm của bạn, hàm mất mát của bạn -- tất cả đều phụ thuộc vào lựa chọn này. Chọn sai là model của bạn tối ưu cho một thứ không đúng.

Không có khoảng cách nào tốt nhất cho mọi trường hợp. L2 hiệu quả với dữ liệu không gian. Cosine similarity thống trị NLP. Jaccard xử lý tập hợp. Khoảng cách chỉnh sửa xử lý chuỗi. Mahalanobis tính đến tương quan. Wasserstein dịch chuyển khối xác suất. Mỗi loại mã hoá một giả định khác nhau về ý nghĩa của "giống nhau".

Bài học này tự tay xây dựng mọi hàm khoảng cách chính, chỉ cho bạn khi nào mỗi hàm là công cụ đúng, và chứng minh rằng cùng một dữ liệu cho ra những láng giềng gần nhất hoàn toàn khác nhau tuỳ metric bạn dùng.

## Khái niệm

### Chuẩn: đo độ lớn của vector

Một norm (chuẩn) đo "kích thước" của một vector. Mọi hàm khoảng cách giữa hai vector đều có thể viết dưới dạng chuẩn của hiệu chúng: d(a, b) = ||a - b||. Vậy hiểu chuẩn là hiểu khoảng cách.

### Chuẩn L1 (khoảng cách Manhattan)

Chuẩn L1 cộng giá trị tuyệt đối của tất cả các thành phần.

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

Nó được gọi là Manhattan distance (khoảng cách Manhattan) vì nó đo quãng đường bạn đi trên một lưới thành phố, nơi bạn chỉ được di chuyển dọc theo các trục. Không có đường chéo.

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

Khi nào dùng L1:
- Dữ liệu thưa ở số chiều cao (đặc trưng văn bản, mã hoá one-hot)
- Khi bạn muốn bền vững với outlier (một chênh lệch lớn duy nhất không chi phối được)
- Bài toán chọn đặc trưng (chuẩn hoá L1 thúc đẩy tính thưa)

Liên hệ với chuẩn hoá L1 (Lasso): thêm ||w||_1 vào hàm mất mát sẽ phạt tổng giá trị tuyệt đối của trọng số. Điều này đẩy các trọng số nhỏ về đúng zero, tức tự động chọn đặc trưng. Hình phạt L1 tạo ra miền ràng buộc hình thoi trong không gian trọng số, và các đỉnh của hình thoi nằm trên các trục, nơi một số trọng số bằng zero.

Liên hệ với hàm mất mát: Mean Absolute Error (MAE) là khoảng cách L1 trung bình giữa dự đoán và nhãn. Nó phạt mọi sai số theo tuyến tính, nên bền vững với outlier hơn MSE.

### Chuẩn L2 (khoảng cách Euclid)

Chuẩn L2 là khoảng cách đường thẳng — Euclidean distance (khoảng cách Euclid). Căn bậc hai của tổng bình phương các thành phần.

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

Đây là khoảng cách bạn học trong giờ hình học. Định lý Pythagoras trong n chiều.

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

Khi nào dùng L2:
- Dữ liệu liên tục có số chiều thấp đến trung bình
- Khi thang đo các đặc trưng tương đương nhau
- Khoảng cách vật lý (dữ liệu không gian, số đọc cảm biến)
- Độ tương đồng ảnh ở mức điểm ảnh

Liên hệ với chuẩn hoá L2 (Ridge): thêm ||w||_2^2 vào hàm mất mát sẽ phạt các trọng số lớn. Khác L1, nó không đẩy trọng số về zero. Nó thu nhỏ mọi trọng số về phía zero theo tỷ lệ. Hình phạt L2 tạo ra miền ràng buộc hình tròn, nên không có đỉnh nào trên các trục. Trọng số trở nên nhỏ nhưng hiếm khi đúng bằng zero.

Liên hệ với hàm mất mát: Mean Squared Error (MSE) là trung bình bình phương các khoảng cách L2. Bình phương phạt sai số lớn nặng hơn sai số nhỏ.

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Chuẩn Lp: họ tổng quát

L1 và L2 là các trường hợp riêng của chuẩn Lp:

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

Các giá trị p khác nhau tạo ra "quả cầu đơn vị" có hình dạng khác nhau (tập hợp mọi điểm cách gốc một khoảng bằng 1):

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### Chuẩn L-vô-cực (khoảng cách Chebyshev)

Khi p tiến tới vô cực, chuẩn Lp hội tụ về thành phần có giá trị tuyệt đối lớn nhất.

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

Khoảng cách giữa hai điểm được quyết định bởi đúng một chiều mà chúng khác nhau nhiều nhất. Mọi chiều khác bị bỏ qua.

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

Khi nào dùng L-vô-cực:
- Khi độ lệch tệ nhất ở bất kỳ chiều đơn lẻ nào là điều quan trọng
- Bàn cờ (quân vua trong cờ vua di chuyển theo L-vô-cực: mỗi bước theo bất kỳ hướng nào tốn 1)
- Dung sai chế tạo (mọi chiều đều phải nằm trong giới hạn)

### Độ tương đồng Cosin và Khoảng cách Cosin

Cosine similarity (độ tương đồng cosin) đo góc giữa hai vector, bỏ qua độ lớn của chúng.

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

Nó nằm trong khoảng -1 (ngược hướng) đến +1 (cùng hướng). Hai vector vuông góc có độ tương đồng cosin bằng 0.

Khoảng cách cosin chuyển nó thành một khoảng cách: cosine_distance = 1 - cosine_similarity. Giá trị nằm trong khoảng 0 (cùng hướng) đến 2 (ngược hướng).

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

Vì sao cosin thống trị NLP và embedding: trong văn bản, độ dài tài liệu không nên ảnh hưởng đến độ tương đồng. Một tài liệu về mèo dài gấp đôi một tài liệu khác cũng về mèo vẫn nên "tương đồng". Độ tương đồng cosin bỏ qua độ lớn (độ dài) và chỉ quan tâm đến hướng. Hai tài liệu có cùng phân bố từ nhưng độ dài khác nhau thì cùng hướng và có độ tương đồng cosin bằng 1.0.

Khi nào dùng độ tương đồng cosin:
- Độ tương đồng văn bản (vector TF-IDF, embedding từ, embedding câu)
- Mọi miền mà độ lớn là nhiễu còn hướng là tín hiệu
- Hệ gợi ý (vector sở thích người dùng)
- Tìm kiếm embedding (vector database gần như luôn dùng cosin hoặc tích vô hướng)

### Độ tương đồng tích vô hướng so với Độ tương đồng Cosin

Tích vô hướng của hai vector là:

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

Độ tương đồng cosin chính là tích vô hướng được chuẩn hoá theo độ lớn của cả hai vector. Khi cả hai vector đã được chuẩn hoá đơn vị (độ lớn = 1), tích vô hướng và độ tương đồng cosin là như nhau.

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

Khi chúng khác nhau: tích vô hướng mang theo thông tin về độ lớn. Vector có độ lớn lớn hơn nhận điểm tích vô hướng cao hơn. Điều này quan trọng ở một số hệ truy hồi, nơi bạn muốn các mục "phổ biến" xếp hạng cao hơn. Độ lớn đóng vai trò tín hiệu ngầm về chất lượng hoặc tầm quan trọng.

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

Trong thực tế:
- Dùng độ tương đồng cosin khi bạn muốn độ tương đồng thuần tuý về hướng
- Dùng tích vô hướng khi độ lớn mang thông tin có nghĩa
- Nhiều vector database (Pinecone, Weaviate, Qdrant) cho bạn chọn giữa hai cách
- Nếu embedding của bạn đã chuẩn hoá L2 thì lựa chọn không quan trọng

### Khoảng cách Mahalanobis

Khoảng cách Euclid đối xử với mọi chiều như nhau. Nhưng nếu các đặc trưng của bạn tương quan với nhau hoặc có thang đo khác nhau, L2 cho kết quả sai lệch.

Khoảng cách Mahalanobis tính đến cấu trúc hiệp phương sai của dữ liệu.

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

trong đó S là ma trận hiệp phương sai của dữ liệu.

Hiểu một cách trực giác: khoảng cách Mahalanobis trước hết khử tương quan và chuẩn hoá dữ liệu (whitening), rồi tính khoảng cách L2 trong không gian đã biến đổi đó. Nếu S là ma trận đơn vị (các đặc trưng không tương quan, phương sai bằng đơn vị), khoảng cách Mahalanobis rút gọn thành khoảng cách Euclid.

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

Khi nào dùng khoảng cách Mahalanobis:
- Phát hiện outlier (điểm có khoảng cách Mahalanobis lớn so với giá trị trung bình là outlier)
- Phân loại khi các đặc trưng có thang đo và tương quan khác nhau
- Khi bạn có đủ dữ liệu để ước lượng một ma trận hiệp phương sai đáng tin cậy
- Kiểm soát chất lượng trong sản xuất (giám sát quy trình đa biến)

### Độ tương đồng Jaccard (cho tập hợp)

Độ tương đồng Jaccard đo phần chồng lấn giữa hai tập hợp.

```
J(A, B) = |A intersect B| / |A union B|
```

Nó nằm trong khoảng 0 (không chồng lấn) đến 1 (hai tập trùng nhau). Khoảng cách Jaccard = 1 - độ tương đồng Jaccard.

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

Khi nào dùng Jaccard:
- So sánh các tập thẻ, danh mục hoặc đặc trưng
- Độ tương đồng tài liệu dựa trên sự xuất hiện của từ (không dựa trên tần suất)
- Phát hiện gần trùng lặp (xấp xỉ MinHash của Jaccard)
- So sánh vector đặc trưng nhị phân (dữ liệu hiện diện/không hiện diện)
- Đánh giá model phân đoạn (Intersection over Union = Jaccard)

### Edit Distance (khoảng cách chỉnh sửa) — Levenshtein Distance (khoảng cách Levenshtein)

Edit distance (khoảng cách chỉnh sửa) đếm số phép toán tối thiểu trên từng ký tự cần để biến chuỗi này thành chuỗi kia. Các phép toán gồm: chèn, xóa hoặc thay thế.

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

Cách tính dùng quy hoạch động. Bạn điền một ma trận, trong đó phần tử (i, j) là khoảng cách chỉnh sửa giữa i ký tự đầu của chuỗi A và j ký tự đầu của chuỗi B.

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

Khi nào dùng khoảng cách chỉnh sửa:
- Kiểm tra và sửa lỗi chính tả
- Căn chỉnh trình tự DNA (với các phép toán có trọng số)
- So khớp chuỗi mờ
- Khử trùng lặp cho dữ liệu văn bản lộn xộn

### KL Divergence (không phải khoảng cách, nhưng người ta dùng như khoảng cách)

Phân kỳ KL đo mức độ khác nhau của phân phối xác suất này so với phân phối xác suất kia. Nội dung này đã có ở Bài 09, nhưng vẫn thuộc về phần thảo luận này vì người ta dùng nó như một "khoảng cách" dù nó không phải vậy.

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

Tính chất then chốt: phân kỳ KL KHÔNG đối xứng.

```
D_KL(P || Q) != D_KL(Q || P)
```

Nghĩa là nó không đạt yêu cầu cơ bản của một metric khoảng cách. Nó cũng không thỏa bất đẳng thức tam giác. Nó là một phân kỳ (divergence), không phải khoảng cách.

KL xuôi (D_KL(P || Q)) là "hướng về trung bình" (mean-seeking): Q cố bao phủ mọi mode của P.
KL ngược (D_KL(Q || P)) là "hướng về mode" (mode-seeking): Q tập trung vào một mode duy nhất của P.

Bạn sẽ gặp phân kỳ KL ở:
- VAE (số hạng KL trong ELBO đẩy phân phối tiềm ẩn về phía prior)
- Chưng cất tri thức (student cố khớp phân phối của teacher)
- RLHF (hình phạt KL giữ model sau tinh chỉnh gần với model gốc)
- Các phương pháp policy gradient (ràng buộc các cập nhật policy)

### Wasserstein Distance (khoảng cách dịch chuyển đất)

Khoảng cách Wasserstein đo "công" tối thiểu cần để biến phân phối xác suất này thành phân phối xác suất khác. Hãy hình dung: nếu một phân phối là một đống đất còn phân phối kia là một cái hố, thì bạn phải chuyển bao nhiêu đất và đi bao xa?

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

Với các phân phối 1 chiều, nó rút gọn thành tích phân của hiệu tuyệt đối giữa hai hàm phân phối tích lũy:

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

Vì sao Wasserstein quan trọng:
- Nó là một metric thực sự (đối xứng, thỏa bất đẳng thức tam giác)
- Nó vẫn cho gradient ngay cả khi hai phân phối không chồng lấp (phân kỳ KL tiến tới vô cực)
- Chính tính chất này khiến nó trở thành trung tâm của Wasserstein GAN (WGAN), những model đã giải quyết được sự bất ổn khi huấn luyện GAN nguyên bản

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

Khi nào dùng Wasserstein:
- Huấn luyện GAN (WGAN, WGAN-GP)
- So sánh các phân phối có thể không chồng lấp
- Các bài toán vận tải tối ưu
- Truy hồi ảnh (so sánh histogram màu)

### Vì sao mỗi tác vụ lại cần một khoảng cách khác nhau

| Tác vụ | Khoảng cách tốt nhất | Lý do |
|------|--------------|-----|
| Độ tương đồng văn bản | Cosine | Độ lớn là nhiễu, hướng mới là ý nghĩa |
| So sánh điểm ảnh | L2 | Quan hệ không gian quan trọng, các đặc trưng cùng thang đo |
| Đặc trưng thưa số chiều cao | L1 | Bền vững, không khuếch đại những khác biệt lớn hiếm gặp |
| Giao của tập (thẻ, danh mục) | Jaccard | Dữ liệu tự nhiên là dạng tập hợp, không phải dạng vector |
| So khớp chuỗi | Khoảng cách chỉnh sửa | Các phép toán khớp với trực giác sửa văn bản của con người |
| Phát hiện ngoại lai | Mahalanobis | Tính đến tương quan và thang đo giữa các đặc trưng |
| So sánh phân phối | Phân kỳ KL | Đo lượng thông tin mất đi khi dùng Q thay cho P |
| Huấn luyện GAN | Wasserstein | Cho gradient ngay cả khi hai phân phối không chồng lấp |
| Embedding (vector DB) | Cosine hoặc tích vô hướng | Embedding được huấn luyện để mã hóa ý nghĩa vào hướng |
| Gợi ý | Tích vô hướng | Độ lớn có thể mã hóa độ phổ biến hoặc độ tin cậy |
| Trình tự DNA | Khoảng cách chỉnh sửa có trọng số | Chi phí thay thế khác nhau tùy cặp nucleotide |
| Kiểm soát chất lượng sản xuất | L-infinity | Độ lệch tệ nhất ở bất kỳ chiều nào mới là điều quan trọng |

### Liên hệ với hàm mất mát

Hàm mất mát chính là các hàm khoảng cách áp lên dự đoán so với nhãn đúng.

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### Liên hệ với điều chuẩn

Điều chuẩn thêm một số hạng phạt theo norm trên các trọng số vào hàm mất mát.

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

Vì sao L1 tạo ra độ thưa còn L2 thì không: hãy hình dung vùng ràng buộc trong không gian trọng số 2 chiều. L1 là một hình thoi, L2 là một hình tròn. Các đường đồng mức của hàm mất mát (các ellipse) dễ chạm vào hình thoi nhất tại một đỉnh, nơi một trọng số bằng 0. Chúng chạm vào hình tròn tại một điểm trơn, nơi cả hai trọng số đều khác 0.

### Tìm kiếm láng giềng gần nhất (nearest neighbor)

Mỗi hàm khoảng cách đều kéo theo một bài toán tìm kiếm láng giềng gần nhất: cho một điểm truy vấn, hãy tìm những điểm gần nhất trong một tập dữ liệu.

Tìm kiếm láng giềng gần nhất chính xác tốn O(n * d) cho mỗi truy vấn trên tập dữ liệu n điểm với d chiều. Với tập dữ liệu lớn, cách này quá chậm.

Các thuật toán láng giềng gần nhất xấp xỉ (Approximate Nearest Neighbor, ANN) đánh đổi một chút độ chính xác để lấy tốc độ nhanh hơn rất nhiều:

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW (Hierarchical Navigable Small World) là thuật toán thống trị trong các vector database hiện đại. Nó dựng một đồ thị nhiều tầng, trong đó mỗi nút nối với các láng giềng gần nhất xấp xỉ của nó. Tìm kiếm bắt đầu ở tầng trên cùng (thưa, nhảy xa) rồi đi xuống tầng dưới cùng (dày, nhảy ngắn).

```figure
norm-unit-balls
```

## Tự tay làm

### Bước 1: Toàn bộ các hàm norm và khoảng cách

Xem `code/distances.py` để có phần cài đặt đầy đủ. Mọi hàm đều được viết từ đầu chỉ bằng toán cơ bản của Python.

### Bước 2: Cùng dữ liệu, khác khoảng cách, khác láng giềng

Bản demo trong `distances.py` tạo một tập dữ liệu, chọn một điểm truy vấn rồi cho thấy láng giềng gần nhất thay đổi thế nào tùy theo metric khoảng cách. Điểm "gần nhất" theo L1 có thể không phải là điểm gần nhất theo L2 hay cosine.

### Bước 3: Tìm kiếm tương đồng trên embedding

Đoạn code có một bản tìm kiếm tương đồng trên embedding giả lập, tìm những "tài liệu" giống nhất với một truy vấn bằng độ tương đồng cosine so với khoảng cách L2, qua đó cho thấy thứ hạng có thể khác nhau.

## Áp dụng

Cách dùng thực tế phổ biến nhất: tìm các mục tương tự trong một vector database.

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

Khi bạn gọi `model.encode(text)` rồi tìm kiếm trong một vector database, đây chính là điều diễn ra bên dưới. Model embedding ánh xạ văn bản thành vector. Vector database tính độ tương đồng cosine (hoặc tích vô hướng) giữa vector truy vấn của bạn với mọi vector đã lưu, dùng các thuật toán ANN để khỏi phải kiểm tra tất cả.

## Bài tập

1. Tính khoảng cách L1, L2 và L-infinity giữa (1, 2, 3) và (4, 0, 6). Kiểm chứng rằng L-inf <= L2 <= L1 luôn đúng với mọi cặp điểm. Chứng minh vì sao thứ tự này được bảo đảm.

2. Tạo hai vector có độ tương đồng cosine cao (> 0.9) nhưng khoảng cách L2 lớn (> 10). Giải thích bằng hình học điều gì đang xảy ra. Sau đó tạo hai vector có độ tương đồng cosine thấp (< 0.3) nhưng khoảng cách L2 nhỏ (< 0.5).

3. Cài đặt một hàm nhận vào tập dữ liệu và một điểm truy vấn, trả về láng giềng gần nhất theo L1, L2, cosine và khoảng cách Mahalanobis (Mahalanobis distance). Tìm một tập dữ liệu mà cả bốn metric không đồng ý với nhau về điểm nào là gần nhất.

4. Tính bằng tay khoảng cách Wasserstein giữa [0.5, 0.5, 0, 0] và [0, 0, 0.5, 0.5] theo phương pháp CDF. Sau đó tính khoảng cách đó giữa [0.25, 0.25, 0.25, 0.25] và [0, 0, 0.5, 0.5]. Khoảng cách nào lớn hơn và vì sao?

5. Cài đặt MinHash cho độ tương đồng Jaccard (Jaccard similarity) xấp xỉ. Sinh 100 tập ngẫu nhiên, tính Jaccard chính xác cho mọi cặp, rồi so sánh với xấp xỉ MinHash dùng 50, 100 và 200 hàm băm. Vẽ đồ thị sai số xấp xỉ.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|----------------------|
| Norm | "Kích thước của một vector" | Một hàm ánh xạ vector thành số vô hướng không âm, thỏa bất đẳng thức tam giác, tính thuần nhất tuyệt đối và chỉ bằng 0 với vector không |
| L1 norm | "Khoảng cách Manhattan" | Tổng giá trị tuyệt đối các thành phần. Tạo ra độ thưa trong tối ưu. Bền vững với ngoại lai |
| L2 norm | "Khoảng cách Euclid" | Căn bậc hai của tổng bình phương các thành phần. Khoảng cách đường thẳng trong không gian Euclid |
| Lp norm | "Norm tổng quát" | Căn bậc p của tổng lũy thừa bậc p của giá trị tuyệt đối các thành phần. L1 và L2 là các trường hợp riêng |
| L-infinity norm | "Norm cực đại" hoặc "khoảng cách Chebyshev" | Giá trị tuyệt đối lớn nhất trong các thành phần. Giới hạn của Lp khi p tiến tới vô cực |
| Cosine similarity | "Góc giữa hai vector" | Tích vô hướng chuẩn hóa theo độ lớn của cả hai vector. Nhận giá trị từ -1 đến +1. Bỏ qua độ dài vector |
| Cosine distance | "1 trừ độ tương đồng cosine" | Chuyển độ tương đồng cosine thành khoảng cách. Nhận giá trị từ 0 đến 2 |
| Dot product | "Cosine chưa chuẩn hóa" | Tổng các tích từng thành phần. Bằng độ tương đồng cosine nhân với độ lớn của cả hai vector |
| Mahalanobis distance | "Khoảng cách có tính đến tương quan" | Khoảng cách L2 trong một không gian đã được tẩy trắng (khử tương quan và chuẩn hóa) bằng ma trận hiệp phương sai của dữ liệu |
| Jaccard similarity | "Mức giao của hai tập" | Kích thước phần giao chia cho kích thước phần hợp. Dùng cho tập hợp, không phải vector |
| Edit distance | "Khoảng cách Levenshtein" | Số phép chèn, xóa và thay thế tối thiểu để biến chuỗi này thành chuỗi kia |
| KL divergence | "Khoảng cách giữa các phân phối" | Không phải khoảng cách thực sự (không đối xứng). Đo số bit thêm vào khi dùng Q để mã hóa P |
| Wasserstein distance | "Khoảng cách dịch chuyển đất" | Công tối thiểu để vận chuyển khối lượng từ phân phối này sang phân phối khác. Là một metric thực sự |
| Approximate nearest neighbor | "Tìm kiếm ANN" | Các thuật toán (HNSW, LSH, IVF) tìm điểm gần nhất một cách xấp xỉ, nhanh hơn nhiều so với tìm kiếm chính xác |
| HNSW | "Thuật toán của vector DB" | Đồ thị Hierarchical Navigable Small World. Đồ thị nhiều tầng cho tìm kiếm láng giềng gần nhất xấp xỉ tốc độ cao |
| L1 regularization | "Lasso" | Cộng norm L1 của trọng số vào hàm mất mát. Đẩy trọng số về 0 (độ thưa) |
| L2 regularization | "Ridge" hoặc "weight decay" | Cộng bình phương norm L2 của trọng số vào hàm mất mát. Co trọng số về gần 0 mà không tạo độ thưa |
| Elastic Net | "L1 + L2" | Kết hợp điều chuẩn L1 và L2. Xử lý các nhóm đặc trưng tương quan tốt hơn từng cái riêng lẻ |

## Đọc thêm

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - thư viện của Meta cho tìm kiếm ANN quy mô hàng tỷ
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - bài báo đưa khoảng cách Earth Mover's vào GAN
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - thuật toán ANN nền tảng
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec, nơi độ tương đồng cosine trở thành mặc định cho embedding
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - hướng dẫn thực hành về các metric khoảng cách và thuật toán láng giềng trong scikit-learn

---

[← Bài trước: Ổn định số học](../../13-numerical-stability/docs/vi.md) · [Bài sau: Thống kê cho Học máy →](../../15-statistics-for-ml/docs/vi.md)
