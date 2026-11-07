# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

**SEED**: 3557

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | x |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | x |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | x |

## 2. Kết quả chính

### Bảng so sánh các mô hình

| Mô hình | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Tin/s (cả lô) | p50 (ms/tin) | p95 (ms/tin) | USD/1k tin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. BoW + LogReg (tự code) | 1.000 | 0.705 | 0.827 | 0.750 | 4.551 | 27383.430 | 0.049 | 0.180 | 0.0 |
| 2. BoW + MLP (Deep Learning) | 1.000 | 0.830 | 0.907 | 0.857 | 7.843 | 30080.788 | 0.281 | 0.441 | 0.0 |
| 3a. Encoder (MiniLM) + kNN | 0.816 | 0.909 | 0.860 | 0.923 | 79.504 | 28.151 | 47.549 | 76.529 | 0.0 |
| 3b. Encoder (MiniLM) + LogReg | 0.951 | 0.875 | 0.911 | 0.889 | 81.392 | 38.735 | 24.242 | 35.136 | 0.0 |
| 3c. Encoder (MiniLM) + MLP head | 0.963 | 0.886 | 0.923 | 0.919 | 80.405 | 38.067 | 25.306 | 31.852 | 0.0 |
| 4a. LLM nhỏ Zero-shot | 0.833 | 1.000 | — | 0.909 | 0.000 | 0.845 | 763.056 | 2890.710 | 0.0 |
| 4b. LLM lớn Zero-shot | 0.900 | 0.900 | — | 0.900 | 0.000 | 0.998 | 502.946 | 2629.412 | 0.0 |
| 4c. LLM nhỏ Few-shot | 0.826 | 0.950 | — | 0.884 | 0.000 | 0.586 | 1321.415 | 3944.791 | 0.0 |

### Ghi chú

- Các mô hình LLM chỉ được đánh giá trên SUB 40 tin nhắn, vì vậy cần so sánh chất lượng bằng cột F1 (SUB 40).
- SUB chỉ có 40 tin nên một dự đoán sai có thể làm F1 thay đổi đáng kể. Chênh lệch F1 dưới 0.05 chưa đủ để khẳng định mô hình nào tốt hơn.
- Thời gian inference được đo từ văn bản thô đến nhãn dự đoán, bao gồm cả bước vectorize hoặc encode.
- p50 và p95 là độ trễ khi xử lý từng tin nhắn.
- Thời gian train của các mô hình Encoder bao gồm thời gian mã hóa tập train bằng MiniLM và huấn luyện bộ phân loại.
- LLM có train = 0 vì không được huấn luyện lại trong thí nghiệm.
- USD/1k tin = 0.0 là giá trị notebook hiển thị, không đồng nghĩa với việc API LLM miễn phí.

### Nhận xét

- **F1 test cao nhất:** Encoder (MiniLM) + MLP head, đạt 0.923.
- **Mô hình nhanh nhất theo p95:** BoW + LogReg, đạt 0.180 ms/tin.
- **Các mô hình đạt SLA p95 < 50 ms:** BoW + LogReg, BoW + MLP, Encoder + LogReg và Encoder + MLP head.
- **Mô hình cân bằng chất lượng và tốc độ:** BoW + MLP đạt F1 = 0.907 với p95 = 0.441 ms/tin.
- **Mô hình có F1 cao nhất trong nhóm đạt SLA:** Encoder + MLP head, đạt F1 = 0.923 với p95 = 31.852 ms/tin.
- Các mô hình LLM có độ trễ cao hơn nhiều so với yêu cầu 50 ms/tin, vì vậy chưa phù hợp để xử lý đồng bộ toàn bộ SMS.

## 3. Dự đoán và thực tế

**Dự đoán 1 – Learning rate:**

Em dự đoán lr=50 sẽ khiến mô hình không hội tụ do bước cập nhật quá lớn. Tuy nhiên, thực nghiệm cho thấy loss vẫn giảm xuống 0.003, thấp hơn lr=0.5 (loss = 0.074). Điều này cho thấy learning rate lớn không phải lúc nào cũng khiến mô hình thất bại.

**Dự đoán 2 – MLP so với Logistic Regression:**

Em dự đoán MLP có F1 tương đương hoặc thấp hơn Logistic Regression vì có nhiều tham số và dễ overfitting. Thực tế MLP đạt F1 = 0.907, cao hơn Logistic Regression (0.827). MLP có khả năng học quan hệ phi tuyến nhưng validation loss bắt đầu tăng sau epoch 5, cho thấy dấu hiệu overfitting.

**Dự đoán 3 – Few-shot LLM:**

Em dự đoán Few-shot sẽ giúp F1 tăng khoảng 0.02–0.05 so với Zero-shot. Tuy nhiên, F1 thực tế giảm từ 0.909 xuống 0.884, đồng thời p95 tăng từ 2400.848 ms lên 3491.186 ms. Như vậy, thêm ví dụ chưa chắc cải thiện chất lượng mà còn tăng độ trễ.

## 4. Phân tích 50 case khó

Mô hình **BoW + MLP tụt F1 nhiều nhất**, từ 0.857 trên SUB 40 xuống 0.324 trên 50 case khó, giảm 0.533. BoW + LogReg cũng giảm từ 0.750 xuống 0.286.

Nhóm khó nhất với BoW là **spam giọng tự nhiên**, với tỷ lệ phân loại đúng 0% ở cả LogReg và MLP. Nguyên nhân là những tin nhắn này sử dụng từ ngữ thông thường, không chứa các từ khóa spam quen thuộc.

**Case #12 – Spam giọng tự nhiên:**

"hi there, sorry to bother you, my friend gave me your number..."

Đây là spam nhưng được viết giống lời làm quen lịch sự. BoW dễ nhầm thành ham vì chỉ nhận diện từ riêng lẻ, không hiểu mục đích dụ người nhận truy cập trang hẹn hò.

**Case #2 – Ham trông giống spam:**

"FREE pizza in the office kitchen right now, come quick before it's gone"

Đây là tin nhắn ham nhưng LLM phòng thủ vẫn dự đoán sai. Nguyên nhân là tin chứa từ "FREE" và lời thúc giục "right now", giống nội dung quảng cáo mặc dù thực tế chỉ là thông báo giữa đồng nghiệp.

**Kết luận:** BoW dễ bị đánh lừa bởi từ khóa và thiếu ngữ cảnh. Embedding hiểu ngữ nghĩa tốt hơn nhưng vẫn có thể nhầm lẫn. LLM xử lý các case khó tốt hơn nhưng không hoàn toàn chính xác, đặc biệt với các tin ham có hình thức giống spam.


## 5. Thử thách cá nhân

**Đề #6: LLM trả JSON thay vì một từ.**

**Cách thực hiện:**

Em xây dựng hàm `build_json_prompt()` yêu cầu LLM trả về JSON gồm hai trường `label` và `reason`. Sau đó viết hàm `parse_json_label()` sử dụng `json.loads()` để đọc nhãn spam/ham.

Em cũng xây dựng chương trình chạy trên 10 tin nhắn SUB để so sánh F1, số token, độ trễ và chi phí giữa cách trả JSON và cách trả lời một từ.

**Kết quả:**

Em đã viết code thử nghiệm và kiểm tra logic tạo prompt, đọc JSON. Tuy nhiên, chưa có số liệu thực nghiệm đầy đủ được xác nhận do gặp lỗi cấu hình API key.

**Giải thích:**

Em dự đoán JSON sẽ tiêu thụ nhiều output token hơn vì cần tạo thêm phần giải thích. Cách trả JSON hữu ích khi cần lưu kết quả có cấu trúc và kiểm tra lý do phân loại. Tuy nhiên, nếu chỉ cần nhãn spam/ham trên lượng lớn tin nhắn, cách trả một từ sẽ tiết kiệm chi phí hơn.

## 6. Phương án cho sản phẩm thật

Nhà mạng cần lọc 10 triệu SMS/ngày, trung bình khoảng 116 tin/giây, với yêu cầu p95 dưới 50 ms/tin.

Theo kết quả thực nghiệm, ba mô hình đạt yêu cầu độ trễ là:

- BoW + LogReg: F1 = 0.827, p95 = 0.076 ms.
- BoW + MLP: F1 = 0.907, p95 = 0.237 ms.
- Encoder + LogReg: F1 = 0.911, p95 = 44.480 ms.

**Em chọn BoW + MLP làm mô hình chính** vì tốc độ xử lý khoảng 39.356 tin/giây, F1 = 0.907, Precision = 1.0 và Recall = 0.830. Mô hình đáp ứng tốt yêu cầu độ trễ, nhưng cần tiếp tục cải thiện Recall để hạn chế bỏ sót tin spam.

**Thiết kế hệ thống lai (Router):**

Em đề xuất kết hợp BoW + MLP, Encoder + LogReg và LLM.

Đầu tiên, BoW + MLP phân loại các tin nhắn. Nếu xác suất spam nhỏ hơn 0.3 thì dự đoán ham, lớn hơn 0.7 thì dự đoán spam. Những tin có xác suất từ 0.3 đến 0.7 được chuyển sang Encoder + LogReg để kiểm tra dựa trên ngữ nghĩa.

Với các trường hợp nghi ngờ lừa đảo phức tạp, LLM có thể được sử dụng để kiểm tra bổ sung theo cơ chế bất đồng bộ. Điều này giúp giảm số lần gọi API nhưng vẫn tận dụng được khả năng hiểu ngôn ngữ của LLM. Cần kiểm tra lại SLA của toàn hệ thống vì các bước xử lý liên tiếp có thể làm tăng độ trễ.

**Chi phí LLM:**

Nếu dùng gpt-6-luna để xử lý toàn bộ 10 triệu SMS/ngày, giả sử mỗi tin sử dụng 100 input token và 1 output token, với giá 0.10 USD/1M input token và 0.50 USD/1M output token:

- Chi phí input: 100 USD/ngày.
- Chi phí output: 5 USD/ngày.
- Tổng: 105 USD/ngày.
- Chi phí 30 ngày: **3.150 USD/tháng**.

Đây chỉ là chi phí API ước tính theo giả định của bài. Hệ thống lai có thể giảm chi phí bằng cách chỉ gửi những tin khó cho LLM.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| ML | Giúp viết Gradient Descent và dự đoán learning rate | Viết code cập nhật trọng số, dự đoán lr=50 khó hội tụ | Chạy thử, phát hiện lr=50 vẫn hội tụ với loss 0.003 |
| DL | Hướng dẫn huấn luyện MLP bằng PyTorch | Viết 5 bước huấn luyện, giải thích overfitting | Kiểm thử, F1 đạt 0.907; val loss thấp nhất epoch 5 |
| Embedding | Viết cosine similarity và kNN | Code chuẩn hóa vector, nhân ma trận và tìm hàng xóm gần nhất | Kiểm tra công thức và kết quả phân loại |
| LLM – API | Giúp làm TODO 4.1 và sửa lỗi API key | Viết call_openai, hướng dẫn cấu hình và xử lý lỗi 401 | Kiểm tra và cập nhật API key trong Google Colab |
| LLM – Prompt | Giúp làm TODO 4.2–4.3 | Viết prompt Zero-shot/Few-shot và hàm parse_label | Kiểm tra các trường hợp spam, ham, not spam và nhãn không hợp lệ |
| LLM – Batch | Giúp làm TODO 4.4 | Viết hàm phân loại hàng loạt, retry, tính token, latency và chi phí | Sửa lỗi IndentationError và kiểm tra logic |
| LLM – JSON | Hướng dẫn thử thách trả JSON thay vì một từ | Viết build_json_prompt, parse_json_label và code so sánh | Kiểm tra logic, chưa xác nhận đầy đủ kết quả thực nghiệm |
| Stress test | Giải thích kết quả 50 case khó và tạo thêm case | Phân tích lỗi của từng mô hình, đề xuất 7 case | Đối chiếu F1 và sửa lỗi JSONDecodeError |
| Tổng kết | Đề xuất hệ thống đạt SLA dưới 50 ms | Phân tích F1, p95, chi phí và đề xuất Router | Đối chiếu bảng số liệu thực nghiệm |
