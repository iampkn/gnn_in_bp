```markdown
# YÊU CẦU DỰ ÁN (PROJECT SPECIFICATION): KHÁM PHÁ VÀ TỐI ƯU QUY TRÌNH BẰNG GRAPH NEURAL NETWORKS (GNN)

## 1. TỔNG QUAN DỰ ÁN (PROJECT OVERVIEW)

Dự án nhằm mục tiêu xây dựng một hệ thống Khám phá Quy trình Tự động (Automated Process Discovery - APD) sử dụng Học máy trên Đồ thị (Graph Machine Learning) kết hợp với Cơ sở dữ liệu Vector (Vector Database). Đầu ra là một mạng Petri (Petri net) hợp lệ và các gợi ý tối ưu/dự báo chi phí dựa trên nhật ký sự kiện (event logs) của các quy trình IT.

## 2. KIẾN TRÚC VÀ CÔNG NGHỆ (TECH STACK)

- **Ngôn ngữ lõi:** Python 3.12+
- **Deep Learning/Graph Framework:** PyTorch, DGL (Deep Graph Library).
- **Vector Database:** Qdrant / Milvus (Chạy qua Docker).
- **Backend:** FastAPI.
- **Frontend:** NextJs -> Đã install sẵn ở path /fe.
- **Thư viện xử lý Process Mining:** `pm4py` (hỗ trợ mô phỏng event log và hiển thị Petri net) -> Đã cài đặt trên môi trường base anaconda.

---

## 3. CHI TIẾT CÁC GIAI ĐOẠN THỰC HIỆN (EXECUTION PHASES)

### PHASE 1: Sinh Dữ Liệu & Tiền Xử Lý (Data Generation)

**Nhiệm vụ:**

1.  **Đọc dữ liệu gốc:** Viết script Python đọc cấu trúc đồ thị cơ sở G1, G2 từ các file `node.csv`, `edge.csv`, `human.csv`, `device.csv`.
2.  **Sinh tự động 20 quy trình IT có logic khác nhau:** ✅ Đã tạo file `process_templates.py` định nghĩa 20 quy trình IT độc lập (G3-G22), mỗi quy trình có topology, activities, gateway patterns khác nhau hoàn toàn. Bao gồm: CI/CD Pipeline, Xử lý sự cố mạng, Quản lý quyền truy cập, Change Management ITIL, Backup & Recovery, Ứng phó bảo mật, Mua sắm phần cứng, Phát hành phần mềm, Bug Tracking, Database Migration, Cloud Infrastructure, Pentest, IT Onboarding, Disaster Recovery, API Integration, Performance Optimization, Agile Sprint, IT Audit, Microservices, ETL Pipeline. Tổng cộng 178 unique activities.
3.  **Mô phỏng Nhật ký sự kiện (Event Logs):**
    - Chạy mô phỏng (simulation) trên 22 đồ thị để tạo ra tập dữ liệu nhật ký sự kiện dạng `Event Logs $L$` chứa nhiều chuỗi thực thi (traces). Mặc định 10 traces/graph (tùy chỉnh qua API parameter `traces_per_graph`). Tổng ~2,500 events cho 22 graphs.
    - Các node `ExclusiveGateway` tạo ra các luồng (traces) khác nhau với xác suất lặp giảm dần.

### PHASE 2: Tích hợp Vector Database

**Nhiệm vụ:**

1.  **Trích xuất đặc trưng (Feature Extraction):**
    - Theo mặc định của kiến trúc, tạo vector đặc trưng ban đầu $h_i^{(0)}$ cho các hành động bằng `one-hot encoding` nhãn của hành động cộng với `tần suất xuất hiện` của nó trong event log.
    - **Mở rộng:** Bổ sung các thông tin như `Cost`, `HumanRes` từ `node.csv` và `Role` từ `human.csv` vào vector nhúng để làm giàu ngữ cảnh. Đối với các "candidate places" (nơi ứng viên) trong đồ thị, khởi tạo bằng zero vectors.
2.  **Lưu trữ:** Dựng Docker container cho Vector DB (Qdrant/Milvus). Đẩy các vector đặc trưng của các sự kiện, nhân sự, thiết bị vào DB để hỗ trợ truy vấn các tác vụ tương đồng về sau.

### PHASE 3: Kiến Trúc GNN Khám Phá Quy Trình (Core GNN Architecture)

**Mô tả Bài toán:** Biến đổi Event Log $L$ thành một đồ thị hợp nhất $G$, bao gồm "Trace graph" (đồ thị nhật ký sự kiện) và "Candidate Petri net graph" (mạng Petri chứa tất cả các node/cạnh ứng viên có thể xảy ra).

**Cấu trúc 4 Mạng Nơ-ron (DGL/PyTorch Implementation):**

- **1. Propagation Network 1 (PN1):**
    - _Loại kiến trúc:_ Graph Convolutional Network (GCN) với cơ chế Attention đa đầu (K-headed attention).
    - _Số lớp (Layers):_ Tối thiểu **3 lớp** để đảm bảo thông tin hành vi từ sự kiện trước và sau có thể hội tụ tại một "nơi" (place).
    - _Đặc điểm đồ thị:_ Cạnh 2 chiều (bi-directional) tích hợp vector định hướng, thêm các self-loops (tự liên kết) để bảo toàn thông tin node.
    - _Hàm kích hoạt:_ Hàm `RELU` tại lớp cuối cùng để cho ra vector đặc trưng cập nhật chứa mật mã cấu trúc.

- **2. Select Candidate Network (SCN):**
    - _Loại kiến trúc:_ Mạng nơ-ron truyền thống, 1 lớp kết nối đầy đủ (Single fully-connected layer).
    - _Chức năng:_ Tính điểm số $s_v$ bằng cách nhân vector node $h_v$ với ma trận trọng số $W$. Sử dụng hàm `SOFTMAX` để chuẩn hóa thành xác suất phân phối $p_v$ và chọn ứng viên có xác suất cao nhất.
    - _Ràng buộc (Constraint):_ Tích hợp thuật toán kiểm tra **S-coverability**. Nếu ứng viên được chọn gây ra mất S-coverability (nguy cơ bế tắc - deadlock), gạch bỏ ứng viên đó và chọn ứng viên có xác suất cao tiếp theo.

- **3. Stop Network (SN):**
    - _Loại kiến trúc:_ Mạng nơ-ron 2 lớp.
    - _Chức năng:_ Tổng hợp vector đặc trưng của toàn đồ thị (graph embedding) thông qua "gating function" dùng hàm `SIGMOID`. Sau đó, áp dụng thêm một hàm `SIGMOID` nữa ở lớp thứ 2 để ra quyết định dừng (Stop) quá trình thêm node hay không.
    - _Ràng buộc:_ Nếu quyết định dừng được đưa ra nhưng mạng Petri chưa liên thông hoàn toàn (không phải workflow net), tự động hủy bỏ quyết định dừng.

- **4. Propagation Network 2 (PN2):**
    - _Loại kiến trúc:_ GCN với cơ chế Attention đa đầu.
    - _Số lớp:_ Ít nhất **2 lớp**.
    - _Chức năng:_ Cập nhật và lan truyền thông tin về quyết định vừa được đưa ra ở SCN đi khắp đồ thị, cung cấp ngữ cảnh cho các lựa chọn ứng viên ở bước lặp tiếp theo.

**Huấn luyện (Training) & Suy luận (Inference):**

- _Hàm Loss:_ Negative log-likelihood để học xác suất phân phối khớp với đồ thị mẫu.
- _Phương pháp Training:_ Sử dụng **Teacher Forcing** theo thuật toán duyệt đồ thị theo chiều rộng (breadth-first) để ép mô hình học đúng trình tự chọn cấu trúc.
- _Phương pháp Inference:_ Sử dụng **Beam Search** (ví dụ beam width = 10 hoặc 50) để tìm ra mạng Petri có xác suất đúng cao nhất.

### PHASE 4: Tích hợp Lớp Dự Báo/Tối Ưu (Regression Head)

**Nhiệm vụ:**
Tại điểm cuối của mạng (sau khi kiến trúc Petri Net được xác lập), thêm một lớp hồi quy (Regression Layer / Linear Layer).

- _Đầu vào:_ Graph embedding đầu ra hoặc chuỗi các node đã được chọn.
- _Đầu ra dự báo:_ Dự báo tổng chi phí `Cost` hoặc thời gian/số lượng `HumanRes` cho một nhánh quy trình cụ thể. Dữ liệu này được ánh xạ ngược lại từ Vector Database.

### PHASE 5: Web Tương Tác (UI/UX Backend)

**Nhiệm vụ:** Viết các API và giao diện để người dùng thao tác.

1.  **API `/generate-data`**: Chạy tập script ở Phase 1. Trả về bảng danh sách Event Logs.
2.  **API `/search-vector`**: Truy vấn Vector DB để tìm các "Task" giống nhau (Ví dụ: tìm các tác vụ có "Cost > 100" và thuộc "Graph G1").
3.  **API `/discover-process`**: Upload Event Log $\rightarrow$ Chạy Inference với mô hình GNN $\rightarrow$ Trả về JSON chứa cấu trúc Petri Net và thông tin dự báo chi phí tối ưu.
4.  **API `/petri-net`**: Trả về cấu trúc đồ thị quy trình (template graph) cho Graph ID đã chọn. Mỗi node hiển thị đúng 1 lần với cost, human_res gốc. Gateway nodes hiển thị với tất cả nhánh rẽ. Không cần event log — đọc trực tiếp từ template graphs (G1-G2 từ CSV, G3-G22 từ `process_templates.py`).
5.  **UI Component**: ✅ Hiển thị cấu trúc quy trình trực quan bằng SVG (interactive: zoom, pan, click node). Mỗi bước chỉ xuất hiện 1 lần, gateway hiển thị đầy đủ nhánh rẽ. Zoom sử dụng native `wheel` event listener (`passive: false`, dependency `[data]`). Node detail panel hiển thị Cost và Human Res.
6.  **API `/explain-net` (XAI)**: ✅ Nhận thông tin template graph + discovered Petri Net, gọi OpenAI GPT-4o-mini để phân tích và giải thích cho nhà quản lý bằng tiếng Việt. Bao gồm: ý nghĩa Place Structure, so sánh với quy trình gốc, đề xuất tối ưu (rút ngắn, song song hóa), cảnh báo bottleneck/gateway. Nút "Ask AI to Explain" hiển thị sau mỗi Petri Net card trên tab Process Discovery. Key OpenAI đọc từ file `.env`. Kết quả trả về dạng Markdown được render đầy đủ (headings, bold, bullet lists) qua `react-markdown` + `@tailwindcss/typography`.

---

## 4. TIÊU CHÍ NGHIỆM THU (ACCEPTANCE CRITERIA)

1.  Luồng tự động hóa chạy thành công từ bước sinh dữ liệu CSV ra Event Log.
2.  Kiến trúc PN1 và SCN phải code theo chuẩn DGL/PyTorch, có chú thích rõ ràng việc áp dụng Multi-head attention và check S-coverability.
3.  Hệ thống chạy được quá trình Inference (sử dụng Beam Search) xuất ra một mạng Petri hợp lệ (Sound Petri Net) mà không bị kẹt deadlock.
4.  Giao diện Web hiển thị được đồ thị và hoạt động ổn định với các API Backend.
```
