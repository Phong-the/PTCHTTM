# AI-Powered Adaptive Learning & Student Performance Prediction

**Môn học:** Phát triển các hệ thống thông minh — PTIT  
**Thành viên:** Hồ Hiểu Thanh, Lê Nhật Linh, La Thế Phong

## Thành phần dự án
- `app.py`: ứng dụng Streamlit, dự đoán mức học tập và gợi ý lộ trình theo quy tắc.
- `train_model.py`: tạo dữ liệu mô phỏng, huấn luyện và so sánh Logistic Regression, Decision Tree, Random Forest; xuất toàn bộ kết quả đánh giá.
- `student_learning_data_synthetic.csv`: 1.200 bản ghi mô phỏng.
- `models/performance_classifier.joblib`: pipeline mô hình đã huấn luyện.
- `outputs/`: metrics, dự đoán tập test, bảng so sánh, biểu đồ và kết quả smoke test.
- `tests/test_model.py`: kiểm thử khói mô hình.
- `docs/DATA_DICTIONARY.md`: từ điển dữ liệu.
- `docs/SYSTEM_DESIGN.md`: kiến trúc logic và các thành phần.
- `MODEL_CARD.md`: phạm vi sử dụng và giới hạn mô hình.
- Ba file kế hoạch/báo cáo/WBS theo mẫu môn học.

## Cài đặt và chạy trên Windows
Yêu cầu Python 3.13 (đã kiểm thử). Mở CMD/PowerShell trong thư mục dự án:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Huấn luyện lại và tạo kết quả đánh giá

```powershell
python train_model.py
python -m unittest discover -s tests -v
```

`train_model.py` sinh lại CSV, model, toàn bộ bảng/biểu đồ trong `outputs/` và
`outputs/smoke_test_results.json`. Mọi bước đều dùng `random_state=42`.

## Tái lập kết quả
Các thư viện được pin trong `requirements.txt`, trong đó `scikit-learn==1.8.0`
là phiên bản đã tạo ra model và các chỉ số đã ghi nhận. Chạy lại trên môi trường
khác phiên bản có thể cho kết quả Random Forest khác.

## Kết quả mô hình lần chạy đã ghi nhận
- Số bản ghi: 1.200.
- Train/test: 960/240, chia phân tầng.
- Mô hình được chọn theo Macro-F1: Random Forest.
- Accuracy: 62.92%; Macro-Precision: 55.25%; Macro-Recall: 55.56%; Macro-F1: 54.94%.

## Lưu ý học thuật
Dữ liệu hiện tại là **dữ liệu mô phỏng**, không phải dữ liệu sinh viên thật. Nhãn được tạo từ các biến đầu vào theo một quy tắc tổng hợp, vì vậy kết quả không thể được diễn giải là hiệu năng dự đoán thực tế. Cần thay bằng dữ liệu giáo dục có nguồn gốc, quyền sử dụng, ẩn danh và được phê duyệt trước khi triển khai ngoài phạm vi minh họa.
