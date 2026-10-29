# Model Card — Student Performance Classifier

- **Mục đích:** nguyên mẫu học thuật phân loại mức học tập và minh họa gợi ý học tập.
- **Mô hình được chọn:** Random Forest, chọn theo Macro-F1 trên tập kiểm thử.
- **Đầu vào:** chuyên cần, giờ tự học, tỷ lệ hoàn thành bài tập, điểm quiz, điểm giữa kỳ, mức độ tham gia.
- **Đầu ra:** `Cần hỗ trợ`, `Đạt yêu cầu`, `Tốt`.
- **Dữ liệu:** 1.200 bản ghi mô phỏng; chia train/test phân tầng 80/20 (960/240), random_state=42.
- **Môi trường tái lập:** Python 3.13, scikit-learn 1.8.0 (xem `requirements.txt`).
- **Kết quả lần chạy:** Accuracy 0.6292; macro-Precision 0.5525; macro-Recall 0.5556; macro-F1 0.5494.
- **Giới hạn trọng yếu:** nhãn được sinh từ chính các biến đầu vào theo quy tắc giả lập. Chỉ số đánh giá đo mức độ học lại quy tắc mô phỏng, không chứng minh hiệu năng dự báo kết quả học tập thực tế.
- **Không dùng cho:** quyết định học vụ, tuyển sinh, cấp học bổng, kỷ luật, hoặc xếp hạng sinh viên ngoài đời.
- **Điều kiện để nâng cấp:** dữ liệu được phép sử dụng, định nghĩa nhãn độc lập và có ý nghĩa giáo dục, kiểm định ngoài mẫu, đánh giá theo nhóm, hiệu chuẩn xác suất, và giám sát của giảng viên.
