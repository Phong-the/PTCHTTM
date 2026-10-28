# Từ điển dữ liệu

Dữ liệu hiện tại là dữ liệu tổng hợp/mô phỏng để minh họa quy trình, không phải hồ sơ sinh viên thật.

| Trường | Kiểu | Miền/đơn vị | Ý nghĩa |
|---|---|---|---|
| attendance_pct | số thực | 0–100 (%) | Tỷ lệ chuyên cần |
| weekly_study_hours | số thực | giờ/tuần | Thời lượng tự học trung bình mỗi tuần |
| assignment_completion_pct | số thực | 0–100 (%) | Tỷ lệ bài tập hoàn thành |
| quiz_average | số thực | 0–100 điểm | Điểm quiz trung bình |
| midterm_score | số thực | 0–100 điểm | Điểm giữa kỳ |
| engagement_level | số thực | 1–5 | Mức độ tham gia học tập |
| performance_level | phân loại | Cần hỗ trợ / Đạt yêu cầu / Tốt | Nhãn mục tiêu mô phỏng |

## Quy tắc tạo nhãn mô phỏng
Một điểm tiềm ẩn được tính từ tổng có trọng số của sáu biến đầu vào cộng nhiễu ngẫu nhiên. Điểm dưới 52 gán nhãn `Cần hỗ trợ`; từ 52 đến dưới 72 gán `Đạt yêu cầu`; từ 72 trở lên gán `Tốt`. Vì nhãn được sinh từ các biến đầu vào theo quy tắc này, metric không thể đại diện cho hiệu năng trên dữ liệu sinh viên ngoài đời.
