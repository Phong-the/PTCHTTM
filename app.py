from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "performance_classifier.joblib"
DATA_PATH = ROOT / "student_learning_data_synthetic.csv"
FEATURES = [
    "attendance_pct", "weekly_study_hours", "assignment_completion_pct",
    "quiz_average", "midterm_score", "engagement_level"
]
LABEL_VI = {
    "Cần hỗ trợ": "Cần hỗ trợ",
    "Đạt yêu cầu": "Đạt yêu cầu",
    "Tốt": "Tốt",
}

st.set_page_config(page_title="AI Adaptive Learning | PTIT", page_icon="🎓", layout="wide")
st.title("🎓 Hệ thống gợi ý lộ trình học tập")
st.caption("AI-Powered Adaptive Learning & Student Performance Prediction | Nguyên mẫu học thuật PTIT")
st.warning(
    "Nguyên mẫu này sử dụng dữ liệu mô phỏng, không phải hồ sơ sinh viên thật. "
    "Kết quả chỉ để minh họa học thuật, không dùng làm căn cứ quyết định học vụ."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

with st.sidebar:
    st.header("Thông tin dự án")
    st.markdown(
        "- **Mô hình:** Random Forest (được chọn theo Macro-F1 trên tập kiểm thử)\n"
        "- **Đầu vào:** 6 chỉ số học tập\n"
        "- **Đầu ra:** 3 mức trình độ\n"
        "- **Dữ liệu:** 1.200 bản ghi mô phỏng"
    )
    st.caption("Nhóm: Hồ Hiểu Thanh · Lê Nhật Linh · La Thế Phong")
    if DATA_PATH.exists():
        st.download_button(
            "Tải bộ dữ liệu mô phỏng CSV",
            data=DATA_PATH.read_bytes(),
            file_name="student_learning_data_synthetic.csv",
            mime="text/csv",
        )

tab_predict, tab_data, tab_about = st.tabs(["Dự đoán & lộ trình", "Khám phá dữ liệu", "Mô tả hệ thống"])

with tab_predict:
    st.subheader("1. Nhập các chỉ số học tập")
    with st.form("student_form"):
        left, right = st.columns(2)
        attendance = left.slider("Chuyên cần (%)", 0, 100, 80)
        study_hours = right.slider("Giờ tự học mỗi tuần", 0.0, 30.0, 8.0, 0.5)
        assignment = left.slider("Hoàn thành bài tập (%)", 0, 100, 75)
        quiz = right.slider("Điểm quiz trung bình (0–100)", 0, 100, 65)
        midterm = left.slider("Điểm giữa kỳ (0–100)", 0, 100, 65)
        engagement = right.slider("Mức độ tham gia (1–5)", 1.0, 5.0, 3.0, 0.5)
        submitted = st.form_submit_button("Phân loại và tạo lộ trình", type="primary", use_container_width=True)

    if submitted:
        sample = pd.DataFrame([{
            "attendance_pct": attendance,
            "weekly_study_hours": study_hours,
            "assignment_completion_pct": assignment,
            "quiz_average": quiz,
            "midterm_score": midterm,
            "engagement_level": engagement,
        }], columns=FEATURES)
        prediction = model.predict(sample)[0]
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]
            classes = model.classes_
        st.subheader("2. Kết quả phân loại")
        c1, c2 = st.columns([1, 2])
        with c1:
            if prediction == "Cần hỗ trợ":
                st.error(f"Mức dự đoán: **{prediction}**")
            elif prediction == "Đạt yêu cầu":
                st.info(f"Mức dự đoán: **{prediction}**")
            else:
                st.success(f"Mức dự đoán: **{prediction}**")
        with c2:
            if probabilities is not None:
                prob_df = pd.DataFrame({"Mức trình độ": classes, "Xác suất mô hình": probabilities})
                st.bar_chart(prob_df.set_index("Mức trình độ"))
                st.caption("Xác suất là đầu ra của mô hình trên dữ liệu mô phỏng, không phải xác suất thành công học tập đã được hiệu chuẩn.")

        recommendations = []
        if attendance < 80:
            recommendations.append(("Chuyên cần", "Đặt mục tiêu chuyên cần ít nhất 85%; kiểm tra lịch học và bù nội dung đã bỏ lỡ."))
        if study_hours < 7:
            recommendations.append(("Tự học", "Lập lịch 4–5 phiên mỗi tuần, tổng khoảng 7–10 giờ; chia phiên học thành các khoảng 25–30 phút."))
        if assignment < 75:
            recommendations.append(("Bài tập", "Hoàn thành bài còn thiếu, xem lại phản hồi và ghi nhật ký lỗi thường gặp sau mỗi lần nộp."))
        if quiz < 65:
            recommendations.append(("Kiến thức nền", "Ôn lại chủ đề có điểm thấp, dùng flashcard và làm quiz ngắn sau mỗi phiên ôn tập."))
        if midterm < 65:
            recommendations.append(("Luyện kiểm tra", "Làm bài theo từng chủ đề trước, sau đó luyện đề có giới hạn thời gian và phân tích câu sai."))
        if engagement < 3:
            recommendations.append(("Tham gia học tập", "Đặt ít nhất một câu hỏi mỗi tuần, tham gia thảo luận hoặc học nhóm để kiểm tra mức độ hiểu."))
        if prediction == "Cần hỗ trợ":
            recommendations.append(("Lộ trình ưu tiên", "Quay lại kiến thức tiên quyết; chia mục tiêu thành các nhiệm vụ nhỏ; trao đổi với giảng viên/trợ giảng và đánh giá lại sau 1 tuần."))
        elif prediction == "Đạt yêu cầu":
            recommendations.append(("Lộ trình củng cố", "Duy trì lịch học hiện tại; ưu tiên 1–2 chỉ số thấp nhất và kiểm tra tiến bộ sau mỗi tuần."))
        else:
            recommendations.append(("Lộ trình nâng cao", "Thực hiện bài tập mở rộng hoặc dự án nhỏ; giải thích lại kiến thức cho bạn học để củng cố mức độ thành thạo."))
        st.subheader("3. Lộ trình học tập đề xuất")
        for title, detail in recommendations:
            st.markdown(f"- **{title}:** {detail}")
        st.download_button(
            "Tải kết quả và lộ trình (.txt)",
            data="\n".join([f"Mức dự đoán: {prediction}", "", "Lộ trình đề xuất:"] + [f"- {t}: {d}" for t, d in recommendations]),
            file_name="lo_trinh_hoc_tap_de_xuat.txt",
            mime="text/plain",
        )

with tab_data:
    st.subheader("Khám phá bộ dữ liệu mô phỏng")
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        st.write(f"Số bản ghi: **{len(df):,}** · Số thuộc tính đầu vào: **{len(FEATURES)}**")
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown("**Phân bố nhãn mục tiêu**")
        st.bar_chart(df["performance_level"].value_counts().rename_axis("Mức trình độ"))
        st.markdown("**Thống kê mô tả**")
        st.dataframe(df[FEATURES].describe().T, use_container_width=True)
        missing = df[FEATURES].isna().sum().rename("Số giá trị thiếu").to_frame()
        st.markdown("**Giá trị thiếu theo thuộc tính**")
        st.dataframe(missing, use_container_width=True)
    else:
        st.info("Không tìm thấy tệp CSV trong thư mục dự án.")

with tab_about:
    st.subheader("Mô tả hệ thống")
    st.markdown(
        """
        **Quy trình xử lý**

        1. Người dùng nhập 6 chỉ số học tập.
        2. Giao diện kiểm tra miền giá trị thông qua các thanh trượt.
        3. Pipeline thay thế giá trị thiếu bằng trung vị và chuẩn hóa dữ liệu.
        4. Bộ phân loại dự đoán một trong ba mức: Cần hỗ trợ, Đạt yêu cầu, Tốt.
        5. Bộ quy tắc minh bạch tạo gợi ý dựa trên chỉ số thấp và mức dự đoán.
        6. Kết quả được hiển thị và có thể tải về.

        **Giới hạn:** Nhãn được tạo theo quy tắc mô phỏng từ các biến đầu vào, do đó kết quả đo lường chủ yếu xác nhận pipeline hoạt động. Cần dữ liệu thực tế có quyền sử dụng, ẩn danh và được giảng viên/đơn vị cho phép trước khi đánh giá khả năng ứng dụng.
        """
    )
