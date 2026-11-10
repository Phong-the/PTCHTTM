import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "performance_classifier.joblib"
DATA_PATH = ROOT / "student_learning_data_synthetic.csv"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
COMPARISON_PATH = ROOT / "outputs" / "model_comparison.png"
CONFUSION_PATH = ROOT / "outputs" / "confusion_matrix.png"
ARCHITECTURE_PATH = ROOT / "outputs" / "system_architecture.png"

FEATURES = [
    "attendance_pct",
    "weekly_study_hours",
    "assignment_completion_pct",
    "quiz_average",
    "midterm_score",
    "engagement_level",
]
FEATURE_LABELS = {
    "attendance_pct": "Chuyên cần (%)",
    "weekly_study_hours": "Giờ tự học/tuần",
    "assignment_completion_pct": "Hoàn thành bài tập (%)",
    "quiz_average": "Điểm quiz trung bình",
    "midterm_score": "Điểm giữa kỳ",
    "engagement_level": "Mức độ tham gia (1–5)",
    "performance_level": "Mức trình độ",
}
LABEL_ORDER = ["Cần hỗ trợ", "Đạt yêu cầu", "Tốt"]
LABEL_COLORS = ["#C0392B", "#E0A526", "#2E7D5B"]
TEAM = "Hồ Hiểu Thanh · Lê Nhật Linh · La Thế Phong"

DEFAULT_MODEL_NAME = "Random Forest"
DEFAULT_METRICS = {
    "accuracy": 0.6292,
    "precision_macro": 0.5525,
    "recall_macro": 0.5556,
    "f1_macro": 0.5494,
}
DEFAULT_N_ROWS = 1200

st.set_page_config(
    page_title="AI Adaptive Learning | PTIT",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.session_state.setdefault("prediction_result", None)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_metrics(path: str) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def reset_result() -> None:
    st.session_state.prediction_result = None


def format_percent(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}%}".replace(".", ",")


def format_integer(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def build_recommendations(values: dict, prediction: str) -> list[tuple[str, str]]:
    recommendations: list[tuple[str, str]] = []
    if values["attendance_pct"] < 80:
        recommendations.append(
            (
                "Chuyên cần",
                "Đặt mục tiêu chuyên cần ít nhất 85%; kiểm tra lịch học và bù nội dung đã bỏ lỡ.",
            )
        )
    if values["weekly_study_hours"] < 7:
        recommendations.append(
            (
                "Tự học",
                "Lập lịch 4–5 phiên mỗi tuần, tổng khoảng 7–10 giờ; chia phiên học thành các khoảng 25–30 phút.",
            )
        )
    if values["assignment_completion_pct"] < 75:
        recommendations.append(
            (
                "Bài tập",
                "Hoàn thành bài còn thiếu, xem lại phản hồi và ghi nhật ký lỗi thường gặp sau mỗi lần nộp.",
            )
        )
    if values["quiz_average"] < 65:
        recommendations.append(
            (
                "Kiến thức nền",
                "Ôn lại chủ đề có điểm thấp, dùng flashcard và làm quiz ngắn sau mỗi phiên ôn tập.",
            )
        )
    if values["midterm_score"] < 65:
        recommendations.append(
            (
                "Luyện kiểm tra",
                "Làm bài theo từng chủ đề trước, sau đó luyện đề có giới hạn thời gian và phân tích câu sai.",
            )
        )
    if values["engagement_level"] < 3:
        recommendations.append(
            (
                "Tham gia học tập",
                "Đặt ít nhất một câu hỏi mỗi tuần, tham gia thảo luận hoặc học nhóm để kiểm tra mức độ hiểu.",
            )
        )
    if prediction == "Cần hỗ trợ":
        recommendations.append(
            (
                "Lộ trình ưu tiên",
                "Quay lại kiến thức tiên quyết; chia mục tiêu thành các nhiệm vụ nhỏ; trao đổi với giảng viên/trợ giảng và đánh giá lại sau 1 tuần.",
            )
        )
    elif prediction == "Đạt yêu cầu":
        recommendations.append(
            (
                "Lộ trình củng cố",
                "Duy trì lịch học hiện tại; ưu tiên 1–2 chỉ số thấp nhất và kiểm tra tiến bộ sau mỗi tuần.",
            )
        )
    else:
        recommendations.append(
            (
                "Lộ trình nâng cao",
                "Thực hiện bài tập mở rộng hoặc dự án nhỏ; giải thích lại kiến thức cho bạn học để củng cố mức độ thành thạo.",
            )
        )
    return recommendations


def build_export_text(result: dict) -> str:
    lines = [
        "LỘ TRÌNH HỌC TẬP ĐỀ XUẤT",
        "",
        f"Mức dự đoán: {result['prediction']}",
        "",
        "Chỉ số đầu vào:",
        f"- Chuyên cần: {result['input']['attendance_pct']}%",
        f"- Giờ tự học mỗi tuần: {result['input']['weekly_study_hours']} giờ",
        f"- Hoàn thành bài tập: {result['input']['assignment_completion_pct']}%",
        f"- Điểm quiz trung bình: {result['input']['quiz_average']}",
        f"- Điểm giữa kỳ: {result['input']['midterm_score']}",
        f"- Mức độ tham gia: {result['input']['engagement_level']}/5",
        "",
        "Lộ trình đề xuất:",
    ]
    lines.extend(f"- {title}: {detail}" for title, detail in result["recommendations"])
    return "\n".join(lines)


try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(
        "Không thể tải mô hình phân loại. Vui lòng kiểm tra tệp mô hình trong thư mục dự án.",
        title="Lỗi khởi tạo",
        icon=":material/error:",
    )
    st.caption(f"Chi tiết kỹ thuật: {exc}")
    st.stop()

selected_model = DEFAULT_MODEL_NAME
model_metrics = DEFAULT_METRICS
n_rows = DEFAULT_N_ROWS
if METRICS_PATH.exists():
    try:
        metrics = load_metrics(str(METRICS_PATH))
        selected_model = metrics.get("selected_model", DEFAULT_MODEL_NAME)
        model_metrics = metrics.get("models", {}).get(selected_model, DEFAULT_METRICS)
        n_rows = int(metrics.get("n_rows", DEFAULT_N_ROWS))
    except Exception:
        pass

accuracy = float(model_metrics.get("accuracy", 0.0))
macro_f1 = float(model_metrics.get("f1_macro", 0.0))

df = None
data_error = None
if DATA_PATH.exists():
    try:
        df = load_dataset(str(DATA_PATH))
    except Exception as exc:
        data_error = str(exc)

with st.sidebar:
    st.markdown("### AI Adaptive Learning")
    st.caption("Đồ án môn Phát triển các hệ thống thông minh · PTIT")
    st.space("small")
    st.markdown("**Mô hình đang sử dụng**")
    st.markdown(f":material/model_training: {selected_model}")
    st.caption(
        f"Accuracy {format_percent(accuracy)} · Macro-F1 {format_percent(macro_f1)} trên tập kiểm thử"
    )
    st.space("small")
    st.markdown("**Phạm vi hệ thống**")
    st.markdown(
        "- Đầu vào: 6 chỉ số học tập\n"
        "- Đầu ra: 3 mức trình độ\n"
        f"- Dữ liệu: {format_integer(n_rows)} bản ghi mô phỏng"
    )
    if DATA_PATH.exists():
        st.download_button(
            "Tải dữ liệu mô phỏng",
            data=DATA_PATH.read_bytes(),
            file_name="student_learning_data_synthetic.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )
    st.space("small")
    st.caption(f"Nhóm thực hiện: {TEAM}")

st.title("Hệ thống gợi ý lộ trình học tập", icon=":material/school:")
st.caption("AI-powered adaptive learning · Nguyên mẫu học thuật PTIT")

with st.container(horizontal=True):
    st.metric("Mô hình", selected_model, border=True)
    st.metric("Accuracy", format_percent(accuracy), border=True)
    st.metric("Macro-F1", format_percent(macro_f1), border=True)
    st.metric("Dữ liệu huấn luyện", f"{format_integer(n_rows)} bản ghi", border=True)

st.warning(
    "Nguyên mẫu sử dụng dữ liệu mô phỏng, không phải hồ sơ sinh viên thật. "
    "Kết quả chỉ dùng để minh họa học thuật, không làm căn cứ quyết định học vụ.",
    title="Lưu ý sử dụng",
    icon=":material/info:",
)

tab_predict, tab_data, tab_about = st.tabs(
    ["Dự đoán & lộ trình", "Khám phá dữ liệu", "Mô tả hệ thống"]
)

with tab_predict:
    st.subheader("Hồ sơ học tập", icon=":material/person:")
    st.caption("Điều chỉnh các chỉ số bên dưới, sau đó chọn “Phân loại và tạo lộ trình”.")

    with st.form("student_form", border=True):
        left, right = st.columns(2, gap="large")
        with left:
            attendance = st.slider(
                "Chuyên cần (%)",
                0,
                100,
                80,
                help="Tỷ lệ buổi học có mặt trên tổng số buổi.",
            )
            assignment = st.slider(
                "Hoàn thành bài tập (%)",
                0,
                100,
                75,
                help="Tỷ lệ bài tập đã hoàn thành.",
            )
            midterm = st.slider(
                "Điểm giữa kỳ (0–100)",
                0,
                100,
                65,
                help="Điểm bài kiểm tra giữa kỳ.",
            )
        with right:
            study_hours = st.slider(
                "Giờ tự học mỗi tuần",
                0.0,
                30.0,
                8.0,
                0.5,
                help="Số giờ tự học ngoài giờ lên lớp mỗi tuần.",
            )
            quiz = st.slider(
                "Điểm quiz trung bình (0–100)",
                0,
                100,
                65,
                help="Điểm trung bình các bài quiz gần đây.",
            )
            engagement = st.slider(
                "Mức độ tham gia (1–5)",
                1.0,
                5.0,
                3.0,
                0.5,
                help="Mức độ tham gia lớp học, thang 1–5.",
            )
        submitted = st.form_submit_button(
            "Phân loại và tạo lộ trình",
            type="primary",
            icon=":material/analytics:",
            width="stretch",
        )

    if submitted:
        values = {
            "attendance_pct": attendance,
            "weekly_study_hours": study_hours,
            "assignment_completion_pct": assignment,
            "quiz_average": quiz,
            "midterm_score": midterm,
            "engagement_level": engagement,
        }
        sample = pd.DataFrame([values], columns=FEATURES)
        try:
            prediction = str(model.predict(sample)[0])
            probabilities = None
            classes = []
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(sample)[0]
                classes = list(model.classes_)
            st.session_state.prediction_result = {
                "prediction": prediction,
                "probabilities": probabilities,
                "classes": classes,
                "recommendations": build_recommendations(values, prediction),
                "input": values,
            }
        except Exception as exc:
            st.error(
                "Không thể tạo dự đoán cho hồ sơ vừa nhập.",
                title="Lỗi dự đoán",
                icon=":material/error:",
            )
            st.caption(f"Chi tiết kỹ thuật: {exc}")

    result = st.session_state.prediction_result
    if result:
        with st.container(horizontal=True, horizontal_alignment="right"):
            st.button(
                "Xóa kết quả",
                icon=":material/restart_alt:",
                on_click=reset_result,
            )

        st.subheader("Kết quả phân loại", icon=":material/insights:")
        st.caption("Kết quả tương ứng với lần gửi hồ sơ gần nhất.")
        with st.container(border=True):
            result_col, chart_col = st.columns([0.9, 1.6], gap="large")
            with result_col:
                st.metric("Mức dự đoán", result["prediction"])
                badge_color = {
                    "Cần hỗ trợ": "red",
                    "Đạt yêu cầu": "orange",
                    "Tốt": "green",
                }.get(result["prediction"], "blue")
                badge_label = {
                    "Cần hỗ trợ": "Ưu tiên can thiệp",
                    "Đạt yêu cầu": "Theo dõi định kỳ",
                    "Tốt": "Duy trì và mở rộng",
                }.get(result["prediction"], "Theo dõi")
                st.badge(
                    badge_label,
                    icon=":material/flag:",
                    color=badge_color,
                )
                st.caption(
                    "Kết quả từ mô hình trên dữ liệu mô phỏng; không phải kết luận học vụ."
                )
            with chart_col:
                if result["probabilities"] is not None:
                    prob_df = pd.DataFrame(
                        {
                            "Mức trình độ": pd.Categorical(
                                result["classes"],
                                categories=LABEL_ORDER,
                                ordered=True,
                            ),
                            "Xác suất": result["probabilities"],
                        }
                    ).sort_values("Mức trình độ")
                    chart = (
                        alt.Chart(prob_df)
                        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                        .encode(
                            x=alt.X(
                                "Mức trình độ:N",
                                sort=LABEL_ORDER,
                                title=None,
                                axis=alt.Axis(labelAngle=0),
                            ),
                            y=alt.Y(
                                "Xác suất:Q",
                                title=None,
                                axis=alt.Axis(format="%", grid=True),
                            ),
                            color=alt.Color(
                                "Mức trình độ:N",
                                legend=None,
                                scale=alt.Scale(
                                    domain=LABEL_ORDER,
                                    range=LABEL_COLORS,
                                ),
                            ),
                            tooltip=[
                                alt.Tooltip("Mức trình độ:N", title="Mức trình độ"),
                                alt.Tooltip("Xác suất:Q", format=".1%", title="Xác suất"),
                            ],
                        )
                        .properties(height=210)
                    )
                    st.altair_chart(
                        chart,
                        width="stretch",
                        alt="Biểu đồ xác suất dự đoán cho ba mức trình độ",
                    )
                    st.caption(
                        "Xác suất là đầu ra của mô hình, chưa được hiệu chuẩn cho quyết định học vụ."
                    )
                else:
                    st.info(
                        "Mô hình hiện tại không cung cấp xác suất dự đoán.",
                        icon=":material/info:",
                    )

        st.subheader("Lộ trình học tập đề xuất", icon=":material/route:")
        with st.container(border=True):
            st.markdown("**Hành động ưu tiên**")
            for title, detail in result["recommendations"]:
                st.markdown(f"- **{title}:** {detail}")

        st.download_button(
            "Tải lộ trình (.txt)",
            data=build_export_text(result),
            file_name="lo_trinh_hoc_tap_de_xuat.txt",
            mime="text/plain",
            icon=":material/download:",
        )

with tab_data:
    st.subheader("Khám phá bộ dữ liệu mô phỏng", icon=":material/database:")
    if df is None:
        st.error(
            "Không thể đọc bộ dữ liệu mô phỏng trong thư mục dự án.",
            title="Thiếu dữ liệu",
            icon=":material/error:",
        )
        if data_error:
            st.caption(f"Chi tiết kỹ thuật: {data_error}")
    else:
        with st.container(horizontal=True):
            st.metric("Bản ghi", format_integer(len(df)), border=True)
            st.metric("Thuộc tính đầu vào", len(FEATURES), border=True)
            st.metric("Lớp mục tiêu", df["performance_level"].nunique(), border=True)
            st.metric(
                "Giá trị thiếu",
                int(df[FEATURES].isna().sum().sum()),
                border=True,
            )

        with st.container(border=True):
            st.subheader("20 bản ghi đầu", icon=":material/table_chart:")
            st.dataframe(
                df.head(20).rename(columns=FEATURE_LABELS),
                width="stretch",
                hide_index=True,
                alt="20 bản ghi đầu của bộ dữ liệu mô phỏng",
            )

        label_counts = (
            df["performance_level"]
            .value_counts()
            .reindex(LABEL_ORDER, fill_value=0)
            .rename_axis("Mức trình độ")
            .reset_index(name="Số bản ghi")
        )
        with st.container(border=True):
            st.subheader("Phân bố nhãn mục tiêu", icon=":material/bar_chart:")
            st.bar_chart(
                label_counts,
                x="Mức trình độ",
                y="Số bản ghi",
                sort=False,
                alt="Phân bố nhãn mục tiêu trong bộ dữ liệu mô phỏng",
            )

        with st.container(border=True):
            st.subheader("Thống kê mô tả", icon=":material/functions:")
            st.dataframe(
                df[FEATURES].describe().T.rename(index=FEATURE_LABELS),
                width="stretch",
                alt="Thống kê mô tả của sáu chỉ số đầu vào",
            )

        missing = (
            df[FEATURES]
            .isna()
            .sum()
            .rename("Số giá trị thiếu")
            .to_frame()
            .rename(index=FEATURE_LABELS)
        )
        with st.container(border=True):
            st.subheader("Giá trị thiếu theo thuộc tính", icon=":material/rule:")
            st.dataframe(
                missing,
                width="stretch",
                alt="Số lượng giá trị thiếu theo từng thuộc tính",
            )

with tab_about:
    st.subheader("Kiến trúc và phạm vi hệ thống", icon=":material/account_tree:")

    if ARCHITECTURE_PATH.exists():
        with st.container(border=True):
            st.subheader("Kiến trúc logic", icon=":material/schema:")
            st.image(
                str(ARCHITECTURE_PATH),
                width="stretch",
                alt="Sơ đồ kiến trúc logic của hệ thống gợi ý lộ trình học tập",
            )

    with st.container(border=True):
        st.subheader("Quy trình xử lý", icon=":material/route:")
        st.markdown(
            "1. Người dùng nhập 6 chỉ số học tập.\n"
            "2. Giao diện giới hạn miền giá trị bằng thanh trượt.\n"
            "3. Pipeline thay thế giá trị thiếu bằng trung vị và chuẩn hóa dữ liệu.\n"
            "4. Mô hình phân loại dự đoán một trong ba mức trình độ.\n"
            "5. Bộ quy tắc minh bạch tạo gợi ý theo chỉ số thấp và mức dự đoán.\n"
            "6. Kết quả được hiển thị, lưu trong phiên và có thể tải về."
        )

    performance_col, ethics_col = st.columns(2, gap="large")
    with performance_col:
        with st.container(border=True, height="stretch"):
            st.subheader("Hiệu năng mô hình", icon=":material/query_stats:")
            st.metric("Accuracy", format_percent(accuracy))
            st.metric("Macro-F1", format_percent(macro_f1))
            st.caption("Đánh giá trên tập kiểm thử 240 mẫu, không dùng để hiệu chuẩn quyết định học vụ.")
            if COMPARISON_PATH.exists():
                st.image(
                    str(COMPARISON_PATH),
                    width="stretch",
                    alt="So sánh hiệu năng ba mô hình phân loại",
                )
    with ethics_col:
        with st.container(border=True, height="stretch"):
            st.subheader("Giới hạn và đạo đức", icon=":material/balance:")
            st.warning(
                "Nhãn được tạo theo quy tắc mô phỏng từ chính các biến đầu vào, "
                "nên kết quả chủ yếu xác nhận pipeline hoạt động.",
                title="Dữ liệu mô phỏng",
                icon=":material/science:",
            )
            st.markdown(
                "- Không sử dụng họ tên, MSSV, email hoặc dữ liệu định danh.\n"
                "- Không dùng kết quả để tự động quyết định học vụ.\n"
                "- Chỉ triển khai với dữ liệu thật khi có quyền sử dụng, ẩn danh và phê duyệt.\n"
                "- Cần kiểm tra sai lệch giữa các nhóm sinh viên trước khi áp dụng thực tế."
            )

    if CONFUSION_PATH.exists():
        with st.container(border=True):
            st.subheader("Ma trận nhầm lẫn", icon=":material/grid_on:")
            st.image(
                str(CONFUSION_PATH),
                width="stretch",
                alt="Ma trận nhầm lẫn của mô hình Random Forest trên tập kiểm thử",
            )

    st.caption(
        f"Đồ án môn Phát triển các hệ thống thông minh · PTIT · Nhóm: {TEAM}"
    )
