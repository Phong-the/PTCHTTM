"""Generate the synthetic dataset, train the models and export evaluation artifacts.

The script is deterministic (random_state=42) and the committed outputs were
produced with the dependency versions pinned in requirements.txt.
"""
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "student_learning_data_synthetic.csv"
RANDOM_STATE = 42

FEATURES = [
    "attendance_pct",
    "weekly_study_hours",
    "assignment_completion_pct",
    "quiz_average",
    "midterm_score",
    "engagement_level",
]
LABELS = ["Cần hỗ trợ", "Đạt yêu cầu", "Tốt"]

SMOKE_PROFILES = [
    ("Học tập tốt", [95, 15, 95, 90, 92, 5]),
    ("Cần hỗ trợ", [40, 2, 20, 25, 30, 1]),
    ("Trung bình", [75, 7, 70, 65, 66, 3]),
]


def generate_synthetic_dataset(n_rows=1200):
    """Create a reproducible synthetic dataset.

    The labels are derived from the input features through a weighted latent
    score plus noise, so the data only demonstrates the pipeline and must not
    be treated as real student records.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    attendance = np.clip(rng.normal(78, 15, n_rows), 25, 100)
    study = np.clip(rng.gamma(2.3, 3.0, n_rows), 0.5, 25)
    assignment = np.clip(100 * rng.beta(5, 2, n_rows), 5, 100)
    quiz = np.clip(rng.normal(68, 17, n_rows), 0, 100)
    midterm = np.clip(rng.normal(65, 18, n_rows), 0, 100)
    engagement = np.clip(rng.normal(3.1, 1.0, n_rows), 1, 5)

    latent = (
        0.20 * attendance
        + 0.12 * (study / 25 * 100)
        + 0.20 * assignment
        + 0.22 * quiz
        + 0.22 * midterm
        + 0.04 * (engagement / 5 * 100)
        + rng.normal(0, 9, n_rows)
    )
    labels = np.where(latent < 52, LABELS[0], np.where(latent < 72, LABELS[1], LABELS[2]))

    df = pd.DataFrame(
        {
            "attendance_pct": attendance.round(1),
            "weekly_study_hours": study.round(1),
            "assignment_completion_pct": assignment.round(1),
            "quiz_average": quiz.round(1),
            "midterm_score": midterm.round(1),
            "engagement_level": engagement.round(1),
            "performance_level": labels,
        }
    )

    for column in ["weekly_study_hours", "quiz_average", "engagement_level"]:
        missing_rows = rng.choice(n_rows, 8, replace=False)
        df.loc[missing_rows, column] = np.nan

    return df


def build_preprocessor():
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer([("num", numeric_pipeline, FEATURES)])


def build_candidates():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=8,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }


def train_and_compare(X_train, y_train, X_test, y_test):
    summary = {}
    fitted_models = {}
    predictions = {}

    for name, classifier in build_candidates().items():
        model = Pipeline(
            [("preprocess", build_preprocessor()), ("model", classifier)]
        ).fit(X_train, y_train)
        prediction = model.predict(X_test)

        fitted_models[name] = model
        predictions[name] = prediction
        summary[name] = {
            "accuracy": round(accuracy_score(y_test, prediction), 4),
            "precision_macro": round(
                precision_score(y_test, prediction, average="macro", zero_division=0), 4
            ),
            "recall_macro": round(
                recall_score(y_test, prediction, average="macro", zero_division=0), 4
            ),
            "f1_macro": round(
                f1_score(y_test, prediction, average="macro", zero_division=0), 4
            ),
        }

    best_name = max(summary, key=lambda name: summary[name]["f1_macro"])
    return summary, fitted_models, predictions, best_name


def run_smoke_tests(model):
    """Run three representative profiles through the saved pipeline."""
    profiles = pd.DataFrame(
        [values for _, values in SMOKE_PROFILES], columns=FEATURES
    )
    output = model.predict(profiles)
    passed = all(label in LABELS for label in output)

    return {
        "status": "passed" if passed else "failed",
        "tests": len(SMOKE_PROFILES),
        "predictions": [
            {"profile": name, "prediction": str(prediction)}
            for (name, _), prediction in zip(SMOKE_PROFILES, output)
        ],
    }


def save_evaluation_artifacts(df, X_test, y_test, summary, predictions, best_name):
    metrics = {
        "dataset_note": "Synthetic data; not real student records",
        "random_state": RANDOM_STATE,
        "n_rows": len(df),
        "train_rows": len(df) - len(X_test),
        "test_rows": len(X_test),
        "class_distribution": df["performance_level"].value_counts().to_dict(),
        "models": summary,
        "selected_model": best_name,
    }
    with open(OUTPUT_DIR / "model_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    comparison = pd.DataFrame(
        [{"model": name, **scores} for name, scores in summary.items()]
    )
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    report = classification_report(
        y_test, predictions[best_name], output_dict=True, zero_division=0
    )
    pd.DataFrame(report).T.to_csv(OUTPUT_DIR / "classification_report.csv")

    prediction_output = X_test.copy()
    prediction_output["actual_level"] = y_test
    prediction_output["predicted_level"] = predictions[best_name]
    prediction_output.to_csv(
        OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )

    df[FEATURES].describe().T.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

    plot_class_distribution(df)
    plot_midterm_distribution(df)
    plot_model_comparison(summary)
    plot_confusion_matrix(y_test, predictions[best_name], best_name)


def plot_class_distribution(df):
    plt.figure(figsize=(8, 5))
    df["performance_level"].value_counts().reindex(LABELS).plot(kind="bar")
    plt.title("Phân bố nhãn trình độ (dữ liệu mô phỏng)")
    plt.xlabel("Mức trình độ")
    plt.ylabel("Số bản ghi")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=160)
    plt.close()


def plot_midterm_distribution(df):
    plt.figure(figsize=(8, 5))
    df["midterm_score"].dropna().plot(kind="hist", bins=20)
    plt.title("Phân bố điểm giữa kỳ (dữ liệu mô phỏng)")
    plt.xlabel("Điểm giữa kỳ")
    plt.ylabel("Tần suất")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "midterm_distribution.png", dpi=160)
    plt.close()


def plot_model_comparison(summary):
    metrics = pd.DataFrame(summary).T
    plt.figure(figsize=(9, 5))
    metrics[["accuracy", "f1_macro"]].plot(kind="bar")
    plt.title("So sánh Accuracy và Macro-F1")
    plt.xlabel("Mô hình")
    plt.ylabel("Điểm số")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=160)
    plt.close()


def plot_confusion_matrix(y_test, prediction, best_name):
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        prediction,
        labels=LABELS,
        display_labels=LABELS,
        ax=axis,
        xticks_rotation=20,
        colorbar=False,
    )
    axis.set_title(f"Ma trận nhầm lẫn — {best_name}")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    df = generate_synthetic_dataset()
    df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df["performance_level"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["performance_level"],
    )

    summary, fitted_models, predictions, best_name = train_and_compare(
        X_train, y_train, X_test, y_test
    )
    best_model = fitted_models[best_name]
    joblib.dump(best_model, MODEL_DIR / "performance_classifier.joblib")

    save_evaluation_artifacts(
        df, X_test, y_test, summary, predictions, best_name
    )

    smoke_results = run_smoke_tests(best_model)
    with open(OUTPUT_DIR / "smoke_test_results.json", "w", encoding="utf-8") as file:
        json.dump(smoke_results, file, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "selected_model": best_name,
                "n_rows": len(df),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "metrics": summary,
                "smoke_tests": smoke_results["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
