"""Reproducible synthetic-data generation, model comparison and evaluation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

ROOT=Path(__file__).parent
OUT=ROOT/"outputs"; MODELS=ROOT/"models"
OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
RANDOM_STATE=42
FEATURES=["attendance_pct","weekly_study_hours","assignment_completion_pct","quiz_average","midterm_score","engagement_level"]
LABELS=["Cần hỗ trợ","Đạt yêu cầu","Tốt"]

# Synthetic data only: values and labels are generated for an academic prototype.
rng=np.random.default_rng(RANDOM_STATE); n=1200
attendance=np.clip(rng.normal(78,15,n),25,100)
study=np.clip(rng.gamma(2.3,3.0,n),0.5,25)
assignment=np.clip(100*rng.beta(5,2,n),5,100)
quiz=np.clip(rng.normal(68,17,n),0,100)
midterm=np.clip(rng.normal(65,18,n),0,100)
engagement=np.clip(rng.normal(3.1,1.0,n),1,5)
latent=(.20*attendance+.12*(study/25*100)+.20*assignment+.22*quiz+
        .22*midterm+.04*(engagement/5*100)+rng.normal(0,9,n))
labels=np.where(latent<52,LABELS[0],np.where(latent<72,LABELS[1],LABELS[2]))
df=pd.DataFrame({"attendance_pct":attendance.round(1),"weekly_study_hours":study.round(1),
 "assignment_completion_pct":assignment.round(1),"quiz_average":quiz.round(1),
 "midterm_score":midterm.round(1),"engagement_level":engagement.round(1),"performance_level":labels})
for col in ["weekly_study_hours","quiz_average","engagement_level"]:
    df.loc[rng.choice(n,8,replace=False),col]=np.nan
df.to_csv(ROOT/"student_learning_data_synthetic.csv",index=False,encoding="utf-8-sig")
X_train,X_test,y_train,y_test=train_test_split(df[FEATURES],df["performance_level"],test_size=.2,random_state=RANDOM_STATE,stratify=df["performance_level"])

def make_preprocessor():
    return ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),FEATURES)])

candidates={"Logistic Regression":LogisticRegression(max_iter=2000,class_weight="balanced"),
 "Decision Tree":DecisionTreeClassifier(max_depth=6,min_samples_leaf=8,random_state=RANDOM_STATE,class_weight="balanced"),
 "Random Forest":RandomForestClassifier(n_estimators=250,max_depth=10,min_samples_leaf=4,random_state=RANDOM_STATE,class_weight="balanced")}
summary={}; fitted={}; predictions={}
for name,clf in candidates.items():
    model=Pipeline([("preprocess",make_preprocessor()),("model",clf)]).fit(X_train,y_train)
    pred=model.predict(X_test); fitted[name]=model; predictions[name]=pred
    summary[name]={"accuracy":round(accuracy_score(y_test,pred),4),
      "precision_macro":round(precision_score(y_test,pred,average="macro",zero_division=0),4),
      "recall_macro":round(recall_score(y_test,pred,average="macro",zero_division=0),4),
      "f1_macro":round(f1_score(y_test,pred,average="macro",zero_division=0),4)}
best_name=max(summary,key=lambda x:summary[x]["f1_macro"])
best=fitted[best_name]; joblib.dump(best,MODELS/"performance_classifier.joblib")
with open(OUT/"model_metrics.json","w",encoding="utf-8") as f:
    json.dump({"dataset_note":"Synthetic data; not real student records","random_state":RANDOM_STATE,"n_rows":len(df),
      "train_rows":len(X_train),"test_rows":len(X_test),"class_distribution":df["performance_level"].value_counts().to_dict(),
      "models":summary,"selected_model":best_name},f,ensure_ascii=False,indent=2)
pd.DataFrame([{"model":name,**scores} for name,scores in summary.items()]).to_csv(OUT/"model_comparison.csv",index=False)
pd.DataFrame(classification_report(y_test,predictions[best_name],output_dict=True,zero_division=0)).T.to_csv(OUT/"classification_report.csv")
predout=X_test.copy(); predout["actual_level"]=y_test; predout["predicted_level"]=predictions[best_name]
predout.to_csv(OUT/"test_predictions.csv",index=False,encoding="utf-8-sig")
df[FEATURES].describe().T.to_csv(OUT/"descriptive_statistics.csv")

# Generate charts reproducibly; do not set custom colors so Matplotlib defaults are used.
plt.figure(figsize=(8,5)); df["performance_level"].value_counts().reindex(LABELS).plot(kind="bar")
plt.title("Phân bố nhãn trình độ (dữ liệu mô phỏng)"); plt.xlabel("Mức trình độ"); plt.ylabel("Số bản ghi"); plt.tight_layout(); plt.savefig(OUT/"class_distribution.png",dpi=160); plt.close()
plt.figure(figsize=(8,5)); df["midterm_score"].dropna().plot(kind="hist",bins=20)
plt.title("Phân bố điểm giữa kỳ (dữ liệu mô phỏng)"); plt.xlabel("Điểm giữa kỳ"); plt.ylabel("Tần suất"); plt.tight_layout(); plt.savefig(OUT/"midterm_distribution.png",dpi=160); plt.close()
metrics_df=pd.DataFrame(summary).T
plt.figure(figsize=(9,5)); metrics_df[["accuracy","f1_macro"]].plot(kind="bar")
plt.title("So sánh Accuracy và Macro-F1"); plt.xlabel("Mô hình"); plt.ylabel("Điểm số"); plt.ylim(0,1); plt.tight_layout(); plt.savefig(OUT/"model_comparison.png",dpi=160); plt.close()
fig,ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay.from_predictions(y_test,predictions[best_name],labels=LABELS,display_labels=LABELS,ax=ax,xticks_rotation=20,colorbar=False)
ax.set_title(f"Ma trận nhầm lẫn — {best_name}"); fig.tight_layout(); fig.savefig(OUT/"confusion_matrix.png",dpi=160); plt.close(fig)

print(json.dumps({"selected_model":best_name,"n_rows":len(df),"train_rows":len(X_train),"test_rows":len(X_test),"metrics":summary},ensure_ascii=False,indent=2))
