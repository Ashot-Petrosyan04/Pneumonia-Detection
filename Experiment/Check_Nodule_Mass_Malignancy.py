import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

df_participant = pd.read_csv("nlst_780_prsn_idc_20210527.csv").drop(columns=['dataset_version'])
df_lung = pd.read_csv("nlst_780_canc_idc_20210527.csv").drop(columns=['dataset_version'])
df_abnormalities = pd.read_csv("nlst_780_ctab_idc_20210527.csv").drop(columns=['dataset_version'])

df_participant = df_participant.dropna(subset=['age', 'gender', "cigsmok", "scr_res0", "scr_res1", "scr_res2", "lesionsize"])
df_lung = df_lung.drop(columns=['lesionsize', 'de_type', 'de_grade', 'de_stag', 'candx_days', 'de_stag_7thed'])
df_abnormalities = df_abnormalities.dropna(subset=['sct_ab_desc', 'sct_long_dia', "sct_margins"])

df = pd.merge(df_lung, df_participant, on="pid", how="inner")
df = pd.merge(df, df_abnormalities, on=["pid", "study_yr"], how="inner")

df["cancer"] = df["can_scr"].apply(lambda x: 1 if x in {1, 3, 4} else 0)

features = ["age", "gender", "cigsmok", "scr_res0", "scr_res1", "scr_res2", "sct_ab_desc", "sct_long_dia", "sct_margins", "lesionsize"]
df = df[features + ["cancer"]]
df = df.dropna()

X = df.drop("cancer", axis=1)
y = df["cancer"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

class_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

models = {
    "XGBoost": XGBClassifier(
        scale_pos_weight=class_ratio,
        eval_metric='logloss',
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    
    results[name] = {
        "AUC": roc_auc_score(y_test, y_proba),
        "Classification Report": classification_report(y_test, y_pred)
    }

for model_name, metrics in results.items():
    print(f"\n{model_name} Performance:")
    print(f"AUC: {metrics['AUC']:.3f}")
    print(metrics['Classification Report'])
