import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score, 
                             roc_curve, make_scorer)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, VotingClassifier,
                              ExtraTreesClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from sklearn.feature_selection import mutual_info_classif

df = pd.read_csv('survey lung cancer.csv')

print("Original dataset shape:", df.shape)
print(df.head())

df.drop_duplicates(inplace=True)
print("After duplicate removal:", df.shape)

df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})

binary_features = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
                   'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 
                   'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 
                   'SWALLOWING DIFFICULTY', 'CHEST PAIN']
                   
for col in binary_features:
    df[col] = df[col] - 1  # converting 1/2 to 0/1

X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

mi = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
print("\nFeature Ranking by Mutual Information (Gain Ratio proxy):")
print(mi_series)

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
rf_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Ranking by Random Forest Importance:")
print(rf_importances)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
mi_series.plot(kind='bar', title='Mutual Information Ranking')
plt.subplot(1,2,2)
rf_importances.plot(kind='bar', title='RF Feature Importance')
plt.tight_layout()
plt.show()

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled['AGE'] = scaler.fit_transform(X[['AGE']])

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)
print("\nAfter SMOTE, class distribution:")
print(y_res.value_counts())

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Gaussian NB': GaussianNB(),
    'Bernoulli NB': BernoulliNB(),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42),
    'Extra Trees': ExtraTreesClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    # Ensemble using Voting (combining XGBoost and AdaBoost as in paper 1)
    'Voting Ensemble': VotingClassifier(estimators=[
                                ('xgb', xgb.XGBClassifier(eval_metric='logloss', random_state=42)),
                                ('ada', AdaBoostClassifier(random_state=42))
                             ], voting='soft'),
    'Multilayer Perceptron': MLPClassifier(max_iter=1000, random_state=42)
}

def evaluate_model(model, X, y):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    metrics = {metric: np.mean(scores[f'test_{metric}']) for metric in scoring}
    return metrics

results = {}
for name, model in models.items():
    print(f"Evaluating: {name}")
    metrics = evaluate_model(model, X_res, y_res)
    results[name] = metrics
    print(metrics)
    print("-"*50)

results_df = pd.DataFrame(results).T.sort_values(by='roc_auc', ascending=False)
print("\nSummary of 10-fold CV results:")
print(results_df)

best_model_name = results_df.index[0]
best_model = models[best_model_name]
print(f"\nBest model selected: {best_model_name}")

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, 
                                                    stratify=y_res, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)[:,1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc_score = roc_auc_score(y_test, y_pred_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {auc_score:.2f})')
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {best_model_name}')
plt.legend(loc='lower right')
plt.show()
