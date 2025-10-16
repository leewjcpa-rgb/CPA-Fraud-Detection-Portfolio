import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score

# --- 1. 데이터 불러오기 및 준비 ---
file_path = "data/shinsegae_bank_statement_For_ML.csv"
df = pd.read_csv(file_path)

X = df.drop(columns=['Date', 'Client_ID', 'Transaction_Description', 'Transaction_Info', 'Balance'])
y = (df['Transaction_Info'] != 'Normal').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("="*60)
print("              <부정 거래 탐지 모델 성능 비교 분석>")
print("="*60)
print(f"학습용 데이터: {len(X_train)}건, 검증용 데이터: {len(X_test)}건")


# ===================================================================================
# 모델 1: 스태킹 AI 모델 (AI 탐정)
# ===================================================================================
print("\n\n--- [모델 1] 스태킹 AI 모델 학습 및 평가 ---")
estimators = [
    ('random_forest', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ('gradient_boosting', GradientBoostingClassifier(n_estimators=100, random_state=42))
]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)

stacking_model.fit(X_train, y_train)
y_pred_stacking = stacking_model.predict(X_test)

print("\n--- 스태킹 AI 모델 성능 평가 리포트 ---")
print(classification_report(y_test, y_pred_stacking, target_names=['정상(0)', '부정(1)']))


# ===================================================================================
# 모델 2: [개선된] 규칙 기반 위험 점수 모델 (엘리트 인간 탐정)
# ===================================================================================
print("\n\n--- [모델 2] 개선된 규칙 기반 모델 평가 ---")

X_test_rules = X_test.copy()
X_test_rules['Risk_Score'] = 0

# [수정!] 훨씬 더 정교해진 차등 점수 비중을 적용합니다.
X_test_rules.loc[X_test_rules['Amount_vs_Avg'] > 10, 'Risk_Score'] += 30
X_test_rules.loc[(X_test_rules['Amount_vs_Avg'] > 5) & (X_test_rules['Amount_vs_Avg'] <= 10), 'Risk_Score'] += 20
X_test_rules.loc[(X_test_rules['Amount_vs_Avg'] > 3) & (X_test_rules['Amount_vs_Avg'] <= 5), 'Risk_Score'] += 10

X_test_rules.loc[X_test_rules['Time_Delta_Seconds'] <= 600, 'Risk_Score'] += 20 
X_test_rules.loc[X_test_rules['Is_OffHours'] == 1, 'Risk_Score'] += 30     

# 임계점(Threshold) 설정: 50점 이상이면 '부정 거래'로 판단
threshold = 50
y_pred_rules = (X_test_rules['Risk_Score'] >= threshold).astype(int)

print(f"\n(규칙: 위험 점수가 {threshold}점 이상일 경우 '부정'으로 판단)")
print("\n--- 개선된 규칙 기반 모델 성능 평가 리포트 ---")
print(classification_report(y_test, y_pred_rules, target_names=['정상(0)', '부정(1)']))


# ===================================================================================
# 최종 비교 분석
# ===================================================================================
print("\n\n" + "="*60)
print("                 <최종 모델 성능 비교표>")
print("="*60)

stacking_recall = recall_score(y_test, y_pred_stacking)
stacking_precision = precision_score(y_test, y_pred_stacking)
stacking_accuracy = accuracy_score(y_test, y_pred_stacking)

rules_recall = recall_score(y_test, y_pred_rules, zero_division=0)
rules_precision = precision_score(y_test, y_pred_rules, zero_division=0)
rules_accuracy = accuracy_score(y_test, y_pred_rules)

comparison_df = pd.DataFrame({
    "성능 지표": ["Recall (부정거래 탐지율)", "Precision (탐지의 정확성)", "Accuracy (전체 정확도)"],
    "규칙 기반 모델 (인간 탐정)": [f"{rules_recall:.2%}", f"{rules_precision:.2%}", f"{rules_accuracy:.2%}"],
    "스태킹 AI 모델 (AI 탐정)": [f"{stacking_recall:.2%}", f"{stacking_precision:.2%}", f"{stacking_accuracy:.2%}"]
})

print(comparison_df.to_string(index=False))
print("\n* Recall (부정거래 탐지율): 실제 부정 거래 100건 중 몇 건을 잡아냈는가? (가장 중요)")
print("* Precision (탐지의 정확성): '부정'이라고 예측한 100건 중 실제 부정이 몇 건인가?")
print("="*60)