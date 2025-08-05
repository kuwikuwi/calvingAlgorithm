import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import Bootstrap
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

# 1) 변수 정의
keep_cols = ['par','age','m4_act_diff','m4_rum_diff','stdrum','TF']

# 2) 데이터 로드
dfs = [
    pd.read_excel(f'./트레이닝데이터/training_fold_{i}.xlsx', usecols=keep_cols)
    for i in range(1,6)
]
df_all = pd.concat(dfs, ignore_index=True)
X = df_all.drop(columns=['TF'])
y = df_all['TF']

# 3) 부트스트랩 설정
n_splits = 100         # 재표집 횟수
bs = Bootstrap(
    n_splits=n_splits,
    train_size=0.8,    # 80% 재표집
    random_state=42
)

# 4) 결과 저장용
metrics = []

# 5) Bootstrap CV 루프
for fold_idx, (train_idx, test_idx) in enumerate(bs.split(X, y), start=1):
    # 재표집(train), OOB(test)
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[test_idx],  y.iloc[test_idx]
    
    # 5-1) SMOTE 적용
    X_tr_bal, y_tr_bal = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
    
    # 5-2) 모델 정의 & 학습
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=20,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    model.fit(X_tr_bal, y_tr_bal)
    
    # 5-3) 예측 & 평가지표 계산
    y_pred = model.predict(X_va)
    y_prob = model.predict_proba(X_va)[:,1]
    tn, fp, fn, tp = confusion_matrix(y_va, y_pred).ravel()
    
    metrics.append({
        'fold': fold_idx,
        'sensitivity': tp/(tp+fn) if tp+fn>0 else np.nan,
        'specificity': tn/(tn+fp) if tn+fp>0 else np.nan,
        'ppv': tp/(tp+fp) if tp+fp>0 else np.nan,
        'npv': tn/(tn+fn) if tn+fn>0 else np.nan,
        'f1': 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else np.nan,
        'auc': roc_auc_score(y_va, y_prob)
    })

# 6) DataFrame으로 정리
metrics_df = pd.DataFrame(metrics)
summary = metrics_df.drop(columns=['fold']).agg(['mean','std']).T
summary.columns = ['mean','std']

# 7) Excel로 저장
with pd.ExcelWriter("bootstrap_cv_results.xlsx") as writer:
    metrics_df.to_excel(writer, sheet_name="per_fold", index=False)
    summary.to_excel(writer, sheet_name="summary")

print("✅ 결과가 'bootstrap_cv_results.xlsx'에 저장되었습니다.")
print(summary.round(3))
