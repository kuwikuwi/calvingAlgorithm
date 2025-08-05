import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1) 제거할 변수 리스트 (기존과 동일)
drop_cols = [
    'name','fold_id_counter','fold_id','TIME',
    'bundate','mdate','TF',
    'expec','expec_mdate_diff','bun_mdate_diff',
    'm4_act_diff','m4_rum_diff',
    'ex_year','ex_mon','ex_season',
    'birth_year','birth_mon','birth_season',
    'lin_year','lin_mon','lin_season',
    'birth','last_in'
]

# 2) 평가할 모델 정의
models = {
    'DecisionTree': DecisionTreeClassifier(
        max_depth=3,        # 과적합 방지용 depth 제한
        min_samples_leaf=30,# 잎 노드 최소 샘플 수
        random_state=42
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=300,   # 트리 100개
        max_features='sqrt',# 분할시 sqrt(#features)
        random_state=42
    )
}

# 3) 5-Fold CV for each model
for name, model in models.items():
    aucs = []
    for i in range(1, 6):
        tr = pd.read_excel(f'./트레이닝데이터/training_fold_{i}.xlsx')
        va = pd.read_excel(f'./트레이닝데이터/validation_fold_{i}.xlsx')
        
        # 레이블
        y_tr = tr['TF']
        y_va = va['TF']
        
        # 피처셋에서 누수/중복 변수 제거 + 숫자형만
        X_tr = tr.drop(columns=drop_cols).select_dtypes(include=[np.number])
        X_va = va.drop(columns=drop_cols).select_dtypes(include=[np.number])
        
        # 학습 & 예측
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:,1]
        aucs.append(roc_auc_score(y_va, preds))
    
    print(f"{name} AUC per fold: {[round(a,4) for a in aucs]}")
    print(f"→ {name} Mean AUC: {np.mean(aucs):.4f}\n")



