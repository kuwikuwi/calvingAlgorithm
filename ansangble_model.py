import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import catboost as cb

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

# 2) 평가 함수: 기존 5-Fold CV 경로 그대로 사용
def evaluate_cv(model):
    aucs = []
    for i in range(1, 6):
        tr = pd.read_excel(f'./fw(1)/트레이닝데이터/training_fold_{i}.xlsx')
        va = pd.read_excel(f'./fw(1)/트레이닝데이터/validation_fold_{i}.xlsx')
        y_tr, y_va = tr['TF'], va['TF']
        X_tr = tr.drop(columns=drop_cols).select_dtypes(include=[np.number])
        X_va = va.drop(columns=drop_cols).select_dtypes(include=[np.number])
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:,1]
        aucs.append(roc_auc_score(y_va, preds))
    return np.mean(aucs), aucs

# 3) Random Forest + RandomizedSearchCV
rf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [5, 20]
}
rs = RandomizedSearchCV(rf, param_dist, n_iter=8, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1)
# For tuning, combine all training folds into one dataset
dfs = [pd.read_excel(f'./fw(1)/트레이닝데이터/training_fold_{i}.xlsx') for i in range(1,6)]
df_all = pd.concat(dfs, ignore_index=True)
X_all = df_all.drop(columns=drop_cols + ['TF']).select_dtypes(include=[np.number])
y_all = df_all['TF']
rs.fit(X_all, y_all)
best_rf = rs.best_estimator_

# 4) LightGBM 기본 튜닝
lgbm = lgb.LGBMClassifier(random_state=42)
lgb_params = {
    'n_estimators': [100, 300],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 63]
}
gs_lgb = GridSearchCV(lgbm, lgb_params, scoring='roc_auc', cv=3, n_jobs=-1)
gs_lgb.fit(X_all, y_all)
best_lgb = gs_lgb.best_estimator_

# 5) CatBoost 기본 튜닝
cbm = cb.CatBoostClassifier(verbose=0, random_state=42)
cb_params = {
    'iterations': [100, 300],
    'learning_rate': [0.05, 0.1],
    'depth': [4, 6]
}
gs_cb = GridSearchCV(cbm, cb_params, scoring='roc_auc', cv=3, n_jobs=-1)
gs_cb.fit(X_all, y_all)
best_cb = gs_cb.best_estimator_

# 6) Stacking Ensemble
stack = StackingClassifier(
    estimators=[
        ('rf', best_rf),
        ('lgb', best_lgb),
        ('cb', best_cb),
    ],
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    cv=3,
    n_jobs=-1
)

# 7) 최종 평가
models = {
    'Best_RF': best_rf,
    'Best_LGBM': best_lgb,
    'Best_CatBoost': best_cb,
    'Stacking': stack
}

for name, model in models.items():
    mean_auc, folds = evaluate_cv(model)
    print(f"{name}: Fold AUC = {[round(a,4) for a in folds]}")
    print(f"→ {name} Mean AUC: {mean_auc:.4f}\n")
