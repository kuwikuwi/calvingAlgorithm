import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# 1) 제거할 변수 리스트
drop_cols = [
    # 식별·메타
    'name','fold_id_counter','fold_id','TIME',
    # 레이블·시점 노출
    'bundate','mdate','TF',
    # 예측일 차이
    'expec','expec_mdate_diff','bun_mdate_diff',
    # 파생량 차이
    'm4_act_diff','m4_rum_diff',
    # 날짜 파생
    'ex_year','ex_mon','ex_season',
    'birth_year','birth_mon','birth_season',
    'lin_year','lin_mon','lin_season',
    # 날짜 원본
    'birth','last_in'
]


# 2) 5-Fold CV
results = []
for i in range(1,6):
    tr = pd.read_excel(f'./트레이닝데이터/training_fold_{i}.xlsx')
    va = pd.read_excel(f'./트레이닝데이터/validation_fold_{i}.xlsx')
    
    # 라벨 분리
    y_tr = tr['TF']
    y_va = va['TF']
    
    # 피처셋에서 제거
    X_tr = tr.drop(columns=drop_cols).select_dtypes(include=[np.number])
    X_va = va.drop(columns=drop_cols).select_dtypes(include=[np.number])
    
    # 모델 학습 및 평가
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_tr, y_tr)
    preds = model.predict_proba(X_va)[:,1]
    auc = roc_auc_score(y_va, preds)
    
    results.append({'fold': i, 'auc': round(auc,4)})

# 3) 결과 출력
import pandas as pd
df_res = pd.DataFrame(results)
print(df_res, "\nMean AUC:", df_res['auc'].mean().round(4))
