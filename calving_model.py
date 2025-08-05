# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import confusion_matrix

# # 1) 제거할 변수 리스트
# drop_cols = [
#     # 식별·메타
#     'name','fold_id_counter','fold_id','TIME',
#     # 레이블·시점 노출
#     'bundate','mdate','TF',
#     # 예측일 차이
#     'expec','expec_mdate_diff','bun_mdate_diff',
#     # 파생량 차이
#     'm4_act_diff','m4_rum_diff',
#     # 날짜 파생
#     'ex_year','ex_mon','ex_season',
#     'birth_year','birth_mon','birth_season',
#     'lin_year','lin_mon','lin_season',
#     # 날짜 원본
#     'birth','last_in'
# ]

# # 2) 전체 데이터 합치기
# dfs = [pd.read_excel(f'./트레이닝데이터/training_fold_{i}.xlsx') for i in range(1,6)]
# df_all = pd.concat(dfs, ignore_index=True)
# X_all = df_all.drop(columns=drop_cols + ['TF']).select_dtypes(include=[np.number])
# y_all = df_all['TF']

# # 3) 모델 정의
# models = {
#     'DecisionTree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42),
#     'RandomForest': RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42),
#     'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
# }

# # 4) 5-Fold CV 성능 수집
# metrics = []
# feature_importances = {}

# for name, model in models.items():
#     sens, spec, ppv, npv, acc = [], [], [], [], []

#     for i in range(1,6):
#         tr = pd.read_excel(f'./트레이닝데이터/training_fold_{i}.xlsx')
#         va = pd.read_excel(f'./트레이닝데이터/validation_fold_{i}.xlsx')

#         X_tr = tr.drop(columns=drop_cols + ['TF']).select_dtypes(include=[np.number])
#         y_tr = tr['TF']
#         X_va = va.drop(columns=drop_cols + ['TF']).select_dtypes(include=[np.number])
#         y_va = va['TF']

#         model.fit(X_tr, y_tr)
#         y_pred = model.predict(X_va)
#         tn, fp, fn, tp = confusion_matrix(y_va, y_pred).ravel()

#         sens.append(tp/(tp+fn))
#         spec.append(tn/(tn+fp))
#         ppv.append(tp/(tp+fp) if tp+fp>0 else np.nan)
#         npv.append(tn/(tn+fn) if tn+fn>0 else np.nan)
#         acc.append((tp+tn)/(tp+tn+fp+fn))

#     # 리스트에 저장
#     metrics.append({
#         'Model': name,
#         'Sensitivity': np.mean(sens),
#         'Specificity': np.mean(spec),
#         'PPV': np.nanmean(ppv),
#         'NPV': np.nanmean(npv),
#         'Accuracy': np.mean(acc)
#     })

#     # 전체 데이터로 재학습해 변수 중요도 저장
#     model.fit(X_all, y_all)
#     feature_importances[name] = pd.Series(model.feature_importances_, index=X_all.columns)

# # 5) metrics → DataFrame 변환
# metrics_df = pd.DataFrame(metrics).set_index('Model')

# # 6) 콘솔 출력 혹은 파일 저장
# print(metrics_df.round(3))
# metrics_df.to_excel("model_performance.xlsx")

# # 7) Matplotlib 테이블로 시각화
# fig, ax = plt.subplots(figsize=(8, 2))
# ax.axis('off')
# tbl = ax.table(
#     cellText=metrics_df.round(3).values,
#     colLabels=metrics_df.columns,
#     rowLabels=metrics_df.index,
#     loc='center'
# )
# tbl.auto_set_font_size(False)
# tbl.set_fontsize(10)
# tbl.scale(1, 1.5)
# plt.tight_layout()
# plt.show()

# # 8) 변수중요도 Top-10 시각화
# for name, imp in feature_importances.items():
#     top10 = imp.nlargest(10)
#     plt.figure()
#     plt.barh(top10.index, top10.values)
#     plt.title(f"Top-10 Feature Importances ({name})")
#     plt.xlabel("Importance")
#     plt.tight_layout()
#     plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

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

keep_cols = ['par','age', 'm4_act_diff', 'm4_rum_diff','stdrum', 'TF']

# 2) 전체 데이터 합치기 (샘플 데이터로 테스트)
# 실제 사용 시에는 아래 주석 해제
dfs = [pd.read_excel(f'./트레이닝데이터/training_fold_{i}.xlsx',
                     usecols=keep_cols) for i in range(1,6)]
df_all = pd.concat(dfs, ignore_index=True)



X_all = df_all.drop(columns=['TF'])
y_all = df_all['TF']

print(f"전체 데이터 크기: {X_all.shape}")
print(f"클래스 분포: {y_all.value_counts()}")

# 3) 피처 엔지니어링
def feature_engineering(X):
    X = X.copy()
    
    # 비율 특성 생성
    if 'mact4' in X.columns and 'mstdact4' in X.columns and X['mstdact4'].std() > 0:
        X['act_cv'] = X['mstdact4'] / (X['mact4'] + 1e-8)  # 활동 변동계수
    
    if 'mrum4' in X.columns and 'mstdrum4' in X.columns and X['mstdrum4'].std() > 0:
        X['rum_cv'] = X['mstdrum4'] / (X['mrum4'] + 1e-8)  # 반추 변동계수
    
    # 현재와 4일 전 비교
    if all(col in X.columns for col in ['mact', 'mact4']):
        X['act_change'] = X['mact'] - X['mact4']
        X['act_ratio'] = X['mact'] / (X['mact4'] + 1e-8)
    
    if all(col in X.columns for col in ['mrum', 'mrum4']):
        X['rum_change'] = X['mrum'] - X['mrum4']
        X['rum_ratio'] = X['mrum'] / (X['mrum4'] + 1e-8)
    
    # 표준화된 점수 (z-score)
    if all(col in X.columns for col in ['mact', 'stdact']):
        X['act_zscore'] = (X['mact'] - X['mact'].mean()) / (X['stdact'] + 1e-8)
    
    if all(col in X.columns for col in ['mrum', 'stdrum']):
        X['rum_zscore'] = (X['mrum'] - X['mrum'].mean()) / (X['stdrum'] + 1e-8)
    
    # 나이 관련 특성
    if 'age' in X.columns:
        X['age_squared'] = X['age'] ** 2
        X['age_log'] = np.log(X['age'] + 1)
        X['is_young'] = (X['age'] < X['age'].quantile(0.25)).astype(int)
        X['is_old'] = (X['age'] > X['age'].quantile(0.75)).astype(int)
    
    # 산차 관련
    if 'par' in X.columns:
        X['is_first_birth'] = (X['par'] == 1).astype(int)
        X['is_multiparous'] = (X['par'] > 3).astype(int)
    
    return X

# 피처 엔지니어링 적용
X_all_eng = feature_engineering(X_all)
print(f"피처 엔지니어링 후 특성 수: {X_all_eng.shape[1]}")

# 4) 개선된 모델 정의 (하이퍼파라미터 튜닝 포함)
models = {
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(
        max_depth=8, 
        min_samples_leaf=5,
        min_samples_split=10,
        max_features='sqrt',
        random_state=42
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        min_samples_split=8,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
}

# 5) 향상된 성능 평가
def enhanced_evaluation(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sens = tp/(tp+fn) if (tp+fn) > 0 else 0
    spec = tn/(tn+fp) if (tn+fp) > 0 else 0
    ppv = tp/(tp+fp) if (tp+fp) > 0 else 0
    npv = tn/(tn+fn) if (tn+fn) > 0 else 0
    acc = (tp+tn)/(tp+tn+fp+fn)
    
    # F1 점수와 균형 정확도 추가
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0
    balanced_acc = (sens + spec) / 2
    
    results = {
        'Sensitivity': sens,
        'Specificity': spec,
        'PPV': ppv,
        'NPV': npv,
        'Accuracy': acc,
        'F1_Score': f1,
        'Balanced_Accuracy': balanced_acc
    }
    
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            results['AUC'] = auc
        except:
            results['AUC'] = np.nan
    
    return results

# 6) Cross-validation with feature selection
metrics = []
feature_importances = {}

print("모델 훈련 및 평가 시작...")

for name, model in models.items():
    print(f"\n{name} 모델 훈련 중...")
    
    fold_metrics = []
    
    # 샘플 데이터로 테스트하는 경우 (실제 사용 시 아래 부분을 원래 코드로 교체)
    for i in range(1, 2):  # 테스트용으로 1개 폴드만
        # 실제 사용 시:
        # tr = pd.read_excel(f'./트레이닝데이터/training_fold_{i}.xlsx')
        # va = pd.read_excel(f'./트레이닝데이터/validation_fold_{i}.xlsx')
        
        # 테스트용 데이터 분할
        from sklearn.model_selection import train_test_split
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_all_eng, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        
        # 스케일링 (로지스틱 회귀를 위해)
        if name == 'LogisticRegression':
            scaler = StandardScaler()
            X_tr_scaled = pd.DataFrame(
                scaler.fit_transform(X_tr), 
                columns=X_tr.columns, 
                index=X_tr.index
            )
            X_va_scaled = pd.DataFrame(
                scaler.transform(X_va), 
                columns=X_va.columns, 
                index=X_va.index
            )
            X_tr_use, X_va_use = X_tr_scaled, X_va_scaled
        else:
            X_tr_use, X_va_use = X_tr, X_va
        
        # 피처 선택 (중요한 특성만 선택)
        if X_tr_use.shape[1] > 20:
            selector = SelectKBest(f_classif, k=min(20, X_tr_use.shape[1]))
            X_tr_selected = selector.fit_transform(X_tr_use, y_tr)
            X_va_selected = selector.transform(X_va_use)
            selected_features = X_tr_use.columns[selector.get_support()]
            X_tr_use = pd.DataFrame(X_tr_selected, columns=selected_features)
            X_va_use = pd.DataFrame(X_va_selected, columns=selected_features)
        
        # 모델 훈련
        model.fit(X_tr_use, y_tr)
        y_pred = model.predict(X_va_use)
        
        # 확률 예측 (가능한 경우)
        try:
            y_prob = model.predict_proba(X_va_use)[:, 1]
        except:
            y_prob = None
        
        # 평가
        metrics_fold = enhanced_evaluation(y_va, y_pred, y_prob)
        fold_metrics.append(metrics_fold)
    
    # 폴드별 평균 계산
    avg_metrics = {}
    for metric in fold_metrics[0].keys():
        values = [m[metric] for m in fold_metrics if not np.isnan(m[metric])]
        avg_metrics[metric] = np.mean(values) if values else np.nan
    
    avg_metrics['Model'] = name
    metrics.append(avg_metrics)
    
    # 전체 데이터로 특성 중요도 계산
    if hasattr(model, 'feature_importances_'):
        # 전체 데이터로 재훈련
        X_full_use = X_all_eng
        if name == 'LogisticRegression':
            scaler = StandardScaler()
            X_full_use = pd.DataFrame(
                scaler.fit_transform(X_all_eng), 
                columns=X_all_eng.columns
            )
        
        if X_full_use.shape[1] > 20:
            selector = SelectKBest(f_classif, k=min(20, X_full_use.shape[1]))
            X_full_selected = selector.fit_transform(X_full_use, y_all)
            selected_features = X_full_use.columns[selector.get_support()]
            X_full_use = pd.DataFrame(X_full_selected, columns=selected_features)
        
        model.fit(X_full_use, y_all)
        feature_importances[name] = pd.Series(
            model.feature_importances_, 
            index=X_full_use.columns
        ).sort_values(ascending=False)

# 7) 결과 정리 및 출력
metrics_df = pd.DataFrame(metrics).set_index('Model')
print(f"\n=== 모델 성능 비교 ===")
print(metrics_df.round(4))

# 8) 최고 성능 모델 식별
best_model_acc = metrics_df['Accuracy'].idxmax()
best_model_f1 = metrics_df['F1_Score'].idxmax()
best_model_auc = metrics_df['AUC'].idxmax() if 'AUC' in metrics_df.columns else None

print(f"\n=== 최고 성능 모델 ===")
print(f"최고 정확도: {best_model_acc} ({metrics_df.loc[best_model_acc, 'Accuracy']:.4f})")
print(f"최고 F1 점수: {best_model_f1} ({metrics_df.loc[best_model_f1, 'F1_Score']:.4f})")
if best_model_auc:
    print(f"최고 AUC: {best_model_auc} ({metrics_df.loc[best_model_auc, 'AUC']:.4f})")

# 9) 특성 중요도 분석
print(f"\n=== 중요 특성 분석 ===")
for name, importance in feature_importances.items():
    print(f"\n{name} 상위 10개 중요 특성:")
    print(importance.head(10).round(4))

# 10) 결과 저장
try:
    metrics_df.to_excel("enhanced_model_performance.xlsx")
    print(f"\n결과가 'enhanced_model_performance.xlsx'에 저장되었습니다.")
except:
    print(f"\nExcel 저장 실패, CSV로 저장합니다.")
    metrics_df.to_csv("enhanced_model_performance.csv")

