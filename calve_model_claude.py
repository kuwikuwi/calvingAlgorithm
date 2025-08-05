import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# 변수 설정
keep_cols = ['par', 'age', 'm4_act_diff', 'm4_rum_diff', 'stdrum', 'TF']

# 피처 엔지니어링 함수
def enhanced_feature_engineering(X):
    X = X.copy()
    
    # 기존 변수 정규화
    X['rum_diff_norm'] = X['m4_rum_diff'] / (X['stdrum'] + 1e-8)
    X['act_rum_ratio'] = X['m4_act_diff'] / (X['m4_rum_diff'] + 1e-8)
    
    # 나이와 산차의 상호작용
    X['age_par_interaction'] = X['age'] * X['par']
    X['is_risky_age'] = ((X['age'] < 800) | (X['age'] > 2000)).astype(int)
    
    # 복합 지표
    X['calving_risk_score'] = (
        np.abs(X['m4_rum_diff']) * 0.4 + 
        np.abs(X['m4_act_diff']) * 0.3 + 
        X['stdrum'] * 0.3
    )
    
    return X

# 모델 정의 함수
def get_models():
    return {
        'LightGBM': LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=20,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
    }

# 평가 함수
def evaluate_model(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Sensitivity': tp/(tp+fn) if (tp+fn) > 0 else 0,
        'Specificity': tn/(tn+fp) if (tn+fp) > 0 else 0,
        'PPV': tp/(tp+fp) if (tp+fp) > 0 else 0,
        'NPV': tn/(tn+fn) if (tn+fn) > 0 else 0,
        'Accuracy': (tp+tn)/(tp+tn+fp+fn),
        'F1_Score': 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0,
        'Balanced_Acc': ((tp/(tp+fn) if (tp+fn) > 0 else 0) + 
                        (tn/(tn+fp) if (tn+fp) > 0 else 0)) / 2,
        'AUC': roc_auc_score(y_true, y_prob)
    }
    
    return metrics

# 메인 비교 함수
def compare_balanced_unbalanced():
    print("데이터 로드 중...")
    
    # 결과 저장용
    all_results = []
    
    # 5-Fold 교차 검증
    for fold in range(1, 6):
        print(f"\n=== Fold {fold} ===")
        
        # 데이터 로드
        tr = pd.read_excel(f'./트레이닝데이터/training_fold_{fold}.xlsx', usecols=keep_cols)
        va = pd.read_excel(f'./트레이닝데이터/validation_fold_{fold}.xlsx', usecols=keep_cols)
        
        # 피처 엔지니어링
        X_tr = enhanced_feature_engineering(tr.drop(columns=['TF']))
        y_tr = tr['TF']
        X_va = enhanced_feature_engineering(va.drop(columns=['TF']))
        y_va = va['TF']
        
        # 각 모델에 대해
        for model_name, model in get_models().items():
            
            # 1. 불균형 데이터로 학습
            print(f"\n{model_name} - 불균형 데이터")
            model_unbalanced = model.__class__(**model.get_params())
            model_unbalanced.fit(X_tr, y_tr)
            
            y_pred_unbal = model_unbalanced.predict(X_va)
            y_prob_unbal = model_unbalanced.predict_proba(X_va)[:, 1]
            
            metrics_unbal = evaluate_model(y_va, y_pred_unbal, y_prob_unbal)
            metrics_unbal['Model'] = model_name
            metrics_unbal['Data_Type'] = 'Unbalanced'
            metrics_unbal['Fold'] = fold
            
            print(f"  불균형 - Sens: {metrics_unbal['Sensitivity']:.3f}, "
                  f"Spec: {metrics_unbal['Specificity']:.3f}, "
                  f"F1: {metrics_unbal['F1_Score']:.3f}")
            
            # 2. SMOTE로 균형 맞춘 데이터로 학습
            print(f"{model_name} - SMOTE 적용")
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_tr_balanced, y_tr_balanced = smote.fit_resample(X_tr, y_tr)
            
            model_balanced = model.__class__(**model.get_params())
            model_balanced.fit(X_tr_balanced, y_tr_balanced)
            
            y_pred_bal = model_balanced.predict(X_va)
            y_prob_bal = model_balanced.predict_proba(X_va)[:, 1]
            
            metrics_bal = evaluate_model(y_va, y_pred_bal, y_prob_bal)
            metrics_bal['Model'] = model_name
            metrics_bal['Data_Type'] = 'SMOTE'
            metrics_bal['Fold'] = fold
            
            print(f"  SMOTE   - Sens: {metrics_bal['Sensitivity']:.3f}, "
                  f"Spec: {metrics_bal['Specificity']:.3f}, "
                  f"F1: {metrics_bal['F1_Score']:.3f}")
            
            # 결과 저장
            all_results.append(metrics_unbal)
            all_results.append(metrics_bal)
    
    # DataFrame으로 변환
    df_results = pd.DataFrame(all_results)
    
    # 평균 계산
    df_avg = df_results.groupby(['Model', 'Data_Type']).agg({
        'Sensitivity': ['mean', 'std'],
        'Specificity': ['mean', 'std'],
        'PPV': ['mean', 'std'],
        'NPV': ['mean', 'std'],
        'Accuracy': ['mean', 'std'],
        'F1_Score': ['mean', 'std'],
        'Balanced_Acc': ['mean', 'std'],
        'AUC': ['mean', 'std']
    }).round(4)
    
    # 컬럼명 정리
    df_avg.columns = ['_'.join(col).strip() for col in df_avg.columns.values]
    df_avg = df_avg.reset_index()
    
    # 변화율 계산
    df_comparison = []
    for model in ['LightGBM', 'XGBoost', 'GradientBoosting']:
        unbal = df_avg[(df_avg['Model'] == model) & (df_avg['Data_Type'] == 'Unbalanced')]
        bal = df_avg[(df_avg['Model'] == model) & (df_avg['Data_Type'] == 'SMOTE')]
        
        if len(unbal) > 0 and len(bal) > 0:
            comparison = {
                'Model': model,
                'Sensitivity_Change': f"{(bal['Sensitivity_mean'].values[0] - unbal['Sensitivity_mean'].values[0]):.3f}",
                'Specificity_Change': f"{(bal['Specificity_mean'].values[0] - unbal['Specificity_mean'].values[0]):.3f}",
                'F1_Change': f"{(bal['F1_Score_mean'].values[0] - unbal['F1_Score_mean'].values[0]):.3f}",
                'AUC_Change': f"{(bal['AUC_mean'].values[0] - unbal['AUC_mean'].values[0]):.3f}"
            }
            df_comparison.append(comparison)
    
    df_comparison = pd.DataFrame(df_comparison)
    
    # 엑셀 파일로 저장
    with pd.ExcelWriter('balanced_vs_unbalanced_comparison.xlsx', engine='openpyxl') as writer:
        # Sheet 1: 전체 결과
        df_results.to_excel(writer, sheet_name='All_Results', index=False)
        
        # Sheet 2: 평균 및 표준편차
        df_avg.to_excel(writer, sheet_name='Average_Results', index=False)
        
        # Sheet 3: 변화율 비교
        df_comparison.to_excel(writer, sheet_name='Change_Comparison', index=False)
        
        # Sheet 4: 요약 테이블
        summary_data = []
        for model in ['LightGBM', 'XGBoost', 'GradientBoosting']:
            for data_type in ['Unbalanced', 'SMOTE']:
                row = df_avg[(df_avg['Model'] == model) & (df_avg['Data_Type'] == data_type)]
                if len(row) > 0:
                    summary_data.append({
                        'Model': model,
                        'Data_Type': data_type,
                        'Sensitivity': f"{row['Sensitivity_mean'].values[0]:.3f} (±{row['Sensitivity_std'].values[0]:.3f})",
                        'Specificity': f"{row['Specificity_mean'].values[0]:.3f} (±{row['Specificity_std'].values[0]:.3f})",
                        'F1_Score': f"{row['F1_Score_mean'].values[0]:.3f} (±{row['F1_Score_std'].values[0]:.3f})",
                        'AUC': f"{row['AUC_mean'].values[0]:.3f} (±{row['AUC_std'].values[0]:.3f})"
                    })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 5: 최적 모델 추천
        recommendations = []
        
        # Sensitivity 최고
        best_sens = df_avg.loc[df_avg['Sensitivity_mean'].idxmax()]
        recommendations.append({
            'Metric': 'Highest Sensitivity',
            'Best_Model': best_sens['Model'],
            'Data_Type': best_sens['Data_Type'],
            'Value': f"{best_sens['Sensitivity_mean']:.3f}"
        })
        
        # Specificity 최고
        best_spec = df_avg.loc[df_avg['Specificity_mean'].idxmax()]
        recommendations.append({
            'Metric': 'Highest Specificity',
            'Best_Model': best_spec['Model'],
            'Data_Type': best_spec['Data_Type'],
            'Value': f"{best_spec['Specificity_mean']:.3f}"
        })
        
        # F1 Score 최고
        best_f1 = df_avg.loc[df_avg['F1_Score_mean'].idxmax()]
        recommendations.append({
            'Metric': 'Highest F1 Score',
            'Best_Model': best_f1['Model'],
            'Data_Type': best_f1['Data_Type'],
            'Value': f"{best_f1['F1_Score_mean']:.3f}"
        })
        
        # AUC 최고
        best_auc = df_avg.loc[df_avg['AUC_mean'].idxmax()]
        recommendations.append({
            'Metric': 'Highest AUC',
            'Best_Model': best_auc['Model'],
            'Data_Type': best_auc['Data_Type'],
            'Value': f"{best_auc['AUC_mean']:.3f}"
        })
        
        df_recommendations = pd.DataFrame(recommendations)
        df_recommendations.to_excel(writer, sheet_name='Best_Models', index=False)
    
    print("\n=== 결과 저장 완료 ===")
    print("파일명: balanced_vs_unbalanced_comparison.xlsx")
    print("\n시트 구성:")
    print("1. All_Results: 모든 폴드의 상세 결과")
    print("2. Average_Results: 평균 및 표준편차")
    print("3. Change_Comparison: SMOTE 적용 전후 변화")
    print("4. Summary: 요약 테이블")
    print("5. Best_Models: 지표별 최적 모델")
    
    # 간단한 요약 출력
    print("\n=== 주요 결과 요약 ===")
    for model in ['LightGBM', 'XGBoost', 'GradientBoosting']:
        print(f"\n{model}:")
        unbal = df_avg[(df_avg['Model'] == model) & (df_avg['Data_Type'] == 'Unbalanced')]
        bal = df_avg[(df_avg['Model'] == model) & (df_avg['Data_Type'] == 'SMOTE')]
        
        if len(unbal) > 0 and len(bal) > 0:
            print(f"  Sensitivity: {unbal['Sensitivity_mean'].values[0]:.3f} → {bal['Sensitivity_mean'].values[0]:.3f} "
                  f"(+{bal['Sensitivity_mean'].values[0] - unbal['Sensitivity_mean'].values[0]:.3f})")
            print(f"  F1 Score: {unbal['F1_Score_mean'].values[0]:.3f} → {bal['F1_Score_mean'].values[0]:.3f} "
                  f"(+{bal['F1_Score_mean'].values[0] - unbal['F1_Score_mean'].values[0]:.3f})")

# 실행
if __name__ == "__main__":
    compare_balanced_unbalanced()