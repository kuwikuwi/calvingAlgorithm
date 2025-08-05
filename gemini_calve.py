import pandas as pd
import numpy as np
# from sklearn.model_selection import GroupKFold # 이제 직접 파일 사용
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
from datetime import timedelta

# 시각화 (선택 사항)
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(df_input):
    """데이터프레임을 받아 전처리 및 특성 공학을 수행하는 함수"""
    df = df_input.copy()

    # --- 날짜 컬럼 datetime으로 변환 ---
    if 'CalvingDate' not in df.columns and 'bundate' in df.columns:
        df.rename(columns={'bundate': 'CalvingDate'}, inplace=True)

    date_cols = ['CalvingDate', 'mdate', 'birth', 'last_in', 'expec']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # --- 필수 날짜 정보 없는 행 제거 ---
    df.dropna(subset=['CalvingDate', 'mdate', 'last_in', 'birth', 'expec'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- 목표 변수 생성 ---
    df['target_days_to_calving'] = (df['CalvingDate'] - df['mdate']).dt.days

    # --- 주요 특성 생성 ---
    df['days_since_last_in'] = (df['mdate'] - df['last_in']).dt.days
    df['age_at_mdate_days'] = (df['mdate'] - df['birth']).dt.days
    df['days_to_expected_calving'] = (df['expec'] - df['mdate']).dt.days

    return df

def get_features_and_target(df_processed, numerical_features, categorical_features):
    """전처리된 데이터프레임에서 피처와 타겟을 분리하는 함수"""
    df = df_processed.copy()
    # 피처 리스트에서 혹시라도 숫자형이 아닌 컬럼이 있다면 변환
    for col in numerical_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 결측치 처리 (수치형: 중앙값, 범주형: 최빈값 - 예시)
    for col in numerical_features:
        df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_features:
        # 최빈값이 여러 개일 수 있으므로 첫 번째 값 사용
        if not df[col].mode().empty:
            df[col].fillna(df[col].mode()[0], inplace=True)
        else: # 만약 모든 값이 NaN이면 특정 값으로 채우거나 다른 처리
            df[col].fillna('Unknown', inplace=True)


    X = df[numerical_features + categorical_features]
    y = df['target_days_to_calving']
    return X, y



# --- 사용할 피처 정의 ---
numerical_features = [
    'mact4', 'mstdact4', 'mrum4', 'mstdrum4',
    'mact', 'mrum', 'stdact', 'stdrum',
    'par', 'age',
    'm4_act_diff', 'm4_rum_diff',
    'days_since_last_in',
    'age_at_mdate_days',
    'days_to_expected_calving'
]

categorical_features = [
    'method',
    'ex_season',
    'birth_season',
    'lin_season'
]

# --- 전처리기 정의 ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# --- LightGBM 모델 파이프라인 ---
# 하이퍼파라미터 예시 (튜닝 필요)
lgbm_params = {
    'objective': 'regression_l1', # MAE에 더 강인한 목적 함수
    'metric': 'mae',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}

lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor(**lgbm_params))
])


num_folds = 5 # 파일이 5개 fold로 나뉘어 있음
fold_mae_scores = []
all_true_y_calving_date = []
all_pred_y_calving_date = []
all_mdate_for_pred = []
all_animal_ids_val = [] # 소 ID 저장을 위해 추가

# 각 fold에 대한 모델 성능 및 예측 저장용
fold_results = []

print("\n--- 교차 검증 시작 (파일 기반) ---")
for i in range(1, num_folds + 1):
    train_file = f'트레이닝데이터/training_fold_{i}.xlsx'
    val_file = f'트레이닝데이터/validation_fold_{i}.xlsx'

    print(f"\n--- Fold {i} ---")
    print(f"Training data: {train_file}")
    print(f"Validation data: {val_file}")

    try:
        df_train_raw = pd.read_excel(train_file)
        df_val_raw = pd.read_excel(val_file)
    except FileNotFoundError:
        print(f"Error: {train_file} or {val_file} not found. Skipping fold {i}.")
        # 다른 폴더에 있다면 경로를 포함해서 지정: '트레이닝데이터/training_fold_1.xlsx'
        # 예: train_file = f'트레이닝데이터/training_fold_{i}.xlsx'
        #     val_file = f'트레이닝데이터/validation_fold_{i}.xlsx'
        continue

    # 데이터 전처리
    df_train_processed = preprocess_data(df_train_raw)
    df_val_processed = preprocess_data(df_val_raw)

    # 피처 및 타겟 분리
    X_train, y_train = get_features_and_target(df_train_processed, numerical_features, categorical_features)
    X_val, y_val = get_features_and_target(df_val_processed, numerical_features, categorical_features)

    # 학습 데이터 또는 검증 데이터가 비어있는 경우 스킵
    if X_train.empty or X_val.empty:
        print(f"Fold {i} skipped: No training or validation data after preprocessing.")
        continue

    # 모델 학습
    # LightGBM의 경우, 범주형 피처를 명시적으로 알려주면 더 잘 처리할 수 있음 (파이프라인 외부에서)
    # 하지만 현재는 OneHotEncoder를 사용하므로 괜찮음
    lgbm_pipeline.fit(X_train, y_train)

    # 예측
    y_pred_days_to_calving = lgbm_pipeline.predict(X_val)

    # MAE 계산 (target_days_to_calving 기준)
    mae_days = mean_absolute_error(y_val, y_pred_days_to_calving)
    fold_mae_scores.append(mae_days)
    print(f"Fold {i} - MAE (days_to_calving): {mae_days:.3f} days")

    # 실제 분만일 vs 예측 분만일 계산 및 저장
    mdate_val = df_val_processed['mdate']
    calving_date_true_val = df_val_processed['CalvingDate']
    animal_ids_val = df_val_processed['name'] # 소 ID

    fold_pred_df = pd.DataFrame({
        'name': animal_ids_val,
        'mdate': mdate_val,
        'true_CalvingDate': calving_date_true_val,
        'true_days_to_calving': y_val,
        'pred_days_to_calving': y_pred_days_to_calving
    })
    fold_pred_df['pred_CalvingDate'] = fold_pred_df.apply(
        lambda row: row['mdate'] + timedelta(days=np.round(row['pred_days_to_calving'])), axis=1
    )
    fold_pred_df['calving_date_error_days'] = (fold_pred_df['pred_CalvingDate'] - fold_pred_df['true_CalvingDate']).dt.days
    fold_results.append(fold_pred_df)

    # 전체 결과 취합용
    all_true_y_calving_date.extend(calving_date_true_val.tolist())
    all_pred_y_calving_date.extend(fold_pred_df['pred_CalvingDate'].tolist())
    all_animal_ids_val.extend(animal_ids_val.tolist())


print(f"\n--- 교차 검증 완료 ---")
if fold_mae_scores: # 적어도 하나의 fold가 성공적으로 실행된 경우
    print(f"평균 MAE (days_to_calving): {np.mean(fold_mae_scores):.3f} days")

    # --- 전체 교차 검증 결과 취합 및 최종 분만일 예측 정확도 계산 ---
    all_results_df = pd.concat(fold_results, ignore_index=True)

    # 분만일 예측 오차 (일 단위)
    mae_calving_date_overall = np.mean(np.abs(all_results_df['calving_date_error_days']))
    print(f"\n전체 교차 검증 - 최종 분만일 예측 MAE: {mae_calving_date_overall:.3f} days")

    # ±N일 이내 정확도
    for n_days in [0, 1, 2, 3, 5, 7]:
        accuracy = np.sum(np.abs(all_results_df['calving_date_error_days']) <= n_days) / len(all_results_df)
        print(f"전체 교차 검증 - 분만일 예측 정확도 (±{n_days}일 이내): {accuracy*100:.2f}%")

    # 예측 결과 저장 (선택 사항)
    # all_results_df.to_excel("all_folds_predictions.xlsx", index=False)
    # print("\n모든 fold의 예측 결과가 all_folds_predictions.xlsx 로 저장되었습니다.")

else:
    print("교차 검증을 위한 유효한 폴드가 없습니다.")


