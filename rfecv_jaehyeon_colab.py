
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

# Google Colab 환경에서 Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 엑셀 파일 경로 설정 (Google Drive에 파일이 있다고 가정)
# 파일 경로는 실제 파일 위치에 맞게 수정해야 합니다.
file_path = 'C:/Users/kibae/Desktop/분만알고리즘/트레이닝데이터' # 'your_excel_file.xlsx'를 실제 파일 이름으로 변경

# 엑셀 파일 불러오기
try:
    df = pd.read_excel(file_path)
    print("엑셀 파일을 성공적으로 불러왔습니다.")
    print("데이터프레임 정보:")
    df.info()
    print("\n데이터프레임 상위 5행:")
    print(df.head())
except FileNotFoundError:
    print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인하세요: {file_path}")
    exit()
except Exception as e:
    print(f"파일 불러오기 중 오류 발생: {e}")
    exit()

target_variable_name = input("타겟 변수 명을 입력하세요 (예: c): ")

# 데이터 컬럼에 있는 변수인지 교차검증
if target_variable_name in df.columns:
    target_variable = df[target_variable_name]
    print(f"'{target_variable_name}'가 타겟 변수로 설정되었습니다.")
    print(target_variable)
else:
    print(f"'{target_variable_name}'는 데이터프레임에 없는 컬럼명입니다. 다음 컬럼 중에서 선택해주세요: {df.columns.tolist()}")

# --- 데이터 전처리 및 특성/타겟 분리 ---
# 예시: 마지막 열을 타겟 변수 (y)로 사용하고, 나머지 열을 특성 (X)으로 사용합니다.
# 실제 데이터에 맞게 수정해야 합니다.
# 숫자형이 아닌 데이터는 RFECV 전에 적절히 인코딩하거나 제거해야 합니다.
# 결측치는 RFECV 전에 처리해야 합니다 (예: 평균, 중앙값으로 대체).


# 숫자형 컬럼만 선택
df_numeric = df.select_dtypes(include=[np.number])

# 타겟이 숫자형인지 확인
if target_variable_name not in df_numeric.columns:
    raise ValueError(f"타겟 변수 '{target_variable_name}'는 숫자형이 아닙니다. RFECV에 사용할 수 없습니다.")

# 결측치 처리 (평균 대체)
df_numeric = df_numeric.fillna(df_numeric.mean())

# X, y 설정
y = df_numeric[target_variable_name]
X = df_numeric.drop(columns=[target_variable_name])

# 데이터 분할 (학습/테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n데이터 분할 결과: 학습 세트 {X_train.shape}, 테스트 세트 {X_test.shape}")

# --- RFECV 모델 설정 및 실행 ---
# RFECV는 특정 Estimator를 사용하여 특성을 재귀적으로 제거하며 성능을 평가합니다.
# 회귀 문제의 경우 SVR, 분류 문제의 경우 SVC 등을 사용합니다.
# 여기서는 회귀 문제 예시로 SVR을 사용합니다.

# Estimator 선택 (예: Support Vector Regressor)
estimator = SVR(kernel="linear")

# RFECV 설정
# cv: 교차 검증 폴드 수
# step: 각 반복에서 제거할 특성 수 또는 비율
# scoring: 평가 지표 (회귀: 'neg_mean_squared_error', 'r2' 등, 분류: 'accuracy', 'f1' 등)
scoring_metric = 'neg_mean_squared_error'

print(f"\nRFECV를 {estimator.__class__.__name__} Estimator와 {scoring_metric} 스코어링을 사용하여 실행합니다.")

rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring=scoring_metric)

# RFECV 학습 및 특성 선택
rfecv.fit(X_train, y_train)

# 선택된 특성 확인
print("\nRFECV 결과:")
print(f"최적의 특성 개수: {rfecv.n_features_}")
print("선택된 특성 마스크:", rfecv.support_) # True는 선택된 특성
print("특성 랭킹:", rfecv.ranking_) # 1은 선택된 특성, 나머지는 제거된 순위

# 선택된 특성 이름 가져오기
selected_features = X_train.columns[rfecv.support_]
print("\n선택된 특성 이름:")
print(selected_features)

# --- 선택된 특성으로 모델 재학습 및 평가 ---
# 선택된 특성만 사용하여 학습 데이터를 만듭니다.
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# 선택된 특성으로 새로운 모델을 학습합니다.
final_model = SVR(kernel="linear") # RFECV에 사용한 것과 동일한 Estimator 사용
final_model.fit(X_train_selected, y_train)

# 테스트 세트 예측
y_pred = final_model.predict(X_test_selected)

# 모델 성능 평가 (회귀 문제 예시: MSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\n선택된 특성으로 학습된 모델 성능 (회귀):")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# --- RFECV 시각화 ---
# 최적 특성 개수 찾기 과정을 시각화할 수 있습니다.
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.title('Recursive Feature Elimination with Cross-Validation (RFECV)')
plt.xlabel("Number of features selected")
plt.ylabel(f"Cross validation score ({scoring_metric})")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

print("\nRFECV 분석 및 모델 평가 완료.")

df_selected = df_numeric[selected_features]
df_selected[target_variable_name] = y  # 타겟 컬럼 다시 붙이기

# 엑셀 파일로 저장
output_path = 'C:/Users/kibae/Desktop/분만알고리즘/jaehyeon_result'  # 원하는 경로로 수정
df_selected.to_excel(output_path, index=False)

print(f"\n선택된 특성만 포함된 데이터를 다음 경로에 저장했습니다:\n{output_path}")