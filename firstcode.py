import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# 1) 변수 및 환경 설정
# ──────────────────────────────────────────────────────────────
EXCEL_PATH = (
    "C:/Users/오치승/OneDrive/바탕 화면/분만알고리즘/fw(1)/"
    "2024_분만개체_범위필터링완료.xlsx"
)

SEQUENCE_LEN = 60   # 과거 60스텝(2시간 간격 × 60 = 120시간 ≈ 5일)
HORIZON      = 15   # 앞으로 15일까지 예측
NUM_CLASSES  = HORIZON + 1   # 1~15일 + 15일 초과(>15) = 16개 클래스

# ──────────────────────────────────────────────────────────────
# 2) 엑셀 파일을 시트별로 읽어 와서 전처리
# ──────────────────────────────────────────────────────────────
records = []
xls = pd.ExcelFile(EXCEL_PATH)

for cow_id in xls.sheet_names:
    df = xls.parse(cow_id)

    # 2-1) 열 이름을 소문자+공백제거 후, 핵심 컬럼 이름으로 통일
    df.columns = df.columns.str.lower().str.strip()
    df = df.rename(columns={
        next(c for c in df if 'date'      in c): 'datetime',        # 센서 시각
        next(c for c in df if 'activity'  in c): 'activity',        # 활동량
        next(c for c in df if 'rumination' in c): 'rumination_time',# 반추시간
        next(c for c in df if 'calving'   in c): 'calving_date',    # 실제 분만일
    })

    # 2-2) 기본 결측(빈칸) 제거
    df = df.dropna(subset=['datetime', 'activity', 'rumination_time'])
    if df.empty:         # 소 한 마리 시트가 비어 있으면 건너뜀
        continue

    # 2-3) 시계열 정렬 + 선형 보간 → 센서 누락 구간 메움
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').set_index('datetime')
    df[['activity', 'rumination_time']] = (
        df[['activity', 'rumination_time']]
          .interpolate(method='time')  # 시간축 기준 선형보간
          .ffill().bfill()             # 양 끝 NaN 채움
    )
    df = df.reset_index()

    # 2-4) 실제 분만(calving_date) 이후 데이터 제외 (과적합 방지)
    df['calving_date'] = pd.to_datetime(df['calving_date'])
    calving = df['calving_date'].iloc[0]
    df = df[df['datetime'] < calving]
    if df.empty:
        continue

    # 2-5) '잔여 일수' 클래스로 변환 (1~15, >15)
    def remaining_class(ts):
        days = (calving - ts).total_seconds() / 86400   # 초→일
        cls  = int(np.floor(days)) + 1                  # 0일→1, 1일→2 …
        return min(max(cls, 1), NUM_CLASSES)
    df['remain_cls'] = df['datetime'].apply(remaining_class)

    # 잔여일이 1일(당일)만 있는 경우는 학습에 의미 없어 제외
    if df['remain_cls'].nunique() == 1 and df['remain_cls'].iloc[0] == 1:
        continue

    # 2-6) 특징 공학 (Feature Engineering)
    # (1) 개체별 Min-Max 정규화 → 활동량·반추량을 0~1로 맞춤
    df['act_n'] = (df['activity'] - df['activity'].min()) / (
                  df['activity'].max() - df['activity'].min() + 1e-6)
    df['rum_n'] = (df['rumination_time'] - df['rumination_time'].min()) / (
                  df['rumination_time'].max() - df['rumination_time'].min() + 1e-6)

    # (2) 24시간 변화율(‘급증·급감’ 포착)
    df['act_roll24']   = df['act_n'].rolling(12, min_periods=1).mean()          # 24h 평균
    df['act_delta24h'] = (df['act_n'] - df['act_roll24']) / (df['act_roll24'] + 1e-6)
    df['rum_roll24']   = df['rum_n'].rolling(12, min_periods=1).mean()
    df['rum_delta24h'] = (df['rum_n'] - df['rum_roll24']) / (df['rum_roll24'] + 1e-6)

    # (3) 하루 주기 정보(사이클) → sin·cos
    df['hour']  = df['datetime'].dt.hour + df['datetime'].dt.minute / 60
    df['sin_t'] = np.sin(2*np.pi * df['hour']/24)
    df['cos_t'] = np.cos(2*np.pi * df['hour']/24)

    # 2-7) 필요한 열만 모아 records 리스트에 저장
    df['cow_id'] = cow_id
    records.append(df[[
        'cow_id', 'datetime',
        'act_n','rum_n',
        'act_delta24h','rum_delta24h',
        'sin_t','cos_t',
        'remain_cls'
    ]])

# 2-8) 全소(개체) 데이터 합치기
full = pd.concat(records, ignore_index=True)
full = full.dropna()                       # 혹시 남은 NaN 제거
full = full.sort_values(['cow_id','datetime'])

# ──────────────────────────────────────────────────────────────
# 3) 시퀀스(입력 시계열) 생성
# ──────────────────────────────────────────────────────────────
feature_cols = [
    'act_n','rum_n',          # 정규화된 활동·반추
    'act_delta24h','rum_delta24h',  # 24h 변화율
    'sin_t','cos_t'           # 하루 주기 정보
]
X_list, y_list = [], []

for _, grp in full.groupby('cow_id'):          # 소별로 시퀀스 자르기
    vals   = grp[feature_cols].values
    labels = grp['remain_cls'].values - 1      # 0-based 클래스
    for i in range(SEQUENCE_LEN, len(vals)):
        window = vals[i-SEQUENCE_LEN:i]        # 과거 60스텝
        if np.isnan(window).any():
            continue
        X_list.append(window)
        y_list.append(tf.one_hot(labels[i], depth=NUM_CLASSES))

X = np.array(X_list, dtype='float32')   # (샘플수, 60, 피처6)
y = np.array(y_list, dtype='float32')   # (샘플수, 16)

# ──────────────────────────────────────────────────────────────
# 4) 학습/검증 데이터 분리
# ──────────────────────────────────────────────────────────────
# 80% 학습(train) / 20% 검증(validation) — 클래스 비율 유지(stratify)
y_arg = np.argmax(y, axis=1)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_arg
)

# ──────────────────────────────────────────────────────────────
# 5) 클래스 불균형 보정 (Oversampling)
# ──────────────────────────────────────────────────────────────
train_arg = np.argmax(y_train, axis=1)
counts    = np.bincount(train_arg, minlength=NUM_CLASSES)
max_count = counts.max()                      # 가장 많은 클래스 크기
idx_pool = []
for cls in range(NUM_CLASSES):
    idx_cls = np.where(train_arg == cls)[0]   # 해당 클래스 인덱스
    if len(idx_cls) == 0:
        continue
    # 부족한 클래스는 복제(rep)해서 max_count 에 맞춤
    rep = np.random.choice(idx_cls, size=max_count, replace=True)
    idx_pool.append(rep)
oversample_idx = np.concatenate(idx_pool)

X_train_bal = X_train[oversample_idx]
y_train_bal = y_train[oversample_idx]

# ──────────────────────────────────────────────────────────────
# 6) 모델 정의 — Bidirectional LSTM
# ──────────────────────────────────────────────────────────────
# ※ LSTM(Long Short-Term Memory) 원리 요약:
#   ┌─ 시계열을 “스마트 메모장”처럼 기억 ───────────────────┐
#   │ ① 셀 상태(Cell State): 과거 정보를 흘려보내는 메모장 │
#   │ ② 게이트 3종:                                            │
#   │    • 잊기 게이트(Forget)  : 불필요한 옛 정보 삭제       │
#   │    • 입력 게이트(Input)   : 새 관측치 중 중요정보 저장   │
#   │    • 출력 게이트(Output)  : 현재 예측에 쓸 정보 선택     │
#   │ ③ Bidirectional: 과거→미래 + 미래→과거 두 방향으로 패턴 │
#   └─────────────────────────────────────────────────────────┘
input_seq = layers.Input((SEQUENCE_LEN, len(feature_cols)), name='sequence')
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(input_seq)
x = layers.Bidirectional(layers.LSTM(32))(x)        # 두 번째 LSTM
x = layers.Dense(64, activation='relu')(x)          
x = layers.Dropout(0.3)(x)                          # 과적합 방지
output = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(input_seq, output)
model.compile(
    optimizer=optimizers.Adam(1e-4),                # 학습률 0.0001 
    loss='categorical_crossentropy',                # 다중 클래스
    metrics=['accuracy']
)

# ──────────────────────────────────────────────────────────────
# 7) 학습 — EarlyStopping
# ──────────────────────────────────────────────────────────────
# val_loss 가 10 epoch 연속 개선되지 않으면 멈추고, 가장 성능 좋던
# 시점의 가중치(restore_best_weights=True) 를 자동 복원
# epoch는 각 학습을 뜻함
es = callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

history = model.fit(
    X_train_bal, y_train_bal,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=128,
    callbacks=[es]
)

# ──────────────────────────────────────────────────────────────
# 8) 검증 결과 평가
# ──────────────────────────────────────────────────────────────
probs  = model.predict(X_val)                # (검증샘플, 16)
preds  = np.argmax(probs, axis=1)            # 예측 클래스
truth  = np.argmax(y_val, axis=1)            # 실제 클래스

print("Validation accuracy:", accuracy_score(truth, preds))
print("Top-3 accuracy     :", 
      top_k_accuracy_score(truth, probs, k=3, labels=list(range(NUM_CLASSES))))

# ──────────────────────────────────────────────────────────────
# 9) 평균 확률 분포 시각화
# ──────────────────────────────────────────────────────────────
avg_prob = probs.mean(axis=0)               # 각 클래스 평균 확률
labels   = [f"{d}d" for d in range(1, HORIZON+1)] + [">15d"]

plt.figure(figsize=(10,4))
plt.bar(labels, avg_prob)
plt.xticks(rotation=45)
plt.ylabel("Probability")
plt.title("Average predicted probability per remaining day")
plt.tight_layout()
plt.show()
