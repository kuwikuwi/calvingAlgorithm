# analyze_class_distribution.py
import glob, os, re, pandas as pd, numpy as np
import matplotlib.pyplot as plt

BASE_DIR   = os.path.dirname(__file__)
TRAIN_DIR  = os.path.join(BASE_DIR, "트레이닝데이터")

# ───────────────────────────────────────
# 1. 예측파일 ⟷ 대응되는 validation 원본파일 로드
# ───────────────────────────────────────
pred_paths = sorted(glob.glob(os.path.join(BASE_DIR,
                                           "pred_validation_fold_*.xlsx")))
val_paths  = sorted(glob.glob(os.path.join(TRAIN_DIR,
                                           "validation_fold_*.xlsx")))

def fold_id(p):               # “…_fold_3.xlsx” → "3"
    return re.search(r"_fold_(\d+)", os.path.basename(p)).group(1)

val_map = {fold_id(p): p for p in val_paths}

#── 작은 헬퍼 : 원본 validation 에서 remain_cls 라벨 계산
def make_remain_cls(df: pd.DataFrame) -> pd.Series:
    if 'CalvingDate' not in df.columns and 'bundate' in df.columns:
        df = df.rename(columns={'bundate': 'CalvingDate'})
    cd = pd.to_datetime(df['CalvingDate'], errors='coerce')
    md = pd.to_datetime(df['mdate'],       errors='coerce')
    d  = (cd - md).dt.days
    # 1-15 ↔ 그대로, 16은 >15
    return d.clip(lower=1).where(d <= 15, 16)

records = []
for p_pred in pred_paths:
    fid       = fold_id(p_pred)              # "1" ~ "5"
    p_val     = val_map[fid]                 # 대응 validation 경로
    pred_df   = pd.read_excel(p_pred)
    val_df    = pd.read_excel(p_val)

    val_df['remain_cls'] = make_remain_cls(val_df)
    merged = pd.merge(val_df[['name','remain_cls']],
                      pred_df[['name','pred_class']], on='name')
    merged['fold'] = fid
    records.append(merged)

data = pd.concat(records, ignore_index=True)

# ───────────────────────────────────────
# 2. 클래스 분포 / 편향 정도 출력
# ───────────────────────────────────────
true_cnt  = (data['remain_cls']
             .value_counts()
             .reindex(range(1,17), fill_value=0)
             .sort_index())
pred_cnt  = (data['pred_class']
             .value_counts()
             .reindex(range(1,17), fill_value=0)
             .sort_index())

print("■ true label 분포 (validation 전체)")
print(pd.DataFrame({'count': true_cnt,
                    'ratio': (true_cnt/len(data)).round(3)}))

print("\n■ top-1 예측 분포")
print(pd.DataFrame({'count': pred_cnt,
                    'ratio': (pred_cnt/len(data)).round(3)}))

# ───────────────────────────────────────
# 3. True vs Predicted 분포 그래프
# ───────────────────────────────────────
labels = [f"{d}d" for d in range(1,16)] + [">15d"]
x      = np.arange(16)

plt.figure(figsize=(10,4))
plt.bar(x-0.15, true_cnt.values/len(data), width=0.3,
        label='True',  edgecolor='k')
plt.bar(x+0.15, pred_cnt.values/len(data), width=0.3,
        label='Pred',  edgecolor='k')
plt.xticks(x, labels, rotation=45)
plt.ylabel("Frequency (ratio)")
plt.title("Class distribution – True vs Predicted (validation)")
plt.legend();  plt.tight_layout();  plt.show()

# ───────────────────────────────────────
# 4. fold 별 정확도(선택) ─ 이미 remain_cls 보유
# ───────────────────────────────────────
fold_acc1 = (data['pred_class'] == data['remain_cls']).groupby(data['fold']).mean()

top3_hit = data.apply(
    lambda r: r['remain_cls'] in
    (r.get('pred_top3_1'), r.get('pred_top3_2'), r.get('pred_top3_3')),
    axis=1
)
fold_acc3 = top3_hit.groupby(data['fold']).mean()

plt.figure(figsize=(6,4))
x = np.arange(len(fold_acc1))
plt.bar(x-0.15, fold_acc1.values, width=0.3, label="Top-1")
plt.bar(x+0.15, fold_acc3.values, width=0.3, label="Top-3")
plt.ylim(0,1)
plt.xticks(x, [f"fold {f}" for f in fold_acc1.index])
plt.ylabel("Accuracy")
plt.title("Fold-wise accuracy (validation)")
plt.legend();  plt.tight_layout();  plt.show()
