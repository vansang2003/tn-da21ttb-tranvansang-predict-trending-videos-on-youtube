

# ==================== IMPORT ====================
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve
)

from xgboost import XGBClassifier

# ==================== CẤU HÌNH ====================
DATA_PATH = 'youtube_top3_features.csv'
TARGET = 'isTrending'

MODEL_PATH = 'xgb_trending_model.pkl'
SCALER_PATH = 'xgb_scaler.pkl'
FEATURES_PATH = 'selected_features.txt'
THRESHOLD_PATH = 'xgb_thresholds.pkl'

SELECTED_FEATURES = ['likeCount', 'viewCount', 'commentCount']
RANDOM_STATE = 42

# ==================== ĐỌC DỮ LIỆU ====================
print('=' * 60)
print('ĐANG ĐỌC DỮ LIỆU')
print('=' * 60)

df = pd.read_csv(DATA_PATH)

# Kiểm tra cột
required_cols = SELECTED_FEATURES + [TARGET]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f'Thiếu cột: {missing_cols}')

# Ép kiểu số
for col in SELECTED_FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Xử lý target
if df[TARGET].dtype != int:
    df[TARGET] = (
        df[TARGET].astype(str)
        .str.strip()
        .replace({'True': 1, 'False': 0, '1': 1, '0': 0})
        .astype(int)
    )

# Loại bỏ NaN
df = df.dropna(subset=SELECTED_FEATURES + [TARGET])

print(f'Số mẫu: {len(df)}')
print('Phân bố lớp:')
print(df[TARGET].value_counts().to_dict())
print()

# ==================== CHUẨN BỊ DỮ LIỆU ====================
X = df[SELECTED_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=RANDOM_STATE
)

# Tính scale_pos_weight
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos if pos > 0 else 1.0

print('=' * 60)
print('THÔNG TIN TRAIN SET')
print('=' * 60)
print(f'Negative samples: {neg}')
print(f'Positive samples: {pos}')
print(f'scale_pos_weight: {scale_pos_weight:.4f}')
print()

# Chuẩn hóa
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== GRID SEARCH ====================
print('=' * 60)
print('GRID SEARCH XGBOOST')
print('=' * 60)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.08, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'scale_pos_weight': [scale_pos_weight]
}

base_model = XGBClassifier(
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='f1',
    cv=cv,
    n_jobs=-1,
    verbose=2
)

print('Đang chạy GridSearch...')
grid_search.fit(X_train_scaled, y_train)

print()
print('KẾT QUẢ GRID SEARCH')
print('Best params:', grid_search.best_params_)
print(f'Best CV F1-score: {grid_search.best_score_:.4f}')
print()

best_model = grid_search.best_estimator_

# ==================== TỐI ƯU THRESHOLD ====================
print('=' * 60)
print('TỐI ƯU THRESHOLD')
print('=' * 60)

y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)

# F1 tối ưu
best_f1_idx = np.argmax(f1_scores)
threshold_f1 = thresholds[best_f1_idx]

# Recall >= 0.8
recall_target = 0.8
recall_idxs = np.where(recall >= recall_target)[0]
threshold_recall = thresholds[recall_idxs[-1]] if len(recall_idxs) > 0 else 0.5

# Precision >= 0.3
precision_target = 0.3
precision_idxs = np.where(precision >= precision_target)[0]
threshold_precision = thresholds[precision_idxs[0]] if len(precision_idxs) > 0 else 0.5

print(f'Threshold F1-optimal: {threshold_f1:.4f}')
print(f'Threshold Recall-oriented (≥0.8): {threshold_recall:.4f}')
print(f'Threshold Precision-oriented (≥0.3): {threshold_precision:.4f}')
print()

# ==================== ĐÁNH GIÁ THEO TỪNG THRESHOLD ====================
def evaluate_threshold(threshold, label):
    y_pred = (y_proba >= threshold).astype(int)
    print(f'--- {label} | threshold = {threshold:.2f} ---')
    print(f'Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}')
    print(f'Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}')
    print(f'F1-score:  {f1_score(y_test, y_pred, zero_division=0):.4f}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print()

evaluate_threshold(threshold_recall, 'RECALL-ORIENTED')
evaluate_threshold(threshold_f1, 'F1-OPTIMAL')
evaluate_threshold(threshold_precision, 'PRECISION-ORIENTED')

roc_auc = roc_auc_score(y_test, y_proba)
print(f'ROC-AUC (overall): {roc_auc:.4f}')
print()

# ==================== LƯU FILE ====================
print('=' * 60)
print('LƯU MODEL & CẤU HÌNH')
print('=' * 60)

joblib.dump(best_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

with open(FEATURES_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(SELECTED_FEATURES))

thresholds_info = {
    'f1_optimal': float(threshold_f1),
    'recall_oriented': float(threshold_recall),
    'precision_oriented': float(threshold_precision)
}
joblib.dump(thresholds_info, THRESHOLD_PATH)

print(f'✓ Model saved: {MODEL_PATH}')
print(f'✓ Scaler saved: {SCALER_PATH}')
print(f'✓ Features saved: {FEATURES_PATH}')
print(f'✓ Thresholds saved: {THRESHOLD_PATH}')

print('=' * 60)
print('HOÀN THÀNH')
print('=' * 60)

