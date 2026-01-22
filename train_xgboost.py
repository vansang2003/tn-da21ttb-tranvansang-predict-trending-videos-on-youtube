import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib

DATA_PATH = 'data/youtube_top3_features.csv'
TARGET = 'isTrending'
MODEL_PATH = 'xgb_trending_model.pkl'
SCALER_PATH = 'xgb_pipeline.pkl'
FEATURES_PATH = 'selected_features.txt'

print('Đang đọc dữ liệu...')
df = pd.read_csv(DATA_PATH)

# Chọn 3 đặc trưng đã được chọn sẵn: likeCount, viewCount, commentCount
selected_features = ['likeCount', 'viewCount', 'commentCount']

# Kiểm tra các cột có tồn tại không
required_cols = selected_features + [TARGET]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f'Thiếu cột: {missing}')

# Chuyển đổi sang số
for col in selected_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Xử lý target
if df[TARGET].dtype != int:
    df[TARGET] = df[TARGET].astype(str).str.strip().replace({'True': 1, 'False': 0, '1': 1, '0': 0}).astype(int)

# Loại bỏ các dòng có giá trị NaN
df = df.dropna(subset=selected_features + [TARGET])

print(f'Số lượng mẫu: {len(df)}')
print(f'Phân bố lớp: {df[TARGET].value_counts().to_dict()}')

X = df[selected_features]
y = df[TARGET]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Tính scale_pos_weight để xử lý mất cân bằng dữ liệu
# scale_pos_weight = số lượng negative samples / số lượng positive samples
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0

print(f'Số lượng negative samples: {negative_count}')
print(f'Số lượng positive samples: {positive_count}')
print(f'scale_pos_weight: {scale_pos_weight:.4f}')

# Chuẩn hóa
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện XGBoost với scale_pos_weight
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train_scaled, y_train)

# Dự đoán & đánh giá
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
try:
    roc_auc = roc_auc_score(y_test, y_proba)
except Exception:
    roc_auc = float('nan')
cm = confusion_matrix(y_test, y_pred)

print(f'Accuracy:  {acc:.4f}')
print(f'Precision: {prec:.4f}')
print(f'Recall:    {rec:.4f}')
print(f'F1-score:  {f1:.4f}')
print(f'ROC-AUC:   {roc_auc:.4f}')
print('Confusion Matrix:\n', cm)
print('Classification Report:\n', classification_report(y_test, y_pred, digits=4))

# Lưu mô hình, scaler và danh sách đặc trưng để dùng dự đoán sau này
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
with open(FEATURES_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(selected_features))
print(f'Đã lưu mô hình tại: {MODEL_PATH}')
print(f'Đã lưu scaler tại: {SCALER_PATH}')
