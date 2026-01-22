import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib

# ==================== CẤU HÌNH ====================
DATA_PATH = 'youtube_top3_features.csv'  # Upload file này lên Colab
TARGET = 'isTrending'
MODEL_PATH = 'xgb_trending_model.pkl'
SCALER_PATH = 'xgb_pipeline.pkl'
FEATURES_PATH = 'selected_features.txt'

# ==================== ĐỌC DỮ LIỆU ====================
print('=' * 60)
print('Đang đọc dữ liệu...')
print('=' * 60)

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
print(f'Phân bố lớp:')
print(df[TARGET].value_counts().to_dict())
print()

# ==================== CHUẨN BỊ DỮ LIỆU ====================
X = df[selected_features]
y = df[TARGET]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Tính scale_pos_weight để xử lý mất cân bằng dữ liệu
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0

print('=' * 60)
print('THÔNG TIN DỮ LIỆU TRAIN')
print('=' * 60)
print(f'Số lượng negative samples: {negative_count}')
print(f'Số lượng positive samples: {positive_count}')
print(f'scale_pos_weight: {scale_pos_weight:.4f}')
print()

# Chuẩn hóa
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== GRID SEARCH ====================
print('=' * 60)
print('BẮT ĐẦU GRID SEARCH...')
print('=' * 60)

# Định nghĩa grid search parameters
# Thêm giá trị vào các tham số quan trọng nhất để tối ưu mô hình
param_grid = {
    'n_estimators': [100, 200],  # Giữ 2 giá trị (đủ để tìm được giá trị tốt)
    'max_depth': [3, 4, 5],  # Thêm 4 - tham số quan trọng nhất, ảnh hưởng lớn đến overfitting
    'learning_rate': [0.05, 0.08, 0.1],  # Thêm 0.08 - tham số quan trọng, ảnh hưởng đến tốc độ hội tụ
    'subsample': [0.8, 0.9],  # Giữ 2 giá trị (đủ)
    'colsample_bytree': [0.8, 0.9],  # Giữ 2 giá trị (đủ)
    'scale_pos_weight': [scale_pos_weight]  # Giữ cố định giá trị đã tính
}

# Tạo base model
base_model = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1  # Sử dụng tất cả CPU cores
)

# Sử dụng StratifiedKFold để đảm bảo phân bố lớp đều trong cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV với scoring là f1 (phù hợp cho dữ liệu mất cân bằng)
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=cv,
    scoring='f1',  # Sử dụng F1-score vì dữ liệu mất cân bằng
    n_jobs=-1,
    verbose=2  # Hiển thị tiến trình
)

print('Đang tìm tham số tối ưu...')
print(f'Số lượng tổ hợp tham số: {len(param_grid["n_estimators"]) * len(param_grid["max_depth"]) * len(param_grid["learning_rate"]) * len(param_grid["subsample"]) * len(param_grid["colsample_bytree"])}')
print('Điều này có thể mất vài phút...')
print()

# Thực hiện grid search
grid_search.fit(X_train_scaled, y_train)

print('=' * 60)
print('KẾT QUẢ GRID SEARCH')
print('=' * 60)
print(f'Tham số tốt nhất: {grid_search.best_params_}')
print(f'F1-score tốt nhất (CV): {grid_search.best_score_:.4f}')
print()

# Lấy model tốt nhất
best_model = grid_search.best_estimator_

# ==================== ĐÁNH GIÁ MÔ HÌNH ====================
print('=' * 60)
print('ĐÁNH GIÁ MÔ HÌNH TRÊN TEST SET')
print('=' * 60)

y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

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
print()
print('Confusion Matrix:')
print(cm)
print()
print('Classification Report:')
print(classification_report(y_test, y_pred, digits=4))
print()

# ==================== LƯU MÔ HÌNH ====================
print('=' * 60)
print('LƯU MÔ HÌNH VÀ CÁC FILE CẦN THIẾT')
print('=' * 60)

# Lưu mô hình tốt nhất
joblib.dump(best_model, MODEL_PATH)
print(f'✓ Đã lưu mô hình tại: {MODEL_PATH}')

# Lưu scaler
joblib.dump(scaler, SCALER_PATH)
print(f'✓ Đã lưu scaler tại: {SCALER_PATH}')

# Lưu danh sách đặc trưng
with open(FEATURES_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(selected_features))
print(f'✓ Đã lưu danh sách đặc trưng tại: {FEATURES_PATH}')

print()
print('=' * 60)
print('HOÀN THÀNH!')
print('=' * 60)
print('Bạn có thể tải xuống 3 file sau:')
print(f'  1. {MODEL_PATH}')
print(f'  2. {SCALER_PATH}')
print(f'  3. {FEATURES_PATH}')
print('=' * 60)
