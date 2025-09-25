import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib

DATA_PATH = 'data/data_youtube_trending_video.csv'
TARGET = 'isTrending'
CORR_THRESHOLD = 0.1 
MODEL_PATH = 'xgb_trending_model.pkl'
SCALER_PATH = 'xgb_pipeline.pkl'
FEATURES_PATH = 'selected_features.txt'

print('Đang đọc dữ liệu...')
df = pd.read_csv(DATA_PATH)

# Tiền xử lý tính đặc trưng đơn giản
def text_length(s): return len(str(s))
def word_count(s): return len(str(s).split())
def tag_count(s): return len(str(s).split(','))

required_cols = ['title','description','tags','categoryId','viewCount','likeCount','commentCount', TARGET]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f'Thiếu cột: {missing}')

df['title_len'] = df['title'].apply(text_length)
df['desc_len'] = df['description'].apply(text_length)
df['desc_words'] = df['description'].apply(word_count)
df['tag_count'] = df['tags'].apply(tag_count)

if df['categoryId'].dtype == object:
    df['categoryId'] = LabelEncoder().fit_transform(df['categoryId'].astype(str))

for col in ['viewCount','likeCount','commentCount','categoryId']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['title_len','desc_len','desc_words','tag_count','categoryId','viewCount','likeCount','commentCount', TARGET])

if df[TARGET].dtype != int:
    df[TARGET] = df[TARGET].astype(str).str.strip().replace({'True': 1, 'False': 0, '1': 1, '0': 0}).astype(int)

# Chọn đặc trưng theo tương quan với target
numeric_cols = ['title_len','desc_len','desc_words','tag_count','categoryId','viewCount','likeCount','commentCount', TARGET]
corr_df = df[numeric_cols].corr(numeric_only=True)
corr_to_target = corr_df[TARGET].drop(labels=[TARGET]).abs().sort_values(ascending=False)
selected_features = corr_to_target[corr_to_target >= CORR_THRESHOLD].index.tolist()
if not selected_features:
    selected_features = corr_to_target.head(5).index.tolist()

print('Selected features:', selected_features)

X = df[selected_features]
y = df[TARGET]

# Train/test split + cân bằng bằng sample_weight
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

# Chuẩn hóa
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train_scaled, y_train, sample_weight=sample_weight)

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
print(f'Đã lưu danh sách đặc trưng tại: {FEATURES_PATH}')