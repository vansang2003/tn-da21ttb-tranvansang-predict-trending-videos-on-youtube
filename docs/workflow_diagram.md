# Sơ đồ luồng hoạt động trang dự đoán trending video
## Chi tiết các bước

### 1. Input Processing
- User nhập URL YouTube (hỗ trợ cả video thường và Shorts)
- Validate URL format và extract Video ID
- Hỗ trợ các format: `youtube.com/watch?v=`, `youtu.be/`, `youtube.com/shorts/`

### 2. Data Fetching
- Gọi YouTube Data API với Video ID
- Lấy thông tin chi tiết: metadata, statistics, thumbnails
- Xử lý lỗi API (quota, network, invalid video)

### 3. Model Loading
- Load XGBoost model đã train (`xgb_trending_model.pkl`)
- Load Scaler để chuẩn hóa features (`xgb_pipeline.pkl`)
- Load danh sách features đã chọn (`selected_features.txt`)

### 4. Feature Engineering
- Tính toán các đặc trưng từ thông tin video:
  - Text features: title_len, desc_len, desc_words, tag_count
  - Engagement features: like_rate, comment_rate
  - Raw metrics: viewCount, likeCount, commentCount
- Sắp xếp theo đúng thứ tự đã train

### 5. Prediction
- Chuẩn hóa features với StandardScaler
- Dự đoán với XGBoost model
- Tính xác suất trending (0-1)

### 6. Explanation Generation
- Phân tích từng đặc trưng theo ngưỡng khoa học
- So sánh với ngưỡng tối ưu (ROC/Youden)
- Tạo lý do tăng/giảm xác suất trending

### 7. UI Display
- Hiển thị thông tin video với thumbnail
- Hiển thị kết quả dự đoán với xác suất
- Hiển thị lý do chi tiết và thuyết phục

## Công nghệ sử dụng

- **Backend**: Django + Python
- **ML**: XGBoost + Scikit-learn
- **API**: YouTube Data API v3
- **Frontend**: HTML + Tailwind CSS
- **Data**: CSV dataset cho training

