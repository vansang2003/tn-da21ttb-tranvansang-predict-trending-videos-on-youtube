# Hướng dẫn sử dụng chương trình – Phân tích & Dự đoán xu hướng video YouTube

Tài liệu này hướng dẫn từ bước cài đặt môi trường đến khi chạy được ứng dụng web (Dashboard, Dự đoán trending, Dữ liệu).

---

## 1. Yêu cầu hệ thống

- **Python**: 3.10 trở lên (khuyến nghị 3.10 hoặc 3.11).
- **Hệ điều hành**: Windows, macOS hoặc Linux.
- **Trình duyệt**: Chrome, Firefox, Edge (để mở giao diện web).
- **Kết nối internet**: Cần có để gọi YouTube Data API và (tùy chọn) cài đặt gói.

---

## 2. Cài đặt

### 2.1. Clone / mở mã nguồn dự án

Nếu dùng Git:

```bash
git clone <url-repo-cua-ban>
cd Trending-video-prediction
```

Hoặc giải nén thư mục dự án và mở terminal tại thư mục gốc của dự án (nơi có file `manage.py`).

### 2.2. Tạo môi trường ảo (khuyến nghị)

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

Sau khi kích hoạt, bạn sẽ thấy `(venv)` ở đầu dòng lệnh.

### 2.3. Cài đặt thư viện Python

Trong thư mục gốc dự án (đã bật `venv`):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Dự án dùng thêm một số thư viện cho phần dự đoán (ML) và gọi API. Nếu khi chạy bị thiếu thư viện, cài bổ sung:

```bash
pip install numpy pandas scikit-learn xgboost joblib requests
```

---

## 3. Cấu hình YouTube Data API (bắt buộc để lấy dữ liệu thật)

Ứng dụng cần **YouTube Data API v3** để:

- Lấy danh sách video trending.
- Lấy thông tin video khi dự đoán (views, likes, comments).

**Các bước ngắn gọn:**

1. Vào [Google Cloud Console](https://console.cloud.google.com/), tạo project (hoặc chọn project có sẵn).
2. Bật **YouTube Data API v3**: **APIs & Services** → **Library** → tìm "YouTube Data API v3" → **Enable**.
3. Tạo API Key: **APIs & Services** → **Credentials** → **Create Credentials** → **API Key** → copy key.

**Cấu hình API Key trong project:**

- **Cách 1 (khuyến nghị):** Tạo file `.env` ở thư mục gốc dự án (cùng cấp với `manage.py`):

  ```env
  YOUTUBE_API_KEY=your_api_key_here
  ```

  Nếu project đã đọc biến môi trường từ `.env` (ví dụ qua `python-dotenv`), ứng dụng sẽ dùng key này.

- **Cách 2:** Sửa trực tiếp trong `main/views.py`: tìm biến `YOUTUBE_API_KEY` và gán API key của bạn.

Chi tiết đầy đủ (tạo project, bật API, giới hạn quota, xử lý lỗi) xem trong file **`YOUTUBE_API_SETUP.md`** ở thư mục gốc dự án.

---

## 4. File model và dữ liệu cho chức năng “Dự đoán trending”

Để trang **Dự đoán** chạy được, backend cần các file sau (đặt ở **thư mục gốc** dự án, cùng cấp với `manage.py`):

| File | Mô tả |
|------|--------|
| `xgb_trending_model.pkl` | Model XGBoost đã train để dự đoán xu hướng. |
| `xgb_scaler.pkl` | StandardScaler dùng để chuẩn hóa đặc trưng (viewCount, likeCount, commentCount). |
| `xgb_thresholds.pkl` | (Tùy chọn) Ngưỡng phân lớp; nếu không có thì dùng 0.5. |
| `selected_features.txt` | Danh sách tên đặc trưng, mỗi dòng một tên (ví dụ: likeCount, viewCount, commentCount). |

- Nếu bạn **đã có sẵn** các file này (từ người hướng dẫn hoặc bản release), chỉ cần đặt đúng thư mục và không cần train lại.
- Nếu bạn **tự train** bằng script `train_xgboost_colab.py` (ví dụ trên Google Colab):
  - Script lưu scaler với tên `xgb_pipeline.pkl`. Bạn cần **đổi tên hoặc copy** thành `xgb_scaler.pkl` để ứng dụng đọc đúng (vì trong code đang dùng `xgb_scaler.pkl`).
  - Đảm bảo `selected_features.txt` khớp với thứ tự và tên đặc trưng đã dùng khi train (thường là `likeCount`, `viewCount`, `commentCount`).

Nếu thiếu một trong các file trên, khi gửi URL video để dự đoán, trang có thể báo lỗi kiểu: “Model chưa được huấn luyện đầy đủ…”.

---

## 5. Chạy chương trình

### 5.1. Chạy server Django

Trong thư mục gốc dự án (đã kích hoạt `venv`):

```bash
python manage.py runserver
```

Hoặc chỉ định cổng:

```bash
python manage.py runserver 8080
```

Khi chạy thành công, bạn sẽ thấy dòng tương tự:

```
Starting development server at http://127.0.0.1:8000/
```

### 5.2. (Tùy chọn) Tailwind CSS

Dự án dùng **django-tailwind**. Nếu giao diện vẫn đúng, có thể không cần làm thêm. Nếu bạn chỉnh sửa CSS và cần build lại:

- Cài Node.js (nếu chưa có), rồi trong thư mục `theme/static_src` (hoặc theo hướng dẫn trong dự án):

  ```bash
  npm install
  npm run build
  ```

- Hoặc chạy lệnh mà project đã cấu hình cho tailwind (ví dụ `python manage.py tailwind build`), nếu có.

Chi tiết build Tailwind nên xem trong `README.md` ở thư mục gốc hoặc tài liệu của `django-tailwind`.

---

## 6. Truy cập ứng dụng

Mở trình duyệt và truy cập:

| Trang | URL (mặc định) | Chức năng |
|--------|----------------------------|-----------|
| **Dashboard** | http://127.0.0.1:8000/ | Tổng quan: video trending, thống kê, biểu đồ (dữ liệu từ YouTube API). |
| **Dữ liệu** | http://127.0.0.1:8000/du-lieu/ | Xem danh sách video trending, thể loại, kênh. |
| **Dự đoán** | http://127.0.0.1:8000/du-doan/ | Nhập URL video YouTube → xem xác suất trending và lý do. |
| **Tài khoản** | http://127.0.0.1:8000/tai-khoan/ | Trang tài khoản / thống kê cá nhân (nếu có). |

- Trang **Dự đoán**: nhập URL video (dạng `https://www.youtube.com/watch?v=...` hoặc Shorts), nhấn nút dự đoán. Kết quả hiển thị xác suất trending và các lý do (dựa trên viewCount, likeCount, commentCount).
- Nếu API key chưa cấu hình hoặc sai, Dashboard / Dữ liệu có thể trống hoặc báo lỗi; cần kiểm tra lại bước 3.

---

## 7. Tóm tắt quy trình từ đầu đến khi chạy được

1. **Cài đặt**: Clone/mở mã nguồn → tạo venv → `pip install -r requirements.txt` (+ các gói bổ sung nếu thiếu).
2. **API**: Tạo và bật YouTube Data API v3, tạo API Key, cấu hình vào `.env` hoặc `main/views.py`.
3. **Model**: Đảm bảo có `xgb_trending_model.pkl`, `xgb_scaler.pkl`, `selected_features.txt` (và nếu có thì `xgb_thresholds.pkl`) ở thư mục gốc; nếu train bằng Colab thì đổi tên `xgb_pipeline.pkl` → `xgb_scaler.pkl`.
4. **Chạy**: `python manage.py runserver` → mở http://127.0.0.1:8000/ và các URL ở bảng trên.

---

## 8. Một số lỗi thường gặp

- **“Model chưa được huấn luyện đầy đủ”**: Thiếu một trong các file `xgb_trending_model.pkl`, `xgb_scaler.pkl`, `selected_features.txt`. Kiểm tra tên file và vị trí (thư mục gốc); nếu vừa train thì đổi `xgb_pipeline.pkl` thành `xgb_scaler.pkl`.
- **Dashboard / Dữ liệu trống hoặc lỗi API**: Kiểm tra API Key, đã bật YouTube Data API v3 và quota (giới hạn 10,000 request/ngày). Xem thêm `YOUTUBE_API_SETUP.md`.
- **Thiếu thư viện (ModuleNotFoundError)**: Cài bổ sung `numpy pandas scikit-learn xgboost joblib requests` như ở mục 2.3.
- **Lỗi khi nhập URL video**: URL phải hợp lệ (ví dụ `youtube.com/watch?v=...` hoặc Shorts). Kiểm tra kết nối internet và API Key.

---

## 9. Tài liệu liên quan

- **`YOUTUBE_API_SETUP.md`** (thư mục gốc): Hướng dẫn chi tiết tạo và cấu hình YouTube Data API v3.
- **`train_xgboost_colab.py`**: Script mẫu để huấn luyện lại model (thường chạy trên Colab, cần file CSV có cột `likeCount`, `viewCount`, `commentCount`, `isTrending`).
- **`docs/workflow_diagram.md`** (nếu có): Sơ đồ luồng xử lý dự đoán trending.

Nếu bạn làm đúng các bước từ mục 1 đến 6, chương trình sẽ cài đặt và chạy được từ đầu đến khi sử dụng được Dashboard và Dự đoán xu hướng video.
