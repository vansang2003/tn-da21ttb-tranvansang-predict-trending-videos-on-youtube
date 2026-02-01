import pandas as pd
import os
import glob

# Nếu bạn upload trực tiếp file lên thì file nằm ở thư mục hiện tại (./)
folder_path = '.'  # Đổi lại nếu thư mục là ./csv

csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

print(f"🔍 Đã tìm thấy {len(csv_files)} file CSV:")
print(csv_files)

dataframes = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
        if not df.empty:
            dataframes.append(df)
    except Exception as e:
        print(f"❌ Lỗi khi đọc file {file}: {e}")

if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv('data_youtube-trending_video.csv', index=False, encoding='utf-8-sig')
    print("✅ Đã gộp xong các file CSV thành công!")
else:
    print("⚠️ Không có DataFrame nào để gộp.")
