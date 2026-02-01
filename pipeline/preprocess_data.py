
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("YOUTUBE TRENDING VIDEO DATA PREPROCESSING")
    print("="*60)

    # Đọc dữ liệu
    print("\n Đang đọc dữ liệu...")
    try:
        df = pd.read_csv('data_youtube_trending_video.csv')
        print(f"Đã đọc thành công: {df.shape[0]} dòng × {df.shape[1]} cột")
    except FileNotFoundError:
        print("Không tìm thấy file 'data_youtube_trending_video.csv'")
        return
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return

    # Hiển thị 10 dòng đầu tiên
    print("\n " + "="*50)
    print("10 DÒNG ĐẦU TIÊN")
    print("="*50)
    print(df.head(10).to_string(index=False))

    # Kiểm tra thông tin tổng quan
    print("\n" + "="*50)
    print("THÔNG TIN TỔNG QUAN VỀ DATASET")
    print("="*50)
    print(f"Kích thước: {df.shape}")
    print(f"\n Kiểu dữ liệu các cột:")
    print(df.dtypes)

    # Kiểm tra missing values
    print("\n" + "-"*30)
    print("MISSING VALUES:")
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    if len(missing_cols) > 0:
        print(missing_cols)
    else:
        print("Không có missing values")

    # Tạo mapping cho categoryId
    category_mapping = {
        1: 'Film & Animation',
        2: 'Autos & Vehicles',
        10: 'Music',
        15: 'Pets & Animals',
        17: 'Sports',
        18: 'Short Movies',
        19: 'Travel & Events',
        20: 'Gaming',
        21: 'Videoblogging',
        22: 'People & Blogs',
        23: 'Comedy',
        24: 'Entertainment',
        25: 'News & Politics',
        26: 'Howto & Style',
        27: 'Education',
        28: 'Science & Technology',
        29: 'Nonprofits & Activism',
        30: 'Movies',
        31: 'Anime/Animation',
        32: 'Action/Adventure',
        33: 'Classics',
        34: 'Comedy',
        35: 'Documentary',
        36: 'Drama',
        37: 'Family',
        38: 'Foreign',
        39: 'Horror',
        40: 'Sci-Fi/Fantasy',
        41: 'Thriller',
        42: 'Shorts',
        43: 'Shows',
        44: 'Trailers'
    }

    print("\n Bắt đầu tiền xử lý dữ liệu...")

    # Tạo bản sao để xử lý
    df_processed = df.copy()
    original_size = df_processed.shape[0]

    # 1. Chuyển đổi categoryId sang tên thể loại
    df_processed['categoryName'] = df_processed['categoryId'].map(category_mapping)
    print("1/7: Đã chuyển đổi categoryId sang categoryName")

    # 2. Hàm chuyển đổi duration từ ISO 8601 sang giây
    def duration_to_seconds(duration):
        if pd.isna(duration):
            return np.nan

        # Pattern để parse ISO 8601 duration
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, str(duration))

        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)

            total_seconds = hours * 3600 + minutes * 60 + seconds
            return total_seconds

        return np.nan

    # 4. Chuyển đổi duration sang giây
    df_processed['duration_seconds'] = df_processed['duration'].apply(duration_to_seconds)
    print("2/7: Đã chuyển đổi duration sang giây")

    # 5. Chuyển đổi publishedAt về định dạng datetime chuẩn
    df_processed['publishedAt'] = pd.to_datetime(df_processed['publishedAt'], errors='coerce')
    print("3/7: Đã chuyển đổi publishedAt về định dạng datetime")

    # 6. Chuyển đổi các cột số về dạng số
    numeric_columns = ['viewCount', 'likeCount', 'commentCount']
    for col in numeric_columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    print("4/7: Đã chuyển đổi các cột số về dạng numeric")

    # 8. Loại bỏ các dòng có giá trị 0 hoặc null
    print("\n Loại bỏ dữ liệu không hợp lệ...")

    # Xác định các cột quan trọng cần kiểm tra
    critical_columns = ['viewCount', 'likeCount', 'commentCount', 'duration_seconds', 'publishedAt']

    # Loại bỏ dòng có null trong các cột quan trọng
    df_clean = df_processed.dropna(subset=critical_columns)
    print(f"6/7: Sau khi loại bỏ null: {df_clean.shape[0]} dòng (từ {original_size})")

    # Loại bỏ dòng có giá trị 0 trong các cột quan trọng
    for col in ['viewCount', 'likeCount', 'commentCount']:
        df_clean = df_clean[df_clean[col] > 0]

    print(f"7/7: Sau khi loại bỏ giá trị 0: {df_clean.shape[0]} dòng")

    # Loại bỏ duration = 0
    df_clean = df_clean[df_clean['duration_seconds'] > 0]
    final_size = df_clean.shape[0]
    print(f"Sau khi loại bỏ duration = 0: {final_size} dòng")

    # Hiển thị kết quả tổng hợp
    print("\n" + "="*60)
    print("KẾT QUẢ TIỀN XỬ LÝ")
    print("="*60)
    print(f"Dataset gốc: {original_size} dòng")
    print(f"Dataset sau xử lý: {final_size} dòng")
    print(f"Tỷ lệ giữ lại: {(final_size/original_size*100):.1f}%")

    trending_count = df_clean['isTrending'].sum()
    non_trending_count = final_size - trending_count
    trending_ratio = trending_count / final_size if final_size > 0 else 0

    print(f"\n Video trending: {trending_count} ({trending_ratio:.3f})")
    print(f"Video không trending: {non_trending_count}")

    # Kiểm tra missing values cuối cùng
    final_missing = df_clean.isnull().sum().sum()
    print(f"Missing values còn lại: {final_missing}")

    # Hiển thị sample dữ liệu sau xử lý
    print("\n" + "="*50)
    print("SAMPLE DỮ LIỆU SAU XỬ LÝ")
    print("="*50)
    sample_cols = ['videoId', 'title', 'channelTitle', 'publishedAt', 'categoryName',
                   'duration_seconds', 'viewCount', 'likeCount', 'commentCount', 'isTrending']
    print(df_clean[sample_cols].head(5).to_string(index=False))
    print("\n...")
    print(df_clean[sample_cols].tail(5).to_string(index=False))

    # Xuất file CSV đã xử lý
    output_filename = 'youtube_trending_data_processed.csv'
    df_clean.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n Đã xuất file: {output_filename}")
    print(f"Kích thước: {df_clean.shape}")

    print("\n HOÀN THÀNH TIỀN XỬ LÝ DỮ LIỆU!")
    print("File đã được lưu: youtube_trending_data_processed.csv")


if __name__ == "__main__":
    main()