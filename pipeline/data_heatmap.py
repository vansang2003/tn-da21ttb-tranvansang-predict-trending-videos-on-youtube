import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*50)
    print("HEATMAP - YOUTUBE TRENDING DATA")
    print("="*50)

    # Đọc dữ liệu
    print("\n Đang đọc dữ liệu...")
    try:
        df = pd.read_csv('youtube_trending_data_processed.csv')
        print(f" Đã đọc thành công: {df.shape[0]:,} dòng × {df.shape[1]} cột")
    except FileNotFoundError:
        print(" Không tìm thấy file 'youtube_trending_data_processed.csv'")
        return

    # Chuẩn bị tất cả cột cho heatmap
    print("\n Đang chuẩn bị dữ liệu cho heatmap...")

    # Copy dataframe
    df_heatmap = df.copy()

    # Mã hóa các cột categorical
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    # Mã hóa categoryName
    if 'categoryName' in df_heatmap.columns:
        df_heatmap['category_encoded'] = le.fit_transform(df_heatmap['categoryName'])
        print(" Đã mã hóa categoryName")

    # Mã hóa channelTitle (chỉ top channels để tránh quá nhiều categories)
    if 'channelTitle' in df_heatmap.columns:
        top_channels = df_heatmap['channelTitle'].value_counts().head(50).index
        df_heatmap['channel_encoded'] = df_heatmap['channelTitle'].map(
            lambda x: le.fit_transform([x])[0] if x in top_channels else -1
        )
        print(" Đã mã hóa top 50 channels")

    # Xử lý publishedAt nếu là datetime
    if 'publishedAt' in df_heatmap.columns and df_heatmap['publishedAt'].dtype == 'object':
        try:
            df_heatmap['publishedAt'] = pd.to_datetime(df_heatmap['publishedAt'], errors='coerce')
            min_date = df_heatmap['publishedAt'].min()
            df_heatmap['days_since_start'] = (df_heatmap['publishedAt'] - min_date).dt.days
            print(" Đã chuyển publishedAt thành days_since_start")
        except:
            print(" Không thể xử lý publishedAt")

    # Chọn tất cả cột số và encoded
    all_cols = []

    # Thêm các cột số gốc
    numeric_cols = ['duration_seconds', 'viewCount', 'likeCount', 'commentCount']
    for col in numeric_cols:
        if col in df_heatmap.columns:
            all_cols.append(col)

    # Thêm các cột encoded
    encoded_cols = ['category_encoded', 'channel_encoded', 'days_since_start']
    for col in encoded_cols:
        if col in df_heatmap.columns:
            all_cols.append(col)

    # Thêm isTrending nếu có
    if 'isTrending' in df_heatmap.columns:
        all_cols.append('isTrending')


    # Tính correlation matrix
    corr_matrix = df_heatmap[all_cols].corr()

    # Tạo heatmap với kích thước phù hợp
    n_cols = len(all_cols)
    figsize = (max(12, n_cols * 0.8), max(10, n_cols * 0.7))
    plt.figure(figsize=figsize)

    # Heatmap với mask (ẩn phần trùng lặp)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Điều chỉnh font size dựa trên số lượng cột
    annot_fontsize = max(6, 12 - n_cols * 0.5)

    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": .8}, fmt='.3f',
                annot_kws={"size": annot_fontsize})

    plt.title(f'Complete Correlation Heatmap - {n_cols} YouTube Features',
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=max(8, 12 - n_cols * 0.3))
    plt.yticks(rotation=0, fontsize=max(8, 12 - n_cols * 0.3))
    plt.tight_layout()

    # Hiển thị correlation values
    print("\n Ma trận correlation:")
    print("-" * 40)
    print(corr_matrix.round(3))


if __name__ == "__main__":
    main()