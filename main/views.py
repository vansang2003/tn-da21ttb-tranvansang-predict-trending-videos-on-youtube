import os
import re
import joblib
import requests
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import pandas as pd
from django.shortcuts import render
import numpy as np

# YouTube API Configuration - Sử dụng API key cố định
YOUTUBE_API_KEY = 'AIzaSyCt8o0RjUnvbzwVQUuKhj9E1sa8glhKgdU' 
MODEL_PATH = 'xgb_trending_model.pkl'
SCALER_PATH = 'xgb_scaler.pkl'
THRESHOLDS_PATH = 'xgb_thresholds.pkl'
FEATURES_PATH = 'selected_features.txt'

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    # Hỗ trợ cả shorts: https://www.youtube.com/shorts/<id>
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/shorts\/)([^&\n?#\/]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_info(video_id, api_key=None):
    """Get video information from YouTube Data API"""
    if not api_key:
        api_key = YOUTUBE_API_KEY
    
    url = f"https://www.googleapis.com/youtube/v3/videos"
    params = {
        'part': 'snippet,statistics,contentDetails',
        'id': video_id,
        'key': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['items']:
            video = data['items'][0]
            snippet = video['snippet']
            statistics = video['statistics']
            
            return {
                'title': snippet.get('title', ''),
                'description': snippet.get('description', ''),
                'channel': snippet.get('channelTitle', ''),
                'categoryId': snippet.get('categoryId', ''),
                'tags': ', '.join(snippet.get('tags', [])),
                'viewCount': int(statistics.get('viewCount', 0)),
                'likeCount': int(statistics.get('likeCount', 0)),
                'commentCount': int(statistics.get('commentCount', 0)),
                'publishedAt': snippet.get('publishedAt', ''),
                'thumbnails': snippet.get('thumbnails', {}),
                'duration': video['contentDetails'].get('duration', '')
            }
    except Exception as e:
        print(f"Error fetching video info: {e}")
        return None
    
    return None

def get_trending_videos(api_key=None, region_code='VN', max_results=50):
    """Get trending videos from YouTube API"""
    if not api_key:
        api_key = YOUTUBE_API_KEY
    
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        'part': 'snippet,statistics,contentDetails',
        'chart': 'mostPopular',
        'regionCode': region_code,
        'maxResults': max_results,
        'key': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        videos = []
        for item in data.get('items', []):
            snippet = item['snippet']
            statistics = item['statistics']
            
            videos.append({
                'id': item['id'],
                'title': snippet.get('title', ''),
                'channel': snippet.get('channelTitle', ''),
                'categoryId': snippet.get('categoryId', ''),
                'viewCount': int(statistics.get('viewCount', 0)),
                'likeCount': int(statistics.get('likeCount', 0)),
                'commentCount': int(statistics.get('commentCount', 0)),
                'publishedAt': snippet.get('publishedAt', ''),
                'thumbnails': snippet.get('thumbnails', {}),
                'description': snippet.get('description', '')
            })
        
        return videos
    except Exception as e:
        print(f"Error fetching trending videos: {e}")
        return []

def get_channel_stats(channel_id, api_key=None):
    """Get channel statistics"""
    if not api_key:
        api_key = YOUTUBE_API_KEY
    
    url = "https://www.googleapis.com/youtube/v3/channels"
    params = {
        'part': 'statistics,snippet',
        'id': channel_id,
        'key': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['items']:
            item = data['items'][0]
            statistics = item['statistics']
            snippet = item['snippet']
            
            return {
                'name': snippet.get('title', ''),
                'subscriberCount': int(statistics.get('subscriberCount', 0)),
                'viewCount': int(statistics.get('viewCount', 0)),
                'videoCount': int(statistics.get('videoCount', 0)),
                'description': snippet.get('description', ''),
                'thumbnails': snippet.get('thumbnails', {})
            }
    except Exception as e:
        print(f"Error fetching channel stats: {e}")
        return None
    
    return None

def get_category_info(category_id, api_key=None):
    """Get category information"""
    if not api_key:
        api_key = YOUTUBE_API_KEY
    
    url = "https://www.googleapis.com/youtube/v3/videoCategories"
    params = {
        'part': 'snippet',
        'id': category_id,
        'key': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['items']:
            return data['items'][0]['snippet']['title']
    except Exception as e:
        print(f"Error fetching category info: {e}")
    
    return "Unknown"

def get_dashboard_data(api_key=None):
    """Get comprehensive dashboard data from YouTube API"""
    if not api_key:
        api_key = YOUTUBE_API_KEY
    
    try:
        # Get trending videos
        trending_videos = get_trending_videos(api_key, max_results=50)
        
        # Calculate statistics
        total_views = sum(video['viewCount'] for video in trending_videos)
        total_likes = sum(video['likeCount'] for video in trending_videos)
        total_comments = sum(video['commentCount'] for video in trending_videos)
        
        # Category distribution with percentage calculation
        categories = {}
        for video in trending_videos:
            category = get_category_info(video['categoryId'], api_key)
            categories[category] = categories.get(category, 0) + 1
        
        # Calculate percentages for categories
        total_videos = len(trending_videos)
        categories_with_percentage = {}
        for category, count in categories.items():
            percentage = (count / total_videos * 100) if total_videos > 0 else 0
            categories_with_percentage[category] = {
                'count': count,
                'percentage': round(percentage, 1)
            }
        
        # Top channels
        channel_stats = {}
        for video in trending_videos[:10]:  # Top 10 videos
            channel_name = video['channel']
            if channel_name not in channel_stats:
                channel_stats[channel_name] = {
                    'name': channel_name,
                    'videoCount': 0,
                    'totalViews': 0,
                    'totalLikes': 0
                }
            channel_stats[channel_name]['videoCount'] += 1
            channel_stats[channel_name]['totalViews'] += video['viewCount']
            channel_stats[channel_name]['totalLikes'] += video['likeCount']
        
        # Sort channels by total views
        top_channels = sorted(channel_stats.values(), key=lambda x: x['totalViews'], reverse=True)[:5]
        
        # Geographic distribution (simulated - YouTube API doesn't provide this directly)
        geographic_data = {
            'Hoa Kỳ': 28.5,
            'Ấn Độ': 22.1,
            'Brazil': 15.8,
            'Nhật Bản': 12.3,
            'Anh': 8.7
        }
        
        # Platform performance (simulated)
        platform_data = {
            'Desktop': 45.2,
            'Mobile': 38.7,
            'Tablet': 16.1
        }
        
        # Trending keywords (simulated - would need additional API calls)
        trending_keywords = [
            {'keyword': '#gaming', 'growth': 156},
            {'keyword': '#music', 'growth': 89},
            {'keyword': '#tutorial', 'growth': 67},
            {'keyword': '#comedy', 'growth': 45},
            {'keyword': '#news', 'growth': -12}
        ]
        
        return {
            'trending_videos': trending_videos,
            'total_videos': len(trending_videos),
            'total_views': total_views,
            'total_likes': total_likes,
            'total_comments': total_comments,
            'avg_views': total_views // len(trending_videos) if trending_videos else 0,
            'avg_likes': total_likes // len(trending_videos) if trending_videos else 0,
            'avg_comments': total_comments // len(trending_videos) if trending_videos else 0,
            'categories': categories_with_percentage,
            'top_channels': top_channels,
            'geographic_data': geographic_data,
            'platform_data': platform_data,
            'trending_keywords': trending_keywords,
            'live_viewers': 2400000,  # Simulated
            'live_channels': 1234,    # Simulated
            'upload_frequency': {
                'weekly': 156,
                'daily': 22.3
            }
        }
        
    except Exception as e:
        print(f"Error getting dashboard data: {e}")
        return {
            'trending_videos': [],
            'total_videos': 0,
            'total_views': 0,
            'total_likes': 0,
            'total_comments': 0,
            'avg_views': 0,
            'avg_likes': 0,
            'avg_comments': 0,
            'categories': {},
            'top_channels': [],
            'geographic_data': {},
            'platform_data': {},
            'trending_keywords': [],
            'live_viewers': 0,
            'live_channels': 0,
            'upload_frequency': {'weekly': 0, 'daily': 0}
        }

def extract_features(info):
    """Extract features from video info for ML model - chỉ 3 đặc trưng: likeCount, viewCount, commentCount"""
    if not info:
        return None
    
    # Chỉ lấy 3 đặc trưng đã được train: likeCount, viewCount, commentCount
    view_count = info.get('viewCount', 0)
    like_count = info.get('likeCount', 0)
    comment_count = info.get('commentCount', 0)
    
    return {
        'viewCount': view_count,
        'likeCount': like_count,
        'commentCount': comment_count
    }

def load_model_bundle():
    """Load trained model, scaler, thresholds and selected feature names."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH)):
        return None, None, None, None
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Load thresholds nếu có, nếu không thì dùng 0.5 mặc định
    threshold = 0.5
    if os.path.exists(THRESHOLDS_PATH):
        try:
            thresholds_data = joblib.load(THRESHOLDS_PATH)
            # Nếu thresholds_data là dict, lấy threshold từ key phù hợp
            if isinstance(thresholds_data, dict):
                # Có thể có các key như 'optimal_threshold', 'threshold', 'best_threshold'
                threshold = thresholds_data.get('optimal_threshold', thresholds_data.get('threshold', thresholds_data.get('best_threshold', 0.5)))
            elif isinstance(thresholds_data, (int, float)):
                threshold = float(thresholds_data)
        except Exception as e:
            print(f"Warning: Không thể load thresholds, sử dụng 0.5 mặc định: {e}")
    
    with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
        selected_features = [line.strip() for line in f if line.strip()]
    
    return model, scaler, selected_features, threshold

def index(request):
    """Dashboard view with real YouTube data"""
    # Get dashboard data using fixed API key
    dashboard_data = get_dashboard_data()
    
    context = {
        'dashboard_data': dashboard_data
    }
    
    return render(request, 'main/index.html', context)

def predict(request):
    """Video prediction view"""
    result = None
    error = None
    video_info = None
    
    if request.method == 'POST':
        api_key = request.POST.get('api_key', YOUTUBE_API_KEY)
        url = request.POST.get('url')
        
        if not url:
            error = "Vui lòng nhập URL video YouTube"
        else:
            video_id = extract_video_id(url)
            if not video_id:
                error = "URL video không hợp lệ"
            else:
                video_info = get_video_info(video_id, api_key)
                if not video_info:
                    error = "Không thể lấy thông tin video. Kiểm tra lại API key và URL"
                else:
                    try:
                        model, scaler, selected_features, threshold = load_model_bundle()
                        if not (model and scaler and selected_features):
                            error = "Model chưa được huấn luyện đầy đủ (model/scaler/features). Vui lòng train model trước"
                        else:
                            # Extract features - chỉ 3 đặc trưng: likeCount, viewCount, commentCount
                            raw = extract_features(video_info)
                            if raw:
                                # Sắp xếp features đúng thứ tự đã dùng khi train
                                vector = [raw.get(name, 0) for name in selected_features]
                                arr = np.array(vector, dtype=float).reshape(1, -1)
                                # Scale & predict
                                arr_scaled = scaler.transform(arr)
                                proba = float(model.predict_proba(arr_scaled)[0][1]) if hasattr(model, 'predict_proba') else float(model.predict(arr_scaled)[0])
                                # Sử dụng threshold từ file xgb_thresholds.pkl thay vì 0.5 cố định
                                pred = int(proba >= threshold)
                                score = int(proba * 100)

                                
                                ups_list = []
                                downs_list = []
                                # Chỉ phân tích 3 đặc trưng: viewCount, likeCount, commentCount
                                # viewCount
                                if 'viewCount' in raw:
                                    v = raw['viewCount']
                                    if v >= 1000000:
                                        ups_list.append(f'Lượt xem rất cao ({v/1000000:.1f}M lượt)')
                                    elif v >= 100000:
                                        ups_list.append(f'Lượt xem cao ({v/1000:.0f}K lượt)')
                                    elif v < 10000:
                                        downs_list.append(f'Lượt xem thấp ({v:,} lượt)')
                                
                                # likeCount
                                if 'likeCount' in raw:
                                    l = raw['likeCount']
                                    if l >= 50000:
                                        ups_list.append(f'Lượt like cao ({l/1000:.0f}K lượt)')
                                    elif l < 100:
                                        downs_list.append(f'Lượt like thấp ({l} lượt)')
                                
                                # likeCount ratio
                                if 'likeCount' in raw and 'viewCount' in raw and raw['viewCount'] > 0:
                                    like_rate = (raw['likeCount'] / raw['viewCount']) * 100
                                    if like_rate >= 5:
                                        ups_list.append(f'Tỷ lệ like tốt ({like_rate:.2f}%)')
                                    elif like_rate < 1:
                                        downs_list.append(f'Tỷ lệ like thấp ({like_rate:.2f}%)')
                                
                                # commentCount
                                if 'commentCount' in raw:
                                    c = raw['commentCount']
                                    if c >= 1000:
                                        ups_list.append(f'Lượt bình luận cao ({c:,} lượt)')
                                    elif c < 10:
                                        downs_list.append(f'Lượt bình luận thấp ({c} lượt)')
                                
                                # commentCount ratio
                                if 'commentCount' in raw and 'viewCount' in raw and raw['viewCount'] > 0:
                                    cmt_rate = (raw['commentCount'] / raw['viewCount']) * 100
                                    if cmt_rate >= 1:
                                        ups_list.append(f'Tỷ lệ bình luận tốt ({cmt_rate:.2f}%)')
                                    elif cmt_rate < 0.1:
                                        downs_list.append(f'Tỷ lệ bình luận thấp ({cmt_rate:.2f}%)')

                                ups_list = ups_list[:3]
                                downs_list = downs_list[:3]
                                parts = []
                                if ups_list:
                                    parts.append('Tăng xác suất: ' + ', '.join(ups_list))
                                if downs_list:
                                    parts.append('Giảm xác suất: ' + ', '.join(downs_list))
                                reasons_text = '; '.join(parts) if parts else 'Dựa trên các chỉ số tương tác và nội dung'
                                
                                result = {
                                    'score': score,
                                    'probability': f"{proba:.1%}",
                                    'reason': reasons_text
                                }
                            else:
                                error = "Không thể xử lý thông tin video"
                    except Exception as e:
                        error = f"Lỗi dự đoán: {str(e)}"
    
    return render(request, 'main/predict.html', {
        'result': result,
        'error': error,
        'video_info': video_info
    })

def data(request):
    """Data view"""
    dashboard_data = get_dashboard_data()
    
    context = {
        'videos': dashboard_data['trending_videos'],
        'categories': dashboard_data['categories'],
        'channels': dashboard_data['top_channels'],
        'trending_count': len(dashboard_data['trending_videos'])
    }
    
    return render(request, 'main/data.html', context)

def channel_list(request):
    """Channel list view"""
    dashboard_data = get_dashboard_data()
    
    context = {
        'channels': dashboard_data['top_channels'],
        'categories': dashboard_data['categories'],
        'trending_channels': len(dashboard_data['top_channels']),
        'total_videos': dashboard_data['total_videos']
    }
    
    return render(request, 'main/channel_list.html', context)

def video_detail(request, video_id):
    """Video detail view"""
    video_info = get_video_info(video_id)
    
    context = {
        'video': video_info
    }
    
    return render(request, 'main/video_detail.html', context)

def account(request):
    """Account view"""
    # Simulated user data
    context = {
        'user': {
            'username': 'user123',
            'email': 'user@example.com'
        },
        'prediction_count': 45,
        'accuracy': 98.6,
        'join_date': '01/01/2024',
        'prediction_history': [
            {
                'time': '2024-01-15 14:30:00',
                'title': 'Sample Video Title',
                'channel': 'Sample Channel',
                'score': 85
            }
        ]
    }
    
    return render(request, 'main/user.html', context)

def dashboard_csv_data(request):
    """Serve aggregated analytics from local CSV for rich charts"""
    csv_path = os.path.join('data', 'youtube_trending_data_processed.csv')
    if not os.path.exists(csv_path):
        # Fallback for older file name
        alt_path = os.path.join('data', 'data_youtube_trending_video.csv')
        csv_path = alt_path
    try:
        df = pd.read_csv(csv_path)
        # Normalize columns
        for col in ['viewCount', 'likeCount', 'commentCount']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0
        if 'categoryId' not in df.columns:
            df['categoryId'] = 'Unknown'
        if 'title' not in df.columns:
            df['title'] = ''
        if 'tags' not in df.columns:
            df['tags'] = ''

        # Basic totals
        totals = {
            'views': int(df['viewCount'].sum()),
            'likes': int(df['likeCount'].sum()),
            'comments': int(df['commentCount'].sum()),
            'videos': int(len(df))
        }

        # View stats
        view_stats = {
            'mean': float(df['viewCount'].mean()) if len(df) else 0.0,
            'median': float(df['viewCount'].median()) if len(df) else 0.0,
            'min': int(df['viewCount'].min()) if len(df) else 0,
            'max': int(df['viewCount'].max()) if len(df) else 0,
            'over1m': int((df['viewCount'] > 1_000_000).sum()),
        }

        # Histogram for viewCount
        hist_bins = [0, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
        hist_labels = ['<1K', '1K-10K', '10K-100K', '100K-1M', '1M-10M', '10M-100M']
        df['view_bin'] = pd.cut(df['viewCount'], bins=hist_bins, labels=hist_labels, include_lowest=True)
        hist_counts = df['view_bin'].value_counts().reindex(hist_labels, fill_value=0)
        histogram = {
            'bins': hist_labels,
            'counts': hist_counts.astype(int).tolist()
        }

        # Correlation view-like
        corr_val = 0.0
        if df['viewCount'].std() > 0 and df['likeCount'].std() > 0:
            corr_val = float(df['viewCount'].corr(df['likeCount']))

        # Scatter sample view vs like
        sample = df.sample(min(1000, len(df)), random_state=42) if len(df) > 0 else df
        scatter_df = sample[['viewCount', 'likeCount']].fillna(0)
        scatter_points = scatter_df.apply(lambda r: [int(r['viewCount']), int(r['likeCount'])], axis=1).tolist()

        # Category distribution with name mapping
        CATEGORY_NAMES = {
            '1': 'Phim & Hoạt hình',
            '2': 'Xe cộ',
            '10': 'Âm nhạc',
            '15': 'Thú cưng & Động vật',
            '17': 'Thể thao',
            '19': 'Du lịch & Sự kiện',
            '20': 'Trò chơi',
            '22': 'Con người & Blog',
            '23': 'Hài kịch',
            '24': 'Giải trí',
            '25': 'Tin tức & Chính trị',
            '26': 'Hướng dẫn & Phong cách',
            '27': 'Giáo dục',
            '28': 'Khoa học & Công nghệ',
            '29': 'Phi lợi nhuận & Hoạt động',
            '30': 'Phim',
            '31': 'Anime/Animation',
            '32': 'Hành động/Phiêu lưu',
            '33': 'Cổ điển',
            '34': 'Hài kịch',
            '35': 'Tài liệu',
            '36': 'Drama',
            '37': 'Gia đình',
            '38': 'Nước ngoài',
            '39': 'Kinh dị',
            '40': 'Khoa học viễn tưởng',
            '41': 'Kịch tính',
            '42': 'Shorts',
            '43': 'Chương trình',
            '44': 'Trailer',
        }
        cat_counts = df['categoryId'].astype(str).value_counts().head(12)
        categories = []
        for cat_id, count in cat_counts.items():
            cat_name = CATEGORY_NAMES.get(cat_id, f'Danh mục {cat_id}')
            categories.append({
                'id': cat_id,
                'name': cat_name,
                'value': int(count)
            })

        # PublishedAt analysis - time of day and day of week
        published_time_data = {'hours': {}, 'weekdays': {}}
        if 'publishedAt' in df.columns:
            try:
                df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
                df_pub = df.dropna(subset=['publishedAt'])
                if len(df_pub) > 0:
                    # Extract hour and weekday
                    df_pub['hour'] = df_pub['publishedAt'].dt.hour
                    df_pub['weekday'] = df_pub['publishedAt'].dt.dayofweek
                    weekday_names = ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'Chủ nhật']
                    
                    # Average views by hour
                    hour_views = df_pub.groupby('hour')['viewCount'].mean().to_dict()
                    published_time_data['hours'] = {str(h): float(v) for h, v in sorted(hour_views.items())}
                    
                    # Average views by weekday
                    weekday_views = df_pub.groupby('weekday')['viewCount'].mean().to_dict()
                    published_time_data['weekdays'] = {weekday_names[w]: float(weekday_views.get(w, 0)) for w in range(7)}
            except Exception as e:
                print(f"Error processing publishedAt: {e}")

        # Duration analysis
        duration_data = {'distribution': [], 'vs_views': []}
        if 'duration' in df.columns:
            try:
                import re
                def parse_duration(dur_str):
                    """Parse ISO 8601 duration (PT4M13S) to seconds"""
                    if pd.isna(dur_str) or not dur_str:
                        return None
                    dur_str = str(dur_str).upper()
                    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', dur_str)
                    if not match:
                        return None
                    hours = int(match.group(1) or 0)
                    minutes = int(match.group(2) or 0)
                    seconds = int(match.group(3) or 0)
                    return hours * 3600 + minutes * 60 + seconds
                
                df['duration_sec'] = df['duration'].apply(parse_duration)
                df_dur = df.dropna(subset=['duration_sec', 'viewCount'])
                
                if len(df_dur) > 0:
                    # Duration distribution bins
                    dur_bins = [0, 60, 180, 300, 600, 1800, 3600, float('inf')]
                    dur_labels = ['<1 phút', '1-3 phút', '3-5 phút', '5-10 phút', '10-30 phút', '30-60 phút', '>60 phút']
                    df_dur['dur_bin'] = pd.cut(df_dur['duration_sec'], bins=dur_bins, labels=dur_labels, include_lowest=True)
                    dur_dist = df_dur['dur_bin'].value_counts().reindex(dur_labels, fill_value=0)
                    duration_data['distribution'] = [{'name': label, 'value': int(count)} for label, count in dur_dist.items()]
                    
                    # Duration vs Views scatter (sample)
                    sample_dur = df_dur.sample(min(1000, len(df_dur)), random_state=42)
                    duration_data['vs_views'] = sample_dur[['duration_sec', 'viewCount']].apply(
                        lambda r: [int(r['duration_sec']), int(r['viewCount'])], axis=1
                    ).tolist()
            except Exception as e:
                print(f"Error processing duration: {e}")

        # Top 10 videos by views - from local dataset only
        top_videos = []
        try:
            df_top = df.sort_values('viewCount', ascending=False).head(10)
            for _, row in df_top.iterrows():
                top_videos.append({
                    'title': str(row.get('title', 'Unknown Title'))[:120],
                    'channel': str(row.get('channelTitle', row.get('channel', 'Unknown Channel')))[:60],
                    'viewCount': int(row.get('viewCount', 0)),
                    'likeCount': int(row.get('likeCount', 0)),
                    'commentCount': int(row.get('commentCount', 0)),
                })
        except Exception as e:
            print(f"Error computing top videos from dataset: {e}")
            top_videos = []

        # Word cloud from whole dataset (tags + title)
        word_cloud_data = []
        try:
            import re
            from collections import Counter
            df_keywords = df.copy()

            # Extract words from tags and titles
            all_words = []
            for _, row in df_keywords.iterrows():
                # Tags
                tags_str = str(row.get('tags', ''))
                if tags_str and tags_str != 'nan':
                    tags = [t.strip().lower() for t in tags_str.split(',') if t.strip()]
                    all_words.extend(tags)
                
                # Title words (remove common words)
                title = str(row.get('title', ''))
                if title and title != 'nan':
                    words = re.findall(r'\b[a-z]{3,}\b', title.lower())
                    all_words.extend(words)
            
            # Count and filter
            word_counts = Counter(all_words)
            # Remove very common words
            stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'she', 'use', 'her', 'many', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'much', 'over', 'such', 'take', 'than', 'them', 'well', 'were', 'what', 'with', 'your', 'this', 'that', 'from', 'have', 'been', 'more', 'will', 'about', 'into', 'their', 'there', 'these', 'would', 'other', 'which', 'could', 'should', 'after', 'before', 'during', 'while', 'where', 'every', 'first', 'second', 'third', 'video', 'videos', 'watch', 'youtube', 'channel', 'subscribe', 'click', 'link'}
            filtered_words = {word: count for word, count in word_counts.items() if word not in stop_words and len(word) > 2}
            word_cloud_data = [{'name': word, 'value': count} for word, count in sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:100]]
        except Exception as e:
            print(f"Error generating word cloud: {e}")
            word_cloud_data = []

        return JsonResponse({
            'totals': totals,
            'viewStats': view_stats,
            'histogram': histogram,
            'correlation': corr_val,
            'scatter': scatter_points,
            'categories': categories,
            'publishedTime': published_time_data,
            'duration': duration_data,
            'topVideos': top_videos,
            'wordCloud': word_cloud_data,
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
