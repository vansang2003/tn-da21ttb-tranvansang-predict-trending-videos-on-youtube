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
PIPELINE_PATH = 'xgb_pipeline.pkl'
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
    """Extract features from video info for ML model"""
    if not info:
        return None
    
    # Text preprocessing
    title_len = len(info.get('title', ''))
    desc_len = len(info.get('description', ''))
    desc_words = len(info.get('description', '').split())
    tag_count = len(info.get('tags', '').split(',')) if info.get('tags') else 0
    
    # Numeric features
    view_count = info.get('viewCount', 0)
    like_count = info.get('likeCount', 0)
    comment_count = info.get('commentCount', 0)
    category_id = int(info.get('categoryId', 0))
    
    return {
        'title_len': title_len,
        'desc_len': desc_len,
        'desc_words': desc_words,
        'tag_count': tag_count,
        'categoryId': category_id,
        'viewCount': view_count,
        'likeCount': like_count,
        'commentCount': comment_count
    }

def load_model_bundle():
    """Load trained model, scaler and selected feature names."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(PIPELINE_PATH) and os.path.exists(FEATURES_PATH)):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(PIPELINE_PATH)
    with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
        selected_features = [line.strip() for line in f if line.strip()]
    return model, scaler, selected_features

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
                        model, scaler, selected_features = load_model_bundle()
                        if not (model and scaler and selected_features):
                            error = "Model chưa được huấn luyện đầy đủ (model/scaler/features). Vui lòng chạy train_xgboost.py trước"
                        else:
                            # Extract features
                            raw = extract_features(video_info)
                            if raw:
                                # Sắp xếp features đúng thứ tự đã dùng khi train
                                vector = [raw.get(name, 0) for name in selected_features]
                                arr = np.array(vector, dtype=float).reshape(1, -1)
                                # Scale & predict
                                arr_scaled = scaler.transform(arr)
                                proba = float(model.predict_proba(arr_scaled)[0][1]) if hasattr(model, 'predict_proba') else float(model.predict(arr_scaled)[0])
                                pred = int(proba >= 0.5)
                                score = int(proba * 100)

                                
                                ups_list = []
                                downs_list = []
                                # viewCount
                                if 'viewCount' in raw:
                                    v = raw['viewCount']
                                    if v >= 100000:
                                        ups_list.append('Lượt xem cao (>=100K)')
                                    elif v < 10000:
                                        downs_list.append('Lượt xem thấp (<10K)')
                                # likeCount ratio
                                if 'likeCount' in raw and 'viewCount' in raw:
                                    like_rate = (raw['likeCount'] / max(raw['viewCount'], 1)) * 100
                                    if like_rate >= 5:
                                        ups_list.append(f'Tỷ lệ like tốt (~{like_rate:.1f}%)')
                                    elif like_rate < 1:
                                        downs_list.append(f'Tỷ lệ like thấp (~{like_rate:.1f}%)')
                                # commentCount ratio
                                if 'commentCount' in raw and 'viewCount' in raw:
                                    cmt_rate = (raw['commentCount'] / max(raw['viewCount'], 1)) * 100
                                    if cmt_rate >= 1:
                                        ups_list.append(f'Tỷ lệ bình luận tốt (~{cmt_rate:.2f}%)')
                                    elif cmt_rate < 0.1:
                                        downs_list.append(f'Tỷ lệ bình luận thấp (~{cmt_rate:.2f}%)')
                                # tag_count
                                if 'tag_count' in raw:
                                    t = raw['tag_count']
                                    if t >= 10:
                                        ups_list.append('Nhiều thẻ tag (>=10)')
                                    elif t <= 2:
                                        downs_list.append('Ít thẻ tag (<=2)')
                                # title_len
                                if 'title_len' in raw:
                                    tl = raw['title_len']
                                    if 20 <= tl <= 70:
                                        ups_list.append('Tiêu đề độ dài hợp lý (20-70)')
                                    elif tl < 15:
                                        downs_list.append('Tiêu đề quá ngắn (<15)')
                                    elif tl > 100:
                                        downs_list.append('Tiêu đề quá dài (>100)')
                                # desc_words
                                if 'desc_words' in raw:
                                    dw = raw['desc_words']
                                    if dw >= 50:
                                        ups_list.append('Mô tả chi tiết (>=50 từ)')
                                    elif dw < 10:
                                        downs_list.append('Mô tả sơ sài (<10 từ)')

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
    csv_path = os.path.join('data', 'data_youtube_trending_video.csv')
    if not os.path.exists(csv_path):
        # Fallback to path used in training script
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

        # Top categories by count
        cat_counts = df['categoryId'].astype(str).value_counts().head(12)
        categories = [{'name': k, 'value': int(v)} for k, v in cat_counts.items()]

        # Total metrics
        totals = {
            'views': int(df['viewCount'].sum()),
            'likes': int(df['likeCount'].sum()),
            'comments': int(df['commentCount'].sum()),
            'videos': int(len(df))
        }

        # Like/Comment vs View scatter sample (limit for performance)
        sample = df.sample(min(1000, len(df)), random_state=42) if len(df) > 0 else df
        scatter = sample[['viewCount', 'likeCount', 'commentCount']].fillna(0).astype(int)
        scatter_points = scatter.apply(lambda r: [int(r['viewCount']), int(r['likeCount']), int(r['commentCount'])], axis=1).tolist()

        # Top videos by views
        top_videos_df = df.sort_values('viewCount', ascending=False).head(20)
        top_videos = {
            'titles': top_videos_df['title'].astype(str).tolist(),
            'views': top_videos_df['viewCount'].astype(int).tolist(),
            'likes': top_videos_df['likeCount'].astype(int).tolist(),
            'comments': top_videos_df['commentCount'].astype(int).tolist()
        }

        # Word cloud data from tags
        def tag_split(s):
            return [t.strip() for t in str(s).split(',') if t.strip()]
        all_tags = []
        df['tags'].apply(lambda s: all_tags.extend(tag_split(s)))
        tag_series = pd.Series(all_tags)
        tag_counts = tag_series.value_counts().head(100) if not tag_series.empty else pd.Series(dtype=int)
        word_cloud = [{'name': k, 'value': int(v)} for k, v in tag_counts.items()]

        # Engagement rate by bins of views
        bins = [0, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
        labels = ['<1K', '1K-10K', '10K-100K', '100K-1M', '1M-10M']
        df['view_bin'] = pd.cut(df['viewCount'], bins=bins, labels=labels, include_lowest=True)
        grp = df.groupby('view_bin').agg({'likeCount':'mean','commentCount':'mean','viewCount':'count'}).reindex(labels)
        engagement_by_bin = {
            'bins': labels,
            'avgLikes': [float(x) if pd.notna(x) else 0.0 for x in grp['likeCount'].tolist()],
            'avgComments': [float(x) if pd.notna(x) else 0.0 for x in grp['commentCount'].tolist()],
            'counts': [int(x) if pd.notna(x) else 0 for x in grp['viewCount'].tolist()]
        }

        return JsonResponse({
            'totals': totals,
            'categories': categories,
            'scatter': scatter_points,
            'topVideos': top_videos,
            'wordCloud': word_cloud,
            'engagementByBin': engagement_by_bin
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
