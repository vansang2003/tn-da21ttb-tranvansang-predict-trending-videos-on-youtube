# Lấy 50 video đang trending
from googleapiclient.discovery import build
from datetime import datetime
import pandas as pd
import json

API_KEY = 'YOUR_API_KEY'

def get_trending_videos():
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    request = youtube.videos().list(
        part='snippet,statistics,contentDetails,status,topicDetails',
        chart='mostPopular',
        regionCode='VN',
        maxResults=50
    )

    response = request.execute()
    videos_data = []

    for item in response['items']:
        snippet = item.get('snippet', {})
        status = item.get('status', {})
        contentDetails = item.get('contentDetails', {})
        statistics = item.get('statistics', {})
        topicDetails = item.get('topicDetails', {})

        video_data = {
            "videoId": item.get("id"),
            "title": snippet.get("title"),
            "channelTitle": snippet.get("channelTitle"),
            "publishedAt": snippet.get("publishedAt"),
            "description": snippet.get("description"),
            "tags": ','.join(snippet.get("tags", [])),
            "categoryId": snippet.get("categoryId"),
            "liveBroadcastContent": snippet.get("liveBroadcastContent"),
            "privacyStatus": status.get("privacyStatus"),
            "license": status.get("license"),
            "duration": contentDetails.get("duration"),
            "dimension": contentDetails.get("dimension"),
            "definition": contentDetails.get("definition"),
            "caption": contentDetails.get("caption"),
            "viewCount": statistics.get("viewCount"),
            "likeCount": statistics.get("likeCount"),
            "commentCount": statistics.get("commentCount"),
            "topicCategories": ','.join(topicDetails.get("topicCategories", [])),
            "isTrending": True
        }

        videos_data.append(video_data)

    df = pd.DataFrame(videos_data)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'youtube_trending_vn_{timestamp}.csv'
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

    #json_filename = f'youtube_trending_vn_{timestamp}.json'
    #with open(json_filename, 'w', encoding='utf-8') as f:
        #json.dump(videos_data, f, ensure_ascii=False, indent=4)

    print(f"Đã lưu dữ liệu vào các file:")
    print(f"- CSV: {csv_filename}")

    return df

if __name__ == "__main__":
    trending_videos = get_trending_videos()
    print("\nTop 50 video trending trên YouTube Việt Nam:")
    print(trending_videos[['title', 'channelTitle', 'viewCount', 'likeCount', 'commentCount']])
