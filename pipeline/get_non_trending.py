# Lấy video không trending trong vòng 7 ngày qua với các từ khóa phổ biến
import pandas as pd
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import time

API_KEY = "YOUR_API_KEY"
YOUTUBE = build("youtube", "v3", developerKey=API_KEY)

# Danh sách từ khóa phổ biến tại Việt Nam
KEYWORDS = [
    "nhạc trẻ", "nhạc remix", "nhạc vàng", "hài", "review", "game",
    "drama", "phim", "ẩm thực", "du lịch", "vlog", "show truyền hình",
    "bóng đá", "trending", "tiktok", "reaction", "học tập", "giải trí",
    "thử thách"
]

# Hàm tìm kiếm video theo từ khóa, lọc trong 7 ngày gần nhất
def search_videos_by_keyword(keyword, max_pages=5, max_results_per_page=50):
    videos = []
    page_token = None
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    published_after = seven_days_ago.isoformat("T") + "Z"

    for _ in range(max_pages):
        try:
            search_response = YOUTUBE.search().list(
                part="id",
                q=keyword,
                type="video",
                regionCode="VN",
                order="viewCount",
                publishedAfter=published_after,
                maxResults=max_results_per_page,
                pageToken=page_token
            ).execute()

            video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]
            if not video_ids:
                break

            details_response = YOUTUBE.videos().list(
                part="snippet,statistics,contentDetails,topicDetails,status",
                id=",".join(video_ids)
            ).execute()

            videos.extend(details_response.get("items", []))
            page_token = search_response.get("nextPageToken")
            if not page_token:
                break
            time.sleep(1)
        except Exception as e:
            print(f"Lỗi khi tìm kiếm với từ khóa '{keyword}': {e}")
            break

    return videos

# Hàm lưu video ra CSV
def save_videos_to_csv(video_items, filename="youtube_vn_videos.csv"):
    data = []
    for item in video_items:
        snippet = item["snippet"]
        statistics = item.get("statistics", {})
        contentDetails = item.get("contentDetails", {})
        topicDetails = item.get("topicDetails", {})
        status = item.get("status", {})

        data.append({
            "videoId": item["id"],
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
            "isTrending": False
        })

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"✅ Đã lưu {len(df)} video vào file: {filename}")


if __name__ == "__main__":
    all_videos = {}

    # Tìm kiếm video theo từ khóa trong 7 ngày qua
    for keyword in KEYWORDS:
        print(f"🔍 Đang tìm video với từ khóa: {keyword}")
        search_results = search_videos_by_keyword(keyword, max_pages=5)
        for video in search_results:
            video_id = video["id"]
            if video_id not in all_videos:
                all_videos[video_id] = video
        time.sleep(1)

    # Lưu dữ liệu ra CSV
    save_videos_to_csv(list(all_videos.values()))
