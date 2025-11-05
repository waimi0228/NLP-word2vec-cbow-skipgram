from googleapiclient.discovery import build
import pandas as pd

# YouTube Data API 金鑰
API_KEY = 'AIzaSyB9YVkw0f639THjGPheUNmmXAjljP18-XX'

# 頻道 ID
CHANNEL_ID = 'UCcHVKeT_5Ta-gTa-sgooQxQ'

# 建立 YouTube API 客戶端
youtube = build('youtube', 'v3', developerKey=API_KEY)

# 取得該頻道「上傳影片」的播放清單 ID
def get_uploads_playlist_id(channel_id):
    res = youtube.channels().list(
        part='contentDetails',
        id=channel_id
    ).execute()
    uploads_playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    return uploads_playlist_id

# 取得播放清單中的影片 ID 和標題
def get_videos_from_playlist(playlist_id, max_results=50):
    video_data = []
    next_page_token = None

    while len(video_data) < max_results:
        res = youtube.playlistItems().list(
            part='snippet',
            playlistId=playlist_id,
            maxResults=min(50, max_results - len(video_data)),
            pageToken=next_page_token
        ).execute()

        for item in res['items']:
            video_data.append({
                'video_id': item['snippet']['resourceId']['videoId'],
                'title': item['snippet']['title']
            })

        next_page_token = res.get('nextPageToken')
        if not next_page_token:
            break

    return video_data

# 抓影片統計資料（觀看、按讚、留言數）
def get_video_details(video_id):
    res = youtube.videos().list(
        part='statistics,snippet',
        id=video_id
    ).execute()

    item = res['items'][0]

    return {
        'video_id': video_id,
        'title': item['snippet']['title'],
        'like_count': item['statistics'].get('likeCount', 0),
        'comment_count': item['statistics'].get('commentCount', 0),
        'view_count': item['statistics'].get('viewCount', 0)
    }

# 抓取留言（預設最多 50 則）
def get_comments(video_id, max_comments=50):
    comments = []
    try:
        res = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=max_comments,
            textFormat='plainText'
        ).execute()

        for item in res['items']:
            comments.append(
                item['snippet']['topLevelComment']['snippet']['textDisplay']
            )
    except:
        pass
    return comments

# 主程式
PLAYLIST_ID = get_uploads_playlist_id(CHANNEL_ID)
videos = get_videos_from_playlist(PLAYLIST_ID, max_results=100)

data = []

for v in videos:
    detail = get_video_details(v['video_id'])
    comment_list = get_comments(v['video_id'], max_comments=50)

    row = {
        'video_id': detail['video_id'],
        'title': detail['title'],
        'like_count': detail['like_count'],
        'comment_count': detail['comment_count'],
        'view_count': detail['view_count']
    }

    for i, comment in enumerate(comment_list):
        row[f'comment{i+1}'] = comment

    data.append(row)

# 輸出為 Excel
df = pd.DataFrame(data)
df.to_excel("youtube_data.xlsx", index=False)
print("已成功儲存 youtube_data.xlsx")
