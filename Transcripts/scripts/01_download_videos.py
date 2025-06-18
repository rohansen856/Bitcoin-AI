import subprocess
import os

VIDEO_URLS = [
    "https://www.youtube.com/watch?v=LlN_9H_4l9c"
]

def download_youtube_video(url, output_path="./downloads"):
    try:
        os.makedirs(output_path, exist_ok=True)
        print(f"\n Downloading video from: {url}")

        
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "-o", os.path.join(output_path, "%(title)s.%(ext)s"),
            url
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Download complete for {url}")
    
    except subprocess.CalledProcessError as e:
        print(f"Download failed for {url}: {e}")
    except Exception as e:
        print(f"Unexpected error for {url}: {e}")

if __name__ == "__main__":
    download_output_dir = "downloads"
    os.makedirs(download_output_dir, exist_ok=True)
    
    for video_url in VIDEO_URLS:
        download_youtube_video(video_url, output_path=download_output_dir)
    print("\nAll video download attempts finished.")