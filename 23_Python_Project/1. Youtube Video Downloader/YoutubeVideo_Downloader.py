from yt_dlp import YoutubeDL
import static_ffmpeg
static_ffmpeg.add_paths()

link = input('Enter your youtube video link: ').strip()

download_option = input('Do you want to download this video? (yes/no): ').strip().lower()

if download_option == "yes":
    ydl_opts = {
        'format': 'bestvideo*+bestaudio/best',
        'merge_output_format': 'mp4',
    }
    with YoutubeDL(ydl_opts) as ydl:
        print("Downloading...")
        ydl.download([link])
        print("Download completed.")
else:
    print("Download aborted.")