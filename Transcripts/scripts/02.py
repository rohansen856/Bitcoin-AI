from moviepy.editor import VideoFileClip
import os

def extract_audio(video_path, output_audio_path):
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return False

    try:
        print(f"\n Extracting audio from: {video_path}")
        video_clip = VideoFileClip(video_path)

        if video_clip.audio is None:
            print(f" No audio track found in {video_path}")
            video_clip.close()
            return False

        video_clip.audio.write_audiofile(output_audio_path, codec='pcm_s16le')
        video_clip.close() 
        print(f" Audio extracted: {output_audio_path}")
        return True

    except Exception as e:
        print(f"  Error extracting audio from {video_path}: {e}")
        if 'video_clip' in locals() and video_clip:
            video_clip.close()
        return False

if __name__ == "__main__":
    input_dir = "./downloads"
    output_dir = "./audio"
    os.makedirs(output_dir, exist_ok=True)

    print("Starting audio extraction process...")
    processed_files = 0
    skipped_files = 0

    if not os.listdir(input_dir):
        print(f"No files found in '{input_dir}'. Please run the download script first.")
    else:
        for filename in os.listdir(input_dir):
            # Process common video formats, including .mkv and .mp4
            if filename.lower().endswith((".mkv", ".mp4", ".webm", ".avi", ".mov", ".flv")):
                video_path = os.path.join(input_dir, filename)
                base_name = os.path.splitext(filename)[0]
                audio_path = os.path.join(output_dir, f"{base_name}.wav") 

                if extract_audio(video_path, audio_path):
                    processed_files += 1
                else:
                    skipped_files +=1
            else:
                print(f"Skipping non-video file: {filename}")
                skipped_files += 1
    
    print(f"\nAudio extraction finished. Processed: {processed_files}, Skipped/Errors: {skipped_files}")