import os
import torchaudio
import soundfile as sf

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1 

def convert_to_mono_16k(input_path, output_path):
    try:
        print(f"\n Processing audio file: {input_path}")
        waveform, sample_rate = torchaudio.load(input_path)

        if waveform.shape[0] > TARGET_CHANNELS:
            print(f"    Converting to mono (original channels: {waveform.shape[0]})")
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != TARGET_SAMPLE_RATE:
            print(f"Resampling from {sample_rate}Hz to {TARGET_SAMPLE_RATE}Hz")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
            sample_rate = TARGET_SAMPLE_RATE 

        if waveform.ndim == 1:
            waveform_to_save = waveform.unsqueeze(1).numpy() 
        elif waveform.ndim == 2 and waveform.shape[0] == 1: 
             waveform_to_save = waveform.squeeze(0).unsqueeze(1).numpy()
        else:
            waveform_to_save = waveform.t().numpy() 

        sf.write(output_path, waveform_to_save, samplerate=TARGET_SAMPLE_RATE, subtype='PCM_16')
        print(f"Converted and saved: {output_path}")
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def process_audio_folder(input_dir="./audio", output_dir="./processed_audio"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting audio preprocessing from '{input_dir}' to '{output_dir}'...")
    
    processed_count = 0
    error_count = 0

    if not os.listdir(input_dir):
        print(f"No files found in '{input_dir}'. Please run the audio extraction script first.")
        return

    for file in os.listdir(input_dir):
        if file.lower().endswith(".wav") or file.lower().endswith(".mp3"):
            input_path = os.path.join(input_dir, file)
            output_filename = os.path.splitext(file)[0] + ".wav"
            output_path = os.path.join(output_dir, output_filename)
            
            if convert_to_mono_16k(input_path, output_path):
                processed_count +=1
            else:
                error_count += 1
        else:
            print(f"Skipping non-audio file: {file}")

    print(f"\nAudio preprocessing finished. Processed: {processed_count}, Errors: {error_count}")

if __name__ == "__main__":
    process_audio_folder()