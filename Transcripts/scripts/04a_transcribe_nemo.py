import os
import torchaudio
import nemo.collections.asr as nemo_asr
import torch

CHUNK_DURATION = 30   # seconds
CHUNK_OVERLAP = 3     # seconds
TARGET_SR = 16000
TMP_DIR = "tmp_chunks"  # Temporary chunk directory

def load_audio_chunks(file_path, chunk_duration=CHUNK_DURATION, overlap=CHUNK_OVERLAP, target_sr=TARGET_SR):
    os.makedirs(TMP_DIR, exist_ok=True)
    waveform, sample_rate = torchaudio.load(file_path)

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)

    waveform = waveform.mean(dim=0)
    total_samples = waveform.shape[0]
    chunk_size = chunk_duration * target_sr
    overlap_size = overlap * target_sr
    step_size = chunk_size - overlap_size

    chunks = []
    for start in range(0, total_samples, step_size):
        end = min(start + chunk_size, total_samples)
        chunk_waveform = waveform[start:end]
        chunk_path = os.path.abspath(os.path.join(TMP_DIR, f"chunk_{start}_{end}.wav"))
        torchaudio.save(chunk_path, chunk_waveform.unsqueeze(0), target_sr)
        chunks.append(chunk_path)

        if end == total_samples:
            break
    return chunks

def transcribe_chunks_nemo(audio_path, model):
    chunk_paths = load_audio_chunks(audio_path)
    full_transcription = ""

    for chunk_path in chunk_paths:
        try:
            output = model.transcribe([chunk_path])
            full_transcription += output[0].text.strip() + " "
        except Exception as e:
            print(f"Error transcribing {chunk_path}: {e}")
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    return full_transcription.strip()

def batch_transcribe(audio_dir="./processed_audio", output_dir="./transcripts_nemo"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    if not torch.cuda.is_available():
        print("CUDA not available. Running on CPU may be very slow.")

    print(f"Loading Nemo model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2",
        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    processed = 0
    failed = 0

    for file in os.listdir(audio_dir):
        if file.lower().endswith(".wav"):
            audio_path = os.path.join(audio_dir, file)
            print(f"\n Transcribing: {file}")
            try:
                transcription = transcribe_chunks_nemo(audio_path, asr_model)
                output_file = os.path.join(output_dir, f"transcript_{os.path.splitext(file)[0]}_nemo.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(transcription + "\n")
                print(f"Saved transcription to {output_file}")
                processed += 1
            except Exception as e:
                print(f"Failed to transcribe {file}: {e}")
                failed += 1
        else:
            print(f"Skipping non-wav file: {file}")

    print(f"\n Transcription complete. Processed: {processed}, Failed: {failed}")

    # Clean up temp directory
    for f in os.listdir(TMP_DIR):
        try:
            os.remove(os.path.join(TMP_DIR, f))
        except:
            pass

if __name__ == "__main__":
    batch_transcribe()
