import os
import torch
import torchaudio
import math
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import soundfile as sf

# Configuration
WHISPER_MODEL_NAME = "openai/whisper-large-v3"
TARGET_SR = 16000

def load_whisper_model_and_processor(model_name=WHISPER_MODEL_NAME):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, Whisper transcription will be slow on CPU.")
    
    print(f"Loading Whisper model '{model_name}' onto {device}...")
    try:
        transcriber = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            torch_dtype=torch.float16 if device == "cuda:0" else torch.float32, 
        )
       
        transcriber.model.config.forced_decoder_ids = None 
        print("Whisper model and processor loaded using pipeline.")
        return transcriber
    except Exception as e:
        print(f"Error loading Whisper model with pipeline: {e}")
        print("Trying to load model and processor manually for more control/debugging...")
        try:
            processor = WhisperProcessor.from_pretrained(model_name)
            model = WhisperForConditionalGeneration.from_pretrained(model_name)
            if device == "cuda:0":
                model = model.half() # for float16 on GPU
            model.to(device)
            model.eval()
            print("Whisper model and processor loaded manually.")
            return {"processor": processor, "model": model, "device": device, "type": "manual"}
        except Exception as e_manual:
            print(f"Error loading Whisper model manually: {e_manual}")
            return None


def load_audio_for_whisper(file_path, target_sr=TARGET_SR):
    """Loads audio, resamples to target_sr, and ensures it's mono."""
    try:
        waveform, sample_rate = sf.read(file_path, dtype='float32')
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        
        if sample_rate != target_sr:
            waveform_tensor = torch.from_numpy(waveform).float()
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform_resampled_tensor = resampler(waveform_tensor)
            return waveform_resampled_tensor.numpy()
        return waveform 
    except Exception as e:
        print(f"    Error loading audio {file_path}: {e}")
        try:
            print("Attempting torchaudio.load as fallback...")
            waveform_t, sample_rate_t = torchaudio.load(file_path)
            if waveform_t.shape[0] > 1: # If stereo
                waveform_t = waveform_t.mean(dim=0, keepdim=False)
            if sample_rate_t != target_sr:
                resampler_t = torchaudio.transforms.Resample(orig_freq=sample_rate_t, new_freq=target_sr)
                waveform_t = resampler_t(waveform_t)
            return waveform_t.numpy()
        except Exception as e_torch:
            print(f"Torchaudio load also failed for {file_path}: {e_torch}")
            return None


def transcribe_audio_whisper(audio_path, transcriber_obj):
    """
    Transcribes an audio file using the loaded Whisper model (pipeline or manual).
    The Hugging Face pipeline with `chunk_length_s` handles long audio well.
    """
    print(f"Loading audio: {audio_path}")
    audio_input = load_audio_for_whisper(audio_path)
    if audio_input is None:
        return "Error: Could not load audio."

    print(f"    Transcribing with Whisper ({WHISPER_MODEL_NAME})... (this may take a while for long audio)")
    
    if isinstance(transcriber_obj, dict) and transcriber_obj.get("type") == "manual":
        processor = transcriber_obj["processor"]
        model = transcriber_obj["model"]
        device = transcriber_obj["device"]
        
        inputs = processor(audio_input, sampling_rate=TARGET_SR, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        if device == "cuda:0" and model.dtype == torch.float16:
            input_features = input_features.half()

        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        transcription_list = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        transcription = " ".join(transcription_list)

    else: 
        output = transcriber_obj(
            audio_input, 
            chunk_length_s=30,
            batch_size=8,   
            return_timestamps=False 
        )
        transcription = output["text"]

    return transcription.strip()


def batch_transcribe_whisper(audio_dir="./processed_audio", output_dir="./transcripts_whisper"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting Whisper transcription from '{audio_dir}' to '{output_dir}'...")

    whisper_components = load_whisper_model_and_processor()
    if whisper_components is None:
        print("Could not load Whisper model. Aborting transcription.")
        return

    processed_count = 0
    error_count = 0
    
    if not os.listdir(audio_dir):
        print(f"No files found in '{audio_dir}'. Please run the audio preprocessing script first.")
        return

    for file in os.listdir(audio_dir):
        if file.lower().endswith(".wav"):
            audio_path = os.path.join(audio_dir, file)
            print(f"\Transcribing with Whisper: {file}")
            
            try:
                transcription = transcribe_audio_whisper(audio_path, whisper_components)
                
                output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_whisper.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(transcription + "\n")
                print(f"Whisper transcription saved to {output_file}")
                processed_count += 1
            except Exception as e:
                print(f"Error during Whisper transcription for {file}: {e}")
                error_count += 1
        else:
            print(f" Skipping non-WAV file: {file}")
            
    print(f"\nWhisper transcription finished. Processed: {processed_count}, Errors: {error_count}")

if __name__ == "__main__":
    batch_transcribe_whisper()