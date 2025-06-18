# YouTube Video Transcription, Correction, and Summarization Workflow

This document outlines a multi-stage workflow for downloading YouTube videos, extracting and processing their audio, transcribing the audio using advanced ASR models, correcting the transcriptions with a grammar correction model, and finally using a large language model (like Gemini) for post-processing and summarization.

## Table of Contents
1.  [Overview](#overview)
2.  [Prerequisites](#prerequisites)
3.  [Workflow Stages](#workflow-stages)
    *   [3.1. Stage 1: Download YouTube Videos](#stage-1-download-youtube-videos)
    *   [3.2. Stage 2: Extract Audio from Videos](#stage-2-extract-audio-from-videos)
    *   [3.3. Stage 3: Preprocess Audio](#stage-3-preprocess-audio)
    *   [3.4. Stage 4: Transcribe Audio (Choose one or both)](#stage-4-transcribe-audio)
        *   [3.4.1. Option A: Nemo Parakeet Transcription](#option-a-nemo-parakeet-transcription)
        *   [3.4.2. Option B: Hugging Face Whisper Transcription](#option-b-hugging-face-whisper-transcription)
    *   [3.5. Stage 5: Correct Transcriptions with GRMR](#stage-5-correct-transcriptions-with-grmr)
    *   [3.6. Stage 6: Post-Processing and Summarization with Gemini](#stage-6-post-processing-and-summarization-with-gemini)
4.  [Running the Workflow](#running-the-workflow)
5.  [Directory Structure](#directory-structure)
6.  [Script Details](#script-details)

## 1. Overview

The pipeline automates several steps:
*   Fetching videos from YouTube.
*   Isolating the audio track.
*   Standardizing audio to a format suitable for ASR (mono, 16kHz).
*   Generating raw transcriptions using either Nemo Parakeet or OpenAI Whisper.
*   Improving the grammatical quality of the transcriptions.
*   Manual/Assisted final review, formatting, and summarization.

## 2. Prerequisites

Make sure you have Python 3.x installed. You'll also need to install the following libraries and tools:

*   **yt-dlp**: For downloading YouTube videos.
    ```bash
    pip install yt-dlp
    # or via your system's package manager
    ```
*   **Python Libraries**:
    ```bash
    pip install moviepy torchaudio soundfile nemo_toolkit[asr] torch transformers accelerate
    ```
    *   `moviepy`: For video/audio editing.
    *   `torchaudio`: For audio loading, resampling, and manipulation.
    *   `soundfile`: For saving audio files.
    *   `nemo_toolkit[asr]`: For NVIDIA NeMo ASR models.
    *   `torch`: PyTorch.
    *   `transformers`: Hugging Face library.
    *   `accelerate`: For efficient model loading.

*   **Hardware**:
    *   A GPU is highly recommended for the transcription and correction stages.

## 3. Workflow Stages

Each stage is typically represented by a Python script. The output of one stage serves as the input for the next.

### 3.1. Stage 1: Download YouTube Videos

*   **Script**: `01_download_videos.py`
*   **Purpose**: Downloads videos from a list of YouTube URLs.
*   **Key Function**: `download_youtube_video(url, output_path)`
*   **Input**: A list of `VIDEO_URLS` defined within the script.
*   **Output**: Video files (e.g., `.mkv`, `.mp4`) saved in the `downloads/` directory.

### 3.2. Stage 2: Extract Audio from Videos

*   **Script**: `02_extract_audio.py`
*   **Purpose**: Separates the audio track from the downloaded video files.
*   **Key Function**: `extract_audio(video_path, output_audio_path)`
*   **Input**: Video files from the `downloads/` directory.
*   **Output**: Audio files (specifically `.wav`) saved in the `audio/` directory.

### 3.3. Stage 3: Preprocess Audio

*   **Script**: `03_preprocess_audio.py`
*   **Purpose**: Converts audio files to a standard format (16kHz, mono) required by most ASR models.
*   **Key Functions**: `convert_to_mono_16k(input_path, output_path)`, `process_audio_folder(input_dir, output_dir)`
*   **Input**: Audio files from the `audio/` directory.
*   **Output**: Processed audio files (mono, 16kHz `.wav`) saved in the `processed_audio/` directory.

### 3.4. Stage 4: Transcribe Audio

Choose one or both transcription options.

#### 3.4.1. Option A: Nemo Parakeet Transcription

*   **Script**: `04a_transcribe_nemo.py`
*   **Purpose**: Transcribes preprocessed audio using NVIDIA's Parakeet model.
*   **Key Functions**: `load_audio_chunks(...)`, `transcribe_chunks_nemo(audio_path, model)`, `batch_transcribe_nemo(audio_dir, output_dir)`
*   **Input**: Processed audio files from `processed_audio/`.
*   **Output**: Text transcriptions (`.txt`) saved in `transcripts_nemo/`.
*   **Note**: Uses chunking for long audio files. The specific Parakeet model name (`nvidia/parakeet-tdt-0.6b-v2`) might need adjustment based on availability; a fallback is included in the script.

#### 3.4.2. Option B: Hugging Face Whisper Transcription

*   **Script**: `04b_transcribe_whisper.py`
*   **Purpose**: Transcribes preprocessed audio using OpenAI's Whisper model via Hugging Face Transformers.
*   **Key Functions**: `load_model(...)`, `load_audio(...)`, `transcribe_whisper_chunks(audio_path, processor, model, device)`, `batch_transcribe_whisper(audio_dir, output_dir)`
*   **Input**: Processed audio files from `processed_audio/`.
*   **Output**: Text transcriptions (`.txt`) saved in `transcripts_whisper/`.
*   **Note**: Implements chunking for files longer than Whisper's native 30-second window. `openai/whisper-large-v3` is used by default.

### 3.5. Stage 5: Correct Transcriptions with GRMR

*   **Script**: `05_correct_transcripts.py`
*   **Purpose**: Improves the grammar and readability of the ASR-generated transcripts using the `qingy2024/GRMR-V3-Q1.7B` model.
*   **Key Functions**: `load_correction_model()`, `correct_text(text_to_correct, tokenizer, model)`, `batch_correct_transcripts(input_dir, output_dir)`
*   **Input**: Text transcriptions from `transcripts_nemo/` (default) or `transcripts_whisper/` (configurable within the script via `DEFAULT_INPUT_DIR`).
*   **Output**: Corrected text transcriptions (`.txt`) saved in `transcripts_corrected/`.
*   **Note**: This model is large and requires significant VRAM. Prompting and output parsing are handled within the script.

### 3.6. Stage 6: Post-Processing and Summarization with Gemini

*   **Purpose**: Final review of corrected transcripts, potential further edits, and generation of summaries or other insights.
*   **Tool**: Google Gemini (or another advanced LLM like Claude, GPT-4). This is a manual or API-driven step.
*   **Process**:
    1.  **Review**: Open the `.txt` files from the `transcripts_corrected/` directory.
    2.  **Manual Edits**: Make any necessary manual corrections.
    3.  **Summarization/Analysis with Gemini**:
        *   Copy the corrected text.
        *   Paste into your Gemini interface.
        *   Use prompts tailored to your needs (e.g., summarize, extract topics, identify action items).
    4.  **Save Output**: Copy the results from Gemini and save them.
*   **Input**: Corrected transcript files from `transcripts_corrected/`.
*   **Output**: Manually reviewed and refined transcripts, summaries, analyses, etc., saved by the user.

## 4. Running the Workflow

1.  **Setup**: Ensure all [Prerequisites](#prerequisites) are installed.
2.  **Configure Scripts**:
    *   In `01_download_videos.py`: Update the `VIDEO_URLS` list.
    *   In `05_correct_transcripts.py`: If necessary, adjust `DEFAULT_INPUT_DIR` to match the output directory of your chosen transcription script (e.g., `transcripts_whisper/`).
3.  **Execute Scripts Sequentially**:
    Run the Python scripts in order from your terminal. It is recommended to place all scripts (`01_*.py` to `05_*.py`) in the same directory.
    ```bash
    python 01_download_videos.py
    python 02_extract_audio.py
    python 03_preprocess_audio.py #This is optional and can be skipped now, added post_processing with models only.

    # Choose one or both transcription scripts:
    # python 04a_transcribe_nemo.py
    # OR/AND
    # python 04b_transcribe_whisper.py

    python 05_correct_transcripts.py

    # After script execution, proceed to Stage 6 (Gemini) manually.
    ```
    The first time running scripts that download large models (Nemo, Whisper, GRMR) will take longer.

## 5. Directory Structure

The workflow will create the following directory structure (or similar) in the location where the scripts are run: