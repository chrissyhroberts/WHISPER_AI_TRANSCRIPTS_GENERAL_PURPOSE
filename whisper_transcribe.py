import torch
import whisper
import sys
import os
import time
import ffmpeg
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

if len(sys.argv) < 2:
    print("Usage: python whisper_transcribe.py <audio_file>")
    sys.exit(1)

file_path = sys.argv[1]

if not os.path.isfile(file_path):
    print(f"Error: File '{file_path}' not found.")
    sys.exit(1)

# ✅ Force CPU execution
device = "cpu"
print(f"Using device: {device} (MPS not fully supported yet)")

# Load Whisper model on CPU using the tiny model
model = whisper.load_model("tiny").to(device)

def get_audio_duration(file_path):
    """Returns the duration of the audio file in seconds using FFmpeg."""
    try:
        probe = ffmpeg.probe(file_path)
        return float(probe['format']['duration'])
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return None

def transcribe_segment(segment):
    """Transcribes a segment of audio (returns timestamped result)."""
    return segment["start"], segment["text"]

def transcribe_audio(file_path):
    """Multithreaded transcription of an audio file using Whisper."""
    print(f"🎤 Processing: {file_path}")

    audio_duration = get_audio_duration(file_path)
    if audio_duration is None:
        print("⚠️ Could not determine audio duration. Progress will be approximate.")

    start_time = time.time()
    results = []

    # ✅ Run Whisper transcription
    with tqdm(total=audio_duration, unit=" sec", dynamic_ncols=True) as pbar:
        result = model.transcribe(file_path, verbose=False, fp16=False)

        # ✅ Use a process pool to manage parallel execution
        with ProcessPoolExecutor(max_workers=4) as executor:  # Use 4 workers (adjust as needed)
            future_to_segment = {executor.submit(transcribe_segment, seg): seg for seg in result["segments"]}
            
            for future in future_to_segment:
                start, text = future.result()
                results.append((start, text))
                
                # ✅ Update progress bar
                elapsed_time = time.time() - start_time
                pbar.update(start - pbar.n)
                pbar.set_description("⏳ Transcribing...")
    
    # ✅ Sort results by start time
    results.sort()

    # ✅ Save transcript
    transcript_path = file_path + ".txt"
    with open(transcript_path, "w") as f:
        f.write(" ".join(text for _, text in results))

    print(f"\n✅ Transcription complete! Saved to: {transcript_path}")

if __name__ == "__main__":
    transcribe_audio(file_path)
