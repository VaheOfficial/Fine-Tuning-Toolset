import os
import whisper
import yt_dlp
from pytube import YouTube  # Commented out as per your request
from pyannote.audio import Pipeline
import pandas as pd
import torch
import time
import shutil
import json
import subprocess
import logging
import torchaudio
import numpy as np
from scipy.io.wavfile import write
import faceRecognition as faceRecognition

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Print if CUDA is available
logging.info(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logging.info(f"CUDA Device Name: {torch.cuda.get_device_name()}")

# Start run timer
startTime = time.time()

# Time calculator function
def time_calculator(portion, start):
    end = time.time()
    length = end - start
    hours = length // 3600
    remaining_seconds = length % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60
    logging.info(f"{portion} took: {hours} hours {minutes} minutes {seconds} seconds.")

# 1. Download YouTube Videos using pytube (commented out)
def download_videos(video_links, output_dir='videos'):
    download_time = time.time()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    audio_files = []
    video_files = []
    
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'noplaylist': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for link in video_links:
            try:
                info_dict = ydl.extract_info(link, download=True)
                video_title = info_dict.get('title', None)
                video_ext = info_dict.get('ext', None)
                video_out_file = os.path.join(output_dir, f"{video_title}.{video_ext}")
                
                video_files.append(video_out_file)
                
                # Extract audio from video
                audio_out_file = os.path.splitext(video_out_file)[0] + '.wav'
                if os.path.exists(audio_out_file):
                    logging.info(f"Audio file {audio_out_file} already exists. Using the existing file.")
                    audio_files.append(audio_out_file)
                else:
                    command = f"ffmpeg -i \"{video_out_file}\" -ac 1 -ar 16000 \"{audio_out_file}\""
                    subprocess.run(command, shell=True, check=True)
                    audio_files.append(audio_out_file)
            except Exception as e:
                logging.error(f"Error while downloading {link}: {e}")

    time_calculator("Download", download_time)
    return audio_files, video_files


# 3. Enhance Voice
def enhance_voice(audio_file):
    enhance_time = time.time()
    output_file = os.path.splitext(audio_file)[0] + '_enhanced.wav'
    
    if os.path.exists(output_file):
        logging.info(f"Enhanced file {output_file} already exists. Using the existing file.")
    else:
        waveform, sample_rate = torchaudio.load(audio_file)
        waveform = waveform.numpy()

        # Apply a simple equalization: boost high frequencies to make voices more distinct
        equalized_waveform = waveform * np.linspace(1, 2, waveform.shape[1])

        # Save the enhanced audio
        write(output_file, sample_rate, equalized_waveform.T)
        logging.info(f"Enhanced {audio_file} to {output_file}")

    time_calculator("Enhancing", enhance_time)
    return output_file

# 4. Transcribe Audio using Whisper
def transcribe_audio(audio_file):
    transcribe_time = time.time()
    transcription_file = os.path.splitext(audio_file)[0] + '_transcription.json'
    
    if os.path.exists(transcription_file):
        logging.info(f"Transcription file {transcription_file} already exists. Loading the existing transcription.")
        with open(transcription_file, 'r') as f:
            result = json.load(f)
    else:
        model = whisper.load_model("medium").to('cuda' if torch.cuda.is_available() else 'cpu')
        result = model.transcribe(audio_file)
        with open(transcription_file, 'w') as f:
            json.dump(result, f)
        logging.info(f"Transcription saved to {transcription_file}")

    time_calculator("Transcribing", transcribe_time)
    return result['text'], result['segments']

# 5. Diarize Audio using pyannote
def diarize_audio(audio_file, hf_token):
    diarize_time = time.time()
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(device)
    pipeline = pipeline.to(device)
    diarization = pipeline(audio_file)
    time_calculator("Diarization", diarize_time)
    return diarization

# 6. Annotate Speakers
def annotate_speakers(transcription_segments, diarization_result):
    annotate_time = time.time()
    annotations = []
    for segment in transcription_segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        
        speaker = 'Unknown'
        for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
            if (turn.start <= start_time <= turn.end) or (turn.start <= end_time <= turn.end) or (start_time <= turn.start and end_time >= turn.end):
                speaker = speaker_label
                break
        
        annotations.append({
            'start_time': start_time,
            'end_time': end_time,
            'text': text,
            'speaker': speaker
        })
    time_calculator("Annotation", annotate_time)
    return pd.DataFrame(annotations)

# Ensure ffmpeg is available
if shutil.which("ffmpeg") is not None:
    logging.info("ffmpeg is installed")
else:
    raise EnvironmentError("ffmpeg is not installed. Please install it from https://ffmpeg.org/download.html")

# Ensure whisper model is available
try:
    model = whisper.load_model("medium")
except AttributeError as e:
    logging.error("Error loading whisper model:", e)
    raise

# Your Hugging Face token (better to load from environment variables or config)
hf_token = os.getenv("HF_TOKEN")

# Example Usage
video_links = [
    'https://www.youtube.com/watch?v=-1reJEP_gsI',
]
audio_files, video_file = download_videos(video_links)

# Assuming you already have the audio files
# Assuming you already have the audio files
# audio_files = ['C:/Users/VaheOfficial/projects/Data Science/YoutubeScraper/videos/The Walking Dead Episode 1 A New Day Perfect Walkthrough Part 1.wav']
# video_file = 'C:/Users/VaheOfficial/projects/Data Science/YoutubeScraper/videos/The Walking Dead Episode 1 A New Day Perfect Walkthrough Part 1.mp4'

for audio_file in audio_files:
    if not os.path.exists(audio_file):
        logging.warning(f"Audio file {audio_file} not found.")
        continue

    try:
        enhanced_audio_file = enhance_voice(audio_file)
    except Exception as e:
        logging.error(f"Error enhancing {audio_file}: {e}")
        continue

    try:
        transcription, segments = transcribe_audio(enhanced_audio_file)
    except RuntimeError as e:
        logging.error(f"Error transcribing {enhanced_audio_file}: {e}")
        continue

    try:
        diarization_result = diarize_audio(enhanced_audio_file, hf_token)
    except Exception as e:
        logging.error(f"Error in diarization for {enhanced_audio_file}: {e}")
        continue

    # Extract faces and match with audio
    try:
        faces, face_timestamps = faceRecognition.extract_faces(video_file)
        face_embeddings = faceRecognition.recognize_faces(faces)
        speaker_face_map = faceRecognition.match_faces_with_audio(diarization_result, face_timestamps, face_embeddings)
    except Exception as e:
        logging.error(f"Error processing video for face recognition: {e}")
        continue

    annotations = annotate_speakers(segments, diarization_result)
    
    # Save the annotations to a file
    annotations_file = os.path.splitext(enhanced_audio_file)[0] + '_annotations.csv'
    annotations.to_csv(annotations_file, index=False)
    logging.info(f"Annotations saved to {annotations_file}")

    # Move the processed audio file and annotation to the processed directory
    processed_dir = 'processed'
    shutil.move(enhanced_audio_file, os.path.join(processed_dir, os.path.basename(enhanced_audio_file)))
    shutil.move(annotations_file, os.path.join(processed_dir, os.path.basename(annotations_file)))
    logging.info(f"Moved {enhanced_audio_file} and {annotations_file} to {processed_dir}")

time_calculator("Total time", startTime)
