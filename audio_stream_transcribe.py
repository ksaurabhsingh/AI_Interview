import wave
import pyaudio
import numpy as np
import io
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import torch
from glob import glob

# Audio configuration constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
OUTPUT_FILE = "output.wav"
SILENCE_TIMEOUT = 5
SILENCE_THRESHOLD = 500


device = torch.device('cpu')
model, decoder, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_stt',
    jit_model='jit_xlarge',
    language='en',
    device=device
)
(read_batch, split_into_batches, read_audio, prepare_model_input) = utils

def is_silent(data):
    audio_data = np.frombuffer(data, dtype=np.int16)
    energy = np.abs(audio_data).mean()
    return energy < SILENCE_THRESHOLD


def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []
    silent_chunks = 0
    silence_limit_chunks = int(SILENCE_TIMEOUT * RATE / CHUNK)
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            if is_silent(data):
                silent_chunks += 1
                if silent_chunks > silence_limit_chunks:
                    print("No voice detected for 5 seconds. Stopping...")
                    break
            else:
                silent_chunks = 0
    except KeyboardInterrupt:
        print("Recording stopped manually.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        with wave.open(OUTPUT_FILE, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        print(f"Recording saved to {OUTPUT_FILE}")

def split_audio_on_silence(file, min_silence_len=500, silence_thresh=-40, chunk_length=10000):
    audio = AudioSegment.from_file(file)
    chunks = split_on_silence(
        audio, 
        min_silence_len=min_silence_len, 
        silence_thresh=silence_thresh,
        keep_silence=200
    )
    if not chunks or len(chunks) > 100:
        print("Using fixed chunking due to inefficient silence-based splitting...")
        chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]
    return chunks

def transcribe_audio_chunk(chunk, recognizer):
    with io.BytesIO() as temp_audio_file:
        chunk.export(temp_audio_file, format="wav")
        temp_audio_file.seek(0)
        with sr.AudioFile(temp_audio_file) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                return text
            except sr.UnknownValueError:
                return "[Unintelligible]"
            except sr.RequestError as e:
                return f"[Error:{e}]"

def transcribe_audio_chunks(chunks, recognizer):
    transcription = []
    for idx, chunk in enumerate(chunks):
        print(f"Processing_chunk {idx + 1}/{len(chunks)}...")
        text = transcribe_audio_chunk(chunk, recognizer)
        transcription.append(text)
    return " ".join(transcription)

def transcribe_with_silero():
    test_files = glob('output.wav')
    batches = split_into_batches(test_files, batch_size=10)
    input = prepare_model_input(read_batch(batches[0]), device=device)
    output = model(input)
    for example in output:
        return decoder(example.cpu())

# Main workflow function to return transcribed text
def transcribe_audio():
    # Step 1: Record audio
    record_audio()
    
    # Step 2: Split the audio into chunks based on silence
    audio_file = "output.wav"
    chunks = split_audio_on_silence(audio_file)
    
    # Step 3: Initialize speech recognizer and transcribe audio chunks
    recognizer = sr.Recognizer()
    print("Transcribing with Google Speech Recognition...")
    full_transcription = transcribe_audio_chunks(chunks, recognizer)
    print("Full Transcription (Google Speech Recognition):")
    print(full_transcription)
    
    # Optional Step 4: Transcribe using Silero (if needed)
    # Uncomment this line to use Silero transcription instead
    # silero_transcription = transcribe_with_silero()
    # print("Full Transcription (Silero):")
    # print(silero_transcription)
    
    return full_transcription  # or return silero_transcription if you prefer Silero output
