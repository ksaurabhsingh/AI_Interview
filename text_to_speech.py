import torch
import re
import pyaudio
import numpy as np
import soundfile as sf

def text_to_speech(text, max_length=100, language='en', speaker='lj_16khz'):

    def split_text(text, max_length=100):
        if not text.strip():
            raise ValueError("Input text is empty ")
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        chunks = []
        current_chunk = []
        current_length = 0
        for sentence in sentences:
            sentence = sentence.strip()  
            if not sentence:  
                continue
            if current_length + len(sentence) > max_length:
                if current_chunk:  
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, symbols, sample_rate, example_text, apply_tts = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language=language,
        speaker=speaker
    )
    model = model.to(device)
    text_chunks = split_text(text, max_length)

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True)

    all_audio_data = []

    for chunk in text_chunks:
        audio = apply_tts(
            texts=[chunk],
            model=model,
            sample_rate=sample_rate,
            symbols=symbols,
            device=device
        )
        audio_data = audio[0].cpu().numpy()
        stream.write(audio_data.astype(np.float32).tobytes())  # Play live audio

        all_audio_data.append(audio_data)  # Collect audio for later saving

    # Save final audio as a file
    final_audio = np.concatenate(all_audio_data, axis=-1)
    sf.write('output_audio.wav', final_audio, sample_rate)

    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()

    return final_audio
