import os
import whisper
from pydub import AudioSegment

class AudioProcessor:
    def __init__(self, model_size="base"):
        print(f"Loading Whisper model: {model_size}...")
        self.model = whisper.load_model(model_size)

    def process_audio(self, file_path):
        """Transcribes audio file using Whisper."""
        print(f"Transcribing {file_path}...")
        result = self.model.transcribe(file_path)
        return result["text"]

    def convert_to_wav(self, input_path):
        """Converts audio to WAV if it's not already."""
        if input_path.endswith(".wav"):
            return input_path
        
        output_path = input_path.rsplit(".", 1)[0] + ".wav"
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return output_path
