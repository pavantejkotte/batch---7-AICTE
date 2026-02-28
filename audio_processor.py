import whisper
import tempfile
import os

class AudioProcessor:
    def __init__(self, model_size="base"):
        print(f"Loading Whisper model: {model_size}...")
        self.model = whisper.load_model(model_size)

    def process_audio(self, uploaded_file):
        """
        Transcribes uploaded audio file using Whisper.
        Works safely on Streamlit Cloud.
        """
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(uploaded_file.read())
            temp_audio_path = temp_audio.name

        print(f"Transcribing {temp_audio_path}...")
        result = self.model.transcribe(temp_audio_path)

        # Clean up temp file
        os.remove(temp_audio_path)

        return result["text"]
