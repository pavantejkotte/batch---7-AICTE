import whisper
import tempfile
import os
import streamlit as st

class AudioProcessor:
    def __init__(self, model_size="base"):
        self.model = self.load_model(model_size)

    @st.cache_resource
    def load_model(self, model_size):
        print(f"Loading Whisper model: {model_size}...")
        return whisper.load_model(model_size)

    def process_audio(self, uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(uploaded_file.read())
            temp_audio_path = temp_audio.name

        result = self.model.transcribe(temp_audio_path)
        os.remove(temp_audio_path)

        return result["text"]
