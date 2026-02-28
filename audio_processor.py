import whisper
import tempfile
import os
import streamlit as st


@st.cache_resource
def load_whisper_model(model_size="base"):
    print(f"Loading Whisper model: {model_size}...")
    return whisper.load_model(model_size)


class AudioProcessor:
    def __init__(self, model_size="base"):
        self.model = load_whisper_model(model_size)

    def process_audio(self, uploaded_file):
        """
        Transcribes uploaded audio file using Whisper.
        Safe for Streamlit Cloud.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(uploaded_file.read())
            temp_audio_path = temp_audio.name

        try:
            result = self.model.transcribe(temp_audio_path)
            return result["text"]
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
