# app.py

import streamlit as st
import whisper
import tempfile
import os

@st.cache_resource
def load_model(model_size="tiny"):
    return whisper.load_model(model_size)

LANG_DISPLAY = {
    "Auto": None,
    "English": "en",
    "Korean": "ko",
    "Japanese": "ja",
}

st.set_page_config(page_title="STT Demo", layout="centered")
st.title("🎙️ STT Demo: Whisper + Streamlit")

st.subheader("📤 Upload Audio File")
uploaded_file = st.file_uploader("Only .wav or .mp3 files", type=["wav", "mp3"])

st.subheader("🛠️ Model Settings")
col1, col2 = st.columns(2)
with col1:
    model_size = st.selectbox("Model", ["tiny", "base", "small"], index=0)
with col2:
    language = st.selectbox("Language", list(LANG_DISPLAY.keys()), index=0)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if st.button("Transcribe"):
        st.info("Loading model and transcribing...")
        model = load_model(model_size)
        result = model.transcribe(tmp_path, language=LANG_DISPLAY[language])
        st.success("Transcription Complete!")
        st.text_area("Transcribed Text", value=result["text"], height=200)
        st.download_button("Download Result", result["text"], file_name="transcription.txt")
else:
    st.warning("Please upload an audio file to proceed.")
