# requirements.txt에 다음 추가 (Cloud 환경이면 작동 여부 보장 안 됨)
# ffmpeg-python
# 만약 안되면 로컬 PC나 Docker로 실행 권장

import streamlit as st
import whisper
import tempfile
import os

st.set_page_config(page_title="STT Admin Console", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("##STT Admin Console")
    st.markdown("1. **Audio Upload**\n텍스트로 변환할 음성 파일 업로드 (mp3, wav)")
    st.markdown("2. **Model Setting**\n변환할 모델 및 언어 선택")
    st.markdown("3. **Transcription Result**\n결과 확인 및 다운로드")
    st.button("Log Data")  # 기능 추가 가능

# Header
st.markdown("###Upload Audio File")
uploaded_file = st.file_uploader("Drag and drop an audio file here", type=["mp3", "wav"])

# Model Settings
st.markdown("###STT Model Setting")
col1, col2, col3, col4 = st.columns(4)

with col1:
    model_name = st.selectbox("Model", ["tiny", "base", "small", "medium", "large"], index=1)
with col2:
    language = st.selectbox("Language", ["Auto", "en", "ko", "ja"])
with col3:
    beam_size = st.selectbox("Beam Size", [1, 3, 5, 10], index=2)
with col4:
    noise_suppression = st.selectbox("Noise Suppression", ["None", "Aggressive", "Light"], index=1)

# Transcription Result
if uploaded_file:
    st.markdown("###Transcription")
    if st.button("변환"):
        with st.spinner("Transcribing..."):
            try:
                model = whisper.load_model(model_name)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                result = model.transcribe(tmp_path, language=None if language == "Auto" else language, beam_size=beam_size)
                transcription = result["text"]

                edited_text = st.text_area("Transcribed Text", transcription, height=200)
                st.download_button("다운로드", edited_text, file_name="transcription.txt")
            except Exception as e:
                st.error(f"Error during transcription: {e}")
else:
    st.info("음성 파일을 업로드하세요.")

