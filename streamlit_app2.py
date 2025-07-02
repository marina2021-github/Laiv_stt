# requirements.txt에 다음 추가 (Cloud 환경이면 작동 여부 보장 안 됨)
# ffmpeg-python
# 만약 안되면 로컬 PC나 Docker로 실행 권장

import streamlit as st
import tempfile
import os
from faster_whisper import WhisperModel

# 페이지 설정
st.set_page_config(page_title="STT Demo: Whisper + Streamlit", layout="centered")

# 타이틀 및 소개
st.title("🎙️ STT Demo: Whisper + Streamlit")
st.markdown("### 📤 Upload Audio File")
st.markdown("Only .wav or .mp3 files")

# 파일 업로드
uploaded_file = st.file_uploader("Drag and drop an audio file here", type=["wav", "mp3"])

# 모델 설정 옵션
st.markdown("### ⚙️ Model Settings")
model_size = st.selectbox("Model", ["tiny", "base", "small", "medium", "large"], index=1)
language_option = st.selectbox("Language", ["Auto", "en", "ko", "ja", "zh", "fr", "de"], index=0)

# 모델 로딩 함수
@st.cache_resource
def load_model(model_size):
    return WhisperModel(model_size, compute_type="float32")

# 변환 버튼
if uploaded_file and st.button("🎧 Transcribe"):
    with st.spinner("Loading model and transcribing..."):
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # 모델 로딩
        model = load_model(model_size)

        # 언어 설정
        lang_code = None if language_option == "Auto" else language_option

        try:
            # 추론
            segments, info = model.transcribe(tmp_path, language=lang_code)

            # 결과 출력
            st.markdown("### 📝 Transcription Result")
            full_text = " ".join([seg.text for seg in segments])
            st.text_area("Transcription", full_text, height=300)

        except Exception as e:
            st.error(f"Transcription failed: {e}")

        finally:
            os.remove(tmp_path)
