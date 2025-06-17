# domains/evaluation/service.py
#whisper 모델을 불러와서 binary 오디오 데이터를 받아 텍스트로 변환하는 핵심 로직

import os
import whisper
import tempfile

model = whisper.load_model("base")

def transcribe_audio(binary_data: bytes, suffix=".mp3") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(binary_data)
        temp_file.flush()
        result = model.transcribe(temp_file.name)
        return result["text"]

