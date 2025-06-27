from faster_whisper import WhisperModel
import tempfile
import os

# 모델 로드 (CPU: compute_type="int8", GPU: "float16" 또는 "int8_float16")
model = WhisperModel("small", device="cpu", compute_type="int8")

def transcribe_audio(binary: bytes, filename: str = "audio.mp3") -> str:
    ext = os.path.splitext(filename)[-1] or ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_file.write(binary)
        temp_file.flush()

        try:
            segments, _ = model.transcribe(temp_file.name, language="ko")
            text = " ".join([seg.text for seg in segments])
            return text
        except Exception as e:
            print(f"STT 실패: {e}")
            return "[오류: STT 실패]"
        finally:
            os.remove(temp_file.name)
