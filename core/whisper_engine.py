import whisper
import tempfile
import os

model = whisper.load_model("base")

def transcribe_audio(binary: bytes, filename: str = "audio.mp3") -> str:
    # 확장자 추출 (예: .mp3, .wav)
    ext = os.path.splitext(filename)[-1] or ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_file.write(binary)
        temp_file.flush()

        try:
            result = model.transcribe(temp_file.name, language="ko")  # 한국어 음성 처리
            return result.get("text", "")
        except Exception as e:
            print(f"STT 실패: {e}")
            return "[오류: STT 실패]"
        finally:
            os.remove(temp_file.name)  # 임시 파일 정리
