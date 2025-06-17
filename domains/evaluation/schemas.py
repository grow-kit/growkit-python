#데이터 구조 정의(응답/요청)
#예시
# FastAPI는 요청/응답 데이터 형식을 Pydantic 모델로 정의함
# 예를 들어 업로드된 오디오 파일에 대해 텍스트 결과를 반환할 때 사용하는 구조
# Dto 클래스와 유사

# domains/evaluation/schemas.py

from pydantic import BaseModel

class TranscriptionResult(BaseModel):
    text: str  # Whisper 결과 텍스트