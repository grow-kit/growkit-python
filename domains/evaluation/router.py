# domains/evaluation/router.py
#실제 HTTP 요청을 처리하는 엔드포인트 정의
#Controller 역할

from fastapi import APIRouter, File, UploadFile
from domains.evaluation.service import transcribe_audio
from domains.evaluation.schemas import TranscriptionResult

router = APIRouter()

import os

@router.post("/test", response_model=TranscriptionResult)
async def test(file: UploadFile = File(...)):
    binary = await file.read()
    ext = os.path.splitext(file.filename)[-1] or ".mp3"
    text = transcribe_audio(binary, suffix=ext)
    return TranscriptionResult(text=text)