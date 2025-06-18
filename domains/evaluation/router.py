# domains/evaluation/router.py
#실제 HTTP 요청을 처리하는 엔드포인트 정의
#Controller 역할

from fastapi import APIRouter, File, UploadFile
from fastapi import HTTPException
from domains.evaluation.service import transcribe_audio
from domains.evaluation.schemas import TranscriptionResult
from domains.evaluation.schemas import EvaluationRequest
from core.gpt_engine import generate_question_with_manual
from core.gpt_engine import generate_feedback
from domains.evaluation.service import transcribe_audio, analyze_video_all
from domains.evaluation.schemas import TranscriptionResult,AnalysisResult

router = APIRouter()

import os

@router.post("/test", response_model=TranscriptionResult)
async def test(file: UploadFile = File(...)):
    binary = await file.read()
    ext = os.path.splitext(file.filename)[-1] or ".mp3"
    text = transcribe_audio(binary, suffix=ext)
    return TranscriptionResult(text=text)
# end def


# 문항 생성 요청
@router.get("/generate-question/{manual_id}")
async def get_question(manual_id: int):
    try:
        question = await generate_question_with_manual(manual_id)
        return {"question": question}
    except HTTPException as e:
        # FastAPI에 다시 예외 전달
        raise e
# end def

# 분석 및 피드백 요청
@router.post("/analyze-response")
async def analyze_response(request: EvaluationRequest):
    result = generate_feedback(request.question, request.answer, request.emotion)
    return result
# end def


@router.post("/audio-video", response_model=AnalysisResult)
async def analyze_from_single_video(video: UploadFile = File(...)):
    binary = await video.read()
    result = analyze_video_all(binary)
    return AnalysisResult(
        text=result["text"],
        emotion=result["emotion"],
        head_pose={
            "head_yaw": result["gaze_direction"],
            "head_pitch": result["head_motion"]
        }
    )