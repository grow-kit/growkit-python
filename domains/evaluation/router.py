# domains/evaluation/router.py
# 실제 HTTP 요청을 처리하는 엔드포인트 정의
# Controller 역할

import os
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from domains.evaluation.schemas import EvaluationRequest, AnalysisResult
from core.gpt_engine import generate_question_with_manual, generate_feedback
from domains.evaluation.service import analyze_video_all

router = APIRouter()


# @router.post("/test", response_model=TranscriptionResult)
# async def test(file: UploadFile = File(...)):
#     binary = await file.read()
#     ext = os.path.splitext(file.filename)[-1] or ".mp3"
#     text = transcribe_audio(binary, suffix=ext)
#     return TranscriptionResult(text=text)
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
        head_pose={
            "head_yaw": result["gaze_direction"],
            "head_pitch": result["head_motion"]
        }
    )


@router.post("/submit-answer")
async def submit_answer(
        video: UploadFile = File(...),
        question: str = Form(...)
):
    binary = await video.read()

    # 1. 분석 수행 (STT + 시선/고개)
    analysis = analyze_video_all(binary)

    # 2. 분석 결과 정리
    answer = analysis["text"]
    emotion_data = {
        "gaze": analysis["gaze_direction"],
        "head": analysis["head_motion"]
    }

    # 3. GPT 채점/피드백 생성
    gpt_result = generate_feedback(question, answer, emotion_data)

    # 4. 결과 응답
    return {
        "question": question,
        "answer": answer,
        "gaze": emotion_data["gaze"],
        "head": emotion_data["head"],
        "score": gpt_result["score"],
        "feedback": gpt_result["feedback"]
    }
