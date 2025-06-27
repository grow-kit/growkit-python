# domains/evaluation/router.py
# 실제 HTTP 요청을 처리하는 엔드포인트 정의
# Controller 역할

import os
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from domains.evaluation.schemas import EvaluationRequest, AnalysisResult
from core.gpt_engine import generate_question_with_manual, fetch_manual, fetch_criteria,generate_feedback_with_criteria
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


# # 분석 및 피드백 요청
# @router.post("/analyze-response")
# async def analyze_response(request: EvaluationRequest):
#     result = generate_feedback(request.question, request.answer, request.emotion)
#     return result
# # end def


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
    video: UploadFile,
    question: str = Form(...),
    manual_id: int = Form(...),
    criteria_id: int = Form(...)
):
    binary = await video.read()

    analysis = analyze_video_all(binary)
    answer = analysis.get("text", "").strip()
    emotion_data = {
        "gaze": analysis.get("gaze_direction", "알 수 없음"),
        "head": analysis.get("head_motion", "알 수 없음")
    }

    # 🎯 answer가 비정상일 경우 고정 응답
    if not answer or answer == "음성 인식 실패" or len(answer) < 5:
        return {
            "question": question,
            "answer": answer,
            "gaze": emotion_data["gaze"],
            "head": emotion_data["head"],
            "score": {},
            "feedback": "⚠️ 답변이 정상적으로 인식되지 않아 평가가 불가능합니다. 문의 후 재평가 요청을 진행해 주세요."
        }

    manual = await fetch_manual(manual_id)
    criteria = await fetch_criteria(criteria_id)

    gpt_result = generate_feedback_with_criteria(
        question, answer, emotion_data, manual, criteria
    )

    return {
        "question": question,
        "answer": answer,
        "gaze": emotion_data["gaze"],
        "head": emotion_data["head"],
        "score": gpt_result["score"],
        "feedback": gpt_result["feedback"]
    }