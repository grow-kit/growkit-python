# domains/evaluation/service.py
#whisper 모델을 불러와서 binary 오디오 데이터를 받아 텍스트로 변환하는 핵심 로직

import os
import cv2
import whisper
from fer import FER
from moviepy.editor import VideoFileClip
import tempfile

# Whisper 모델 로드
model = whisper.load_model("base")

# Cascade 분류기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# 음성만 입력된 경우 STT
def transcribe_audio(binary_data: bytes, suffix=".mp3") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(binary_data)
        temp_file.flush()
        result = model.transcribe(temp_file.name)
    os.remove(temp_file.name)
    return result["text"]


# 영상에서 오디오 추출 후 STT
def transcribe_audio_from_video(video_path: str) -> str:
    audio_path = video_path.replace(".mp4", ".wav")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

    result = model.transcribe(audio_path, language="ko")
    os.remove(audio_path)
    return result["text"]


# 감정 + 시선 + 고개 움직임 분석
def analyze_emotion_and_pose(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    emotion_detector = FER(mtcnn=False)

    emotions_list = []
    gaze_directions = []
    yaw_distances = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        emotions = emotion_detector.detect_emotions(frame)
        emotions_list += [e["emotions"] for e in emotions if e]

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi)

            if len(eyes) >= 2:
                # 시선 추정
                centers = [(ex + ew // 2) for (ex, ey, ew, eh) in eyes]
                avg_x = sum(centers) / len(centers)

                if avg_x < w // 2 - 10:
                    gaze_directions.append("왼쪽")
                elif avg_x > w // 2 + 10:
                    gaze_directions.append("오른쪽")
                else:
                    gaze_directions.append("정면")

                # 고개 움직임 추정
                yaw = abs(centers[0] - centers[1])
                yaw_distances.append(yaw)

    cap.release()

    # 결과 요약
    dominant_emotion = (
        max(set([max(e, key=e.get) for e in emotions_list]), key=[max(e, key=e.get) for e in emotions_list].count)
        if emotions_list else "unknown"
    )
    dominant_gaze = max(set(gaze_directions), key=gaze_directions.count) if gaze_directions else "알 수 없음"
    head_motion = (
        "움직임 있음" if max(yaw_distances, default=0) - min(yaw_distances, default=0) > 10 else "안정적"
    )

    return {
        "emotion": dominant_emotion,
        "gaze_direction": dominant_gaze,
        "head_stability": head_motion
    }


# 전체 분석 통합 함수
def analyze_video_all(binary_video: bytes) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(binary_video)
        temp_file.flush()
        video_path = temp_file.name

    try:
        stt_text = transcribe_audio_from_video(video_path)
        visual_result = analyze_emotion_and_pose(video_path)
    finally:
        os.remove(video_path)

    return {
        "text": stt_text,
        "emotion": visual_result["emotion"],
        "gaze_direction": visual_result["gaze_direction"],
        "head_motion": visual_result["head_stability"]
    }
