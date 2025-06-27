# domains/evaluation/service.py
#whisper 모델을 불러와서 binary 오디오 데이터를 받아 텍스트로 변환하는 핵심 로직

import os
import cv2
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip
import tempfile
import subprocess

# 영상 처리 로직
def fix_video_metadata(input_path: str) -> str:
    output_path = input_path.replace(".mp4", "_fixed.mp4")
    # 🔍 실제 컨테이너 검사: 웹M이면 mp4로 변환
    if input_path.endswith(".webm"):
        output_path = input_path.replace(".webm", "_converted.mp4")
        cmd = [
            "ffmpeg", "-i", input_path,
            "-c:v", "libx264", "-c:a", "aac",
            "-movflags", "faststart",
            output_path
        ]
    else:
        cmd = [
            "ffmpeg", "-i", input_path,
            "-c:v", "copy", "-c:a", "copy",
            "-movflags", "faststart",
            output_path
        ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path
#end def

# Whisper 모델 (빠른 버전)
stt_model = WhisperModel("base", device="cpu", compute_type="int8")

# Haar Cascade 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# 🎙️ 영상에서 음성 추출 → STT 텍스트 변환
def transcribe_audio_from_video(video_path: str) -> str:
    fixed_path = fix_video_metadata(video_path)
    audio_path = video_path.replace(".mp4", ".wav")

    clip = VideoFileClip(fixed_path)
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    clip.close()  # ✅ 파일 점유 해제

    segments, _ = stt_model.transcribe(audio_path, language="ko")
    text = " ".join([segment.text for segment in segments])

    os.remove(audio_path)
    os.remove(fixed_path)  # ✅ 재인코딩된 영상도 정리
    return text


# 👁️ 시선 + 고개 움직임 분석
def analyze_pose_only(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    gaze_directions = []
    yaw_distances = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi)

            if len(eyes) >= 2:
                centers = [(ex + ew // 2) for (ex, ey, ew, eh) in eyes]
                avg_x = sum(centers) / len(centers)

                if avg_x < w // 2 - 10:
                    gaze_directions.append("왼쪽")
                elif avg_x > w // 2 + 10:
                    gaze_directions.append("오른쪽")
                else:
                    gaze_directions.append("정면")

                yaw = abs(centers[0] - centers[1])
                yaw_distances.append(yaw)

    cap.release()

    gaze_result = max(set(gaze_directions), key=gaze_directions.count) if gaze_directions else "알 수 없음"
    head_motion = (
        "움직임 있음" if max(yaw_distances, default=0) - min(yaw_distances, default=0) > 10 else "안정적"
    )

    return {
        "gaze_direction": gaze_result,
        "head_stability": head_motion
    }


# 🔄 전체 분석 통합
def analyze_video_all(binary_video: bytes) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(binary_video)
        temp_file.flush()
        video_path = temp_file.name

    stt_text = "음성 인식 실패"
    pose_result = {
        "gaze_direction": "알 수 없음",
        "head_stability": "알 수 없음"
    }

    try:
        try:
            stt_text = transcribe_audio_from_video(video_path)
        except Exception as e:
            print(f"🎙️ 음성 분석 실패: {e}")

        try:
            pose_result = analyze_pose_only(video_path)
        except Exception as e:
            print(f"👁️ 시선 분석 실패: {e}")

    finally:
        os.remove(video_path)

    return {
        "text": stt_text,
        "gaze_direction": pose_result.get("gaze_direction", "알 수 없음"),
        "head_motion": pose_result.get("head_stability", "알 수 없음")
    }




