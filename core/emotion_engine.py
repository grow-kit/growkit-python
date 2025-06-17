from fer import FER
import cv2
import numpy as np

def analyze_emotion(video_file):
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(video_file)
    emotions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = detector.detect_emotions(frame)
        if result:
            emotions.append(result[0]['emotions'])
    cap.release()

    if emotions:
        avg = {k: sum(d[k] for d in emotions)/len(emotions) for k in emotions[0]}
        return avg
    return {}
