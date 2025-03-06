import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

def match_embeddings(embedding1, embedding2, threshold=0.3):
    """ Compare embeddings using cosine similarity """
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity > (1 - threshold)

def recognize_worker(video_source, database):
    """ Recognizes worker in real-time with multi-frame verification """
    cap = cv2.VideoCapture(video_source)
    matched_worker = None
    confidence_scores = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extract_landmarks(frame)
        if not landmarks:
            continue  # No face detected

        aligned_face = frame  # Align based on landmarks if needed
        real_time_embedding = DeepFace.represent(aligned_face, model_name="ArcFace", enforce_detection=False)
        
        for worker, data in database.items():
            stored_embedding = data["embedding"]
            if match_embeddings(real_time_embedding, stored_embedding):
                confidence_scores[worker] = confidence_scores.get(worker, 0) + 1

        if any(score >= 10 for score in confidence_scores.values()):  # Multi-frame verification
            matched_worker = max(confidence_scores, key=confidence_scores.get)
            break

    cap.release()
    return matched_worker

worker = recognize_worker(0, database)  # 0 for webcam
if worker:
    print(f"✅ Attendance registered for: {worker}")
else:
    print("❌ No match found.")

