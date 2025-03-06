import os
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from insightface.utils import face_align

# Initialize InsightFace & MediaPipe
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)  # Use GPU if available
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True)

# Directory Paths
KNOWN_FACES_DIR = r"G:\Python Programs\StoredEmbeddings"

def extract_landmarks(image):
    """ Extracts facial lan+-dmarks using MediaPipe """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]  # Return first detected face
    return None

def detect_and_align_face(image):
    """ Detect, align face, and refine using landmarks """
    faces = app.get(image)
    if not faces:
        return None, None
    
    face = faces[0]
    aligned_face = face_align.norm_crop(image, landmark=face.kps)
    landmarks = extract_landmarks(aligned_face)
    
    return aligned_face, landmarks

def generate_embedding(face_image):
    """ Generate ArcFace embedding """
    embedding = DeepFace.represent(face_image, model_name="ArcFace", enforce_detection=False)
    return np.array(embedding[0]["embedding"]) if embedding else None

def convert_images_to_embeddings():
    """ Process images and store embeddings """
    for file in os.listdir(KNOWN_FACES_DIR):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(KNOWN_FACES_DIR, file)
            image = cv2.imread(image_path)

            aligned_face, landmarks = detect_and_align_face(image)
            if aligned_face is not None:
                embedding = generate_embedding(aligned_face)
                if embedding is not None:
                    np.save(os.path.splitext(image_path)[0] + ".npy", embedding)
                    print(f"Saved embedding: {image_path}")

# Real-time Face Recognition
url = "http://192.168.166.65:8080/video"

def load_stored_embeddings():
    """ Load stored embeddings """
    stored_embeddings = {}
    for file in os.listdir(KNOWN_FACES_DIR):
        if file.endswith(".npy"):
            name = os.path.splitext(file)[0]
            stored_embeddings[name] = np.load(os.path.join(KNOWN_FACES_DIR, file))
    return stored_embeddings

def recognize_face(frame, stored_embeddings, threshold=0.6):
    """ Detect, align, extract landmarks, and recognize face """
    aligned_face, landmarks = detect_and_align_face(frame)
    if aligned_face is None or landmarks is None:
        return None, None, None

    embedding = generate_embedding(aligned_face)
    if embedding is None:
        return None, None, None

    best_match, highest_similarity = "Unknown", -1
    for name, stored_embedding in stored_embeddings.items():
        similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = name
    
    return best_match if highest_similarity >= threshold else "Unknown", landmarks, highest_similarity

if __name__ == "__main__":
    stored_embeddings = load_stored_embeddings()
    cap = cv2.VideoCapture(url)

    frame_count = 0
    process_interval = 20

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % process_interval == 0:
    
            match, landmarks, confidence = recognize_face(frame, stored_embeddings)
            if landmarks:
                for lm in landmarks.landmark:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

            label = f"{match} ({confidence:.2f})" if confidence else "Unknown"
            cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
