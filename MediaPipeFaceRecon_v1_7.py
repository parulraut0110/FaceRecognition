import os
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from insightface.utils import face_align

# Initialize InsightFace & MediaPipe with CUDA support
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)  # Uses GPU if available
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True)

# Directory Paths
KNOWN_FACES_DIR = r"G:\Python Programs\StoredEmbeddings"

# Enable CUDA for OpenCV
cv2.setUseOptimized(True)
cv2.ocl.setUseOpenCL(True)

def extract_landmarks(image):
    """ Extracts facial landmarks using MediaPipe """
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
    """ Generate ArcFace embedding with CUDA support """
    embedding = DeepFace.represent(face_image, model_name="ArcFace", enforce_detection=False, detector_backend="retinaface")
    return np.array(embedding[0]["embedding"]) if embedding else None

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
    
    frame_count = 0  # Counter to track every 30th frame
    match, landmarks, confidence = None, None, None  # Initialize variables to avoid errors
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 == 0:  # Use only every 30th frame for recognition
            match, landmarks, confidence = recognize_face(frame, stored_embeddings)

        # Detect face bounding box
        faces = app.get(frame)
        if faces:
            x1, y1, x2, y2 = faces[0].bbox.astype(int)
            
            # Add padding and ensure within image bounds
            h, w, _ = frame.shape
            padding = int(0.2 * (y2 - y1))  # 20% padding
            x1, y1 = max(x1 - padding, 0), max(y1 - padding, 0)
            x2, y2 = min(x2 + padding, w), min(y2 + padding, h)

            # Crop and maintain aspect ratio
            cropped_face = frame[y1:y2, x1:x2]
            aspect_ratio = (x2 - x1) / (y2 - y1)
            
            if aspect_ratio > 1:  # Wider than tall
                new_width = 640
                new_height = int(640 / aspect_ratio)
            else:  # Taller than wide
                new_height = 640
                new_width = int(640 * aspect_ratio)
            
            cropped_face = cv2.resize(cropped_face, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # Draw landmarks if available
            if landmarks:
                for lm in landmarks.landmark:
                    x, y = int(lm.x * cropped_face.shape[1]), int(lm.y * cropped_face.shape[0])
                    cv2.circle(cropped_face, (x, y), 1, (255, 0, 0), -1)

            label = f"{match} ({confidence:.2f})" if confidence else "Unknown"
            cv2.putText(cropped_face, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display cropped face
            cv2.imshow("Face Recognition", cropped_face)
        else:
            cv2.imshow("Face Recognition", frame)  # Show original if no face detected

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
