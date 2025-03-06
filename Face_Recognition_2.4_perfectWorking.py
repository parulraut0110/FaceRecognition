import face_recognition
import dlib
import os
import cv2
import numpy as np
from deepface import DeepFace
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from sklearn.metrics.pairwise import cosine_similarity

# Initialize InsightFace Detector
app = FaceAnalysis(name="buffalo_l")  # Large model for better accuracy
app.prepare(ctx_id=0)  # Use GPU (ctx_id=0) if available

def detect_and_align_face(image):
    """ Detect face, align it, and return the cropped face with bounding box coordinates """
    faces = app.get(image)
    if not faces:
        return None, None  # No face detected
    face = faces[0]  # Take the first detected face
    aligned_face = face_align.norm_crop(image, landmark=face.kps)  # Align face
    bbox = face.bbox.astype(int)  # Get bounding box
    return aligned_face, bbox

def generate_embedding(face_image):
    """ Convert face into embedding using ArcFace """
    return DeepFace.represent(face_image, model_name="ArcFace", enforce_detection=False)[0]["embedding"]

def calculate_similarity(embedding1, embedding2):
    """ Compute Cosine Similarity between embeddings """
    return cosine_similarity([embedding1], [embedding2])[0][0]

def preprocess_image(image):
    """ Apply pre-processing like noise filtering """
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def recognize_face(frame, known_faces):
    """ Detect & recognize face from frame """
    frame = preprocess_image(frame)  # Noise filtering
    aligned_face, bbox = detect_and_align_face(frame)
    
    if aligned_face is None:
        return None, None
    
    # Generate embedding for detected face
    test_embedding = generate_embedding(aligned_face)

    # Compare with known embeddings
    best_match = None
    best_distance = -1
    for name, embeddings in known_faces.items():
        for embedding in embeddings:
            similarity = calculate_similarity(test_embedding, embedding)
            if similarity > best_distance:  # Higher similarity is better
                best_distance = similarity
                best_match = name

    return best_match if best_distance > 0.6 else None, bbox  # Threshold set at 0.6

# Load known faces
def load_known_faces():
    known_faces = {}
    for name in os.listdir("known_Faces"):
        known_faces[name] = []
        image_path = os.path.join("known_Faces", name)
        image = cv2.imread(image_path)
        aligned_face, _ = detect_and_align_face(image)
        if aligned_face is not None:
            embedding = generate_embedding(aligned_face)
            known_faces[name].append(embedding)
    return known_faces

if __name__ == "__main__":
    known_faces = load_known_faces()
    print(f"Loaded {len(known_faces)} known faces.")

    cap = cv2.VideoCapture(0)  # Ensure using the correct camera index
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        match, bbox = recognize_face(frame, known_faces)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if match else (0, 0, 255)  # Green if recognized, red otherwise
            label = match if match else "Unknown"

            # Draw bounding rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()