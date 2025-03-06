import os
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from insightface.app import FaceAnalysis
from insightface.utils import face_align

destination_dir = ""  # Declare destination_dir as a global variable

def init():
    global destination_dir  # Use the global keyword to modify the global variable
    destination_dir = "face Encodings"
    os.makedirs(destination_dir, exist_ok=True)


# Initialize InsightFace Detector
app = FaceAnalysis(name="buffalo_l")  # Large model for better accuracy
app.prepare(ctx_id=0)  # Use GPU (ctx_id=0) if available

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def detect_and_align_face(image):
    """ Detect face, align it, and return the cropped face with bounding box coordinates """
    faces = app.get(image)
    if not faces:
        return None  # No face detected
    face = faces[0]  # Take the first detected face
    aligned_face = face_align.norm_crop(image, landmark=face.kps)  # Align face
    return aligned_face

def detect_face_landmarks(image):
    """ Detect face landmarks using MediaPipe Face Mesh """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        return None  # No face detected
    
    return results.multi_face_landmarks[0]  # Return the first detected face landmarks

def extract_face_from_landmarks(image, landmarks):
    """ Extract face using facial landmarks (bounding box approach) """
    height, width, _ = image.shape
    x_coords = [int(landmark.x * width) for landmark in landmarks.landmark]
    y_coords = [int(landmark.y * height) for landmark in landmarks.landmark]

    # Define bounding box using min/max coordinates
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Crop the face region
    cropped_face = image[y_min:y_max, x_min:x_max]
    
    return cropped_face if cropped_face.size > 0 else None  # Return None if invalid crop


def generate_embedding(face_image):
    """ Convert face into embedding using ArcFace """
    embedding = DeepFace.represent(face_image, model_name="ArcFace", enforce_detection=False)
    if embedding:
        return np.array(embedding[0]["embedding"])
    return None

# Process images and store embeddings as .npy files
def convert_images_to_embeddings():
    global destination_dir
    folder_path = "Faces"
    os.makedirs(destination_dir, exist_ok=True)  # Ensure the destination directory exists

    for file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file)
        image = cv2.imread(image_path)

        aligned_face = detect_and_align_face(image)

        # Detect facial landmarks using MediaPipe
        landmarks = detect_face_landmarks(aligned_face)
        if landmarks:
            refined_face = extract_face_from_landmarks(aligned_face, landmarks)
            if refined_face is not None:
                embedding = generate_embedding(refined_face)
                if embedding is not None:
                    # Ensure filename is correctly constructed
                    embedding_filename = os.path.splitext(file)[0] + ".npy"  
                    embedding_filename = os.path.join(destination_dir, embedding_filename)  
                    
                    np.save(embedding_filename, embedding)
                    print(f"✅ Saved embedding: {embedding_filename}")
                else:
                    print(f"⚠️ Embedding not generated for {file}")
            else:
                print(f"⚠️ Failed to extract face from {file}")
        else:
            print(f"⚠️ No landmarks detected in {file}")

if __name__ == "__main__":
    init()
    convert_images_to_embeddings()
    print("✅ Embeddings stored successfully.")
