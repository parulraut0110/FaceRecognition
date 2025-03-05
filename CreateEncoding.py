import os
import cv2
import numpy as np
from deepface import DeepFace
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# Initialize InsightFace Detector
app = FaceAnalysis(name="buffalo_l")  # Large model for better accuracy
app.prepare(ctx_id=0)  # Use GPU (ctx_id=0) if available

def detect_and_align_face(image):
    """ Detect face, align it, and return the cropped face with bounding box coordinates """
    faces = app.get(image)
    if not faces:
        return None  # No face detected
    face = faces[0]  # Take the first detected face
    aligned_face = face_align.norm_crop(image, landmark=face.kps)  # Align face
    return aligned_face

def generate_embedding(face_image):
    """ Convert face into embedding using ArcFace """
    embedding = DeepFace.represent(face_image, model_name="ArcFace", enforce_detection=False)
    if embedding:
        return np.array(embedding[0]["embedding"])
    return None

# Process images and store embeddings as .npy files
def convert_images_to_embeddings():
    folder_path = "known_Faces"
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, file)
            image = cv2.imread(image_path)

            aligned_face = detect_and_align_face(image)
            if aligned_face is not None:
                embedding = generate_embedding(aligned_face)
                if embedding is not None:
                    # Save embedding as .npy with the same name as the original file
                    embedding_filename = os.path.splitext(image_path)[0] + ".npy"
                    np.save(embedding_filename, embedding)
                    print(f"Saved embedding: {embedding_filename}")

if __name__ == "__main__":
    convert_images_to_embeddings()
    print("Embeddings stored successfully.")
