import os
import cv2
import numpy as np
from deepface import DeepFace
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from sklearn.metrics.pairwise import cosine_similarity

url = "http://192.168.1.101:8080/video"  # Corrected IP & video feed URL


# Initialize InsightFace Detector
app = FaceAnalysis(name="buffalo_l")  # Large model for better accuracy
app.prepare(ctx_id=0)  # Use GPU (ctx_id=0) if available

# Directory Paths
KNOWN_FACES_DIR = r"C:\Users\parul\Face_Rec_Env\known_faces\New folder"

def load_stored_embeddings():
    """ Load stored embeddings from .npy files instead of processing images at runtime """
    stored_embeddings = {}

    for file in os.listdir(KNOWN_FACES_DIR):
        if file.endswith(".npy"):  # Only load .npy files
            name = os.path.splitext(file)[0]  # Extract name without extension
            embedding_path = os.path.join(KNOWN_FACES_DIR, file)
            stored_embeddings[name] = np.load(embedding_path)

    return stored_embeddings

def recognize_face(frame, stored_embeddings, threshold=0.6):
    """ Detect, align, and recognize faces from a live webcam frame """
    faces = app.get(frame)
    if not faces:
        return None, None  # No face detected

    face = faces[0]  # Take the first detected face
    aligned_face = face_align.norm_crop(frame, landmark=face.kps)  # Align face

    # Generate embedding
    embedding = DeepFace.represent(aligned_face, model_name="ArcFace", enforce_detection=False)
    if not embedding:
        return None, None

    input_embedding = np.array(embedding[0]["embedding"])
    
    # Compare with stored embeddings
    best_match = None
    highest_similarity = -1

    for name, stored_embedding in stored_embeddings.items():
        similarity = cosine_similarity([input_embedding], [stored_embedding])[0][0]

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = name

    if highest_similarity >= threshold:
        return best_match, face.bbox.astype(int)  # Return recognized name and bounding box
    return "Unknown", face.bbox.astype(int)  # No match found

if __name__ == "__main__":
    stored_embeddings = load_stored_embeddings()
    print(f"Loaded {len(stored_embeddings)} stored embeddings.")

    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width, _ = frame.shape  # Get original dimensions

        # Define the cropping region (center 640x640)
        crop_size = 800
        start_x = max((width - crop_size) // 2, 0)
        start_y = max((height - crop_size) // 2, 0)
        end_x = start_x + crop_size
        end_y = start_y + crop_size

        # Ensure cropping does not exceed frame boundaries
        #cropped_frame = frame[start_y:end_y, start_x:end_x]

        cv2.imshow("IP Webcam Feed (Cropped)", frame)
        match, bbox = recognize_face(frame, stored_embeddings)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if match != "Unknown" else (0, 0, 255)  # Green if recognized, red otherwise
            label = match if match else "Unknown"

            # Draw bounding rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
