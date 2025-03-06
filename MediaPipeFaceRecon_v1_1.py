import cv2
import mediapipe as mp
import numpy as np
import json
from scipy.spatial import procrustes

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Path for storing landmarks
landmark_storage = "stored_landmarks.json"

# Function to extract and save facial landmarks
def store_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])  # Normalized
            data = {"landmarks": landmarks.tolist(), "size": image.shape[:2]}  # Store image size
            with open(landmark_storage, "w") as f:
                json.dump(data, f)
            return landmarks, image.shape[:2]
    
    return None, None

# Load stored landmarks
def load_landmarks():
    try:
        with open(landmark_storage, "r") as f:
            data = json.load(f)
        return np.array(data["landmarks"]), tuple(data["size"])
    except FileNotFoundError:
        print("Stored landmark file not found. Please store a face first.")
        return None, None

# Function to align and compare landmarks
def align_and_compare(landmarks1, landmarks2):
    landmarks1, landmarks2, disparity = procrustes(landmarks1, landmarks2)
    return disparity

# Store subject's image landmarks if not already stored
stored_image_path = r"G:\Python Programs\Faces\me.jpg"  # Update as needed
stored_landmarks, stored_size = load_landmarks()

if stored_landmarks is None:
    stored_landmarks, stored_size = store_landmarks(stored_image_path)
    if stored_landmarks is None:
        print("No face detected in stored image. Exiting.")
        exit()

# Start webcam for real-time comparison
cap = cv2.VideoCapture(0)
live_face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

frame_skip = 5  # Process every 5th frame to control frequency
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:  # Skip frames for better performance
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = live_face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            live_landmarks = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])

            # Resize live landmarks to match stored image size
            height, width = frame.shape[:2]
            live_landmarks[:, 0] *= stored_size[1] / width  # Adjust x-coordinates
            live_landmarks[:, 1] *= stored_size[0] / height  # Adjust y-coordinates

            # Compare stored and real-time landmarks
            similarity_score = align_and_compare(stored_landmarks, live_landmarks)
            match_score = max(0, 100 - similarity_score * 100)  # Scale score

            # Create a small pop-up window
            pop_up = np.zeros((200, 400, 3), dtype=np.uint8)  # Black background
            cv2.putText(pop_up, f"Match Score: {match_score:.2f}%", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Draw stored landmarks in Green
            for x, y in stored_landmarks * [stored_size[1], stored_size[0]]:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

            # Draw live landmarks in Red
            for x, y in live_landmarks * [stored_size[1], stored_size[0]]:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), -1)

            # Display match score on the main window
            cv2.putText(frame, f"Match Score: {match_score:.2f}%", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show small pop-up window
            cv2.imshow("Matching Score", pop_up)

    # Show main window with landmarks
    cv2.imshow("Face Recognition - Landmark Matching", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
