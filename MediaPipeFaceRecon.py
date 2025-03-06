import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Load the reference image and extract landmarks
def get_landmarks_from_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in face_landmarks.landmark]
            return landmarks
    return None

# Load stored face image landmarks
reference_image_path = r"G:\Python Programs\Static Images For Testing\me.jpg"  # Replace with your stored image path
stored_landmarks = get_landmarks_from_image(reference_image_path)

if stored_landmarks is None:
    print("No face detected in stored image. Exiting.")
    exit()

# Start webcam for real-time detection
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as live_face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = live_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                live_landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark]

                # Draw stored landmarks overlay
                for point in stored_landmarks:
                    cv2.circle(frame, point, 1, (0, 255, 0), -1)  # Green dots for stored image

                # Draw real-time landmarks overlay
                for point in live_landmarks:
                    cv2.circle(frame, point, 1, (0, 0, 255), -1)  # Red dots for real-time landmarks

                # Compute similarity (Euclidean Distance)
                diff = np.linalg.norm(np.array(live_landmarks) - np.array(stored_landmarks))
                similarity_score = max(0, 100 - diff * 0.1)  # Scale similarity score

                # Display similarity score
                cv2.putText(frame, f"Match Score: {similarity_score:.2f}%", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow("Face Recognition - Landmark Matching", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
