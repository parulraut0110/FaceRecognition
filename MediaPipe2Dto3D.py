import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the image
image_path = r"G:\Python Programs\Faces\me.jpg"  # Update this with your image path
image = cv2.imread(image_path)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Convert to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image
results = face_mesh.process(rgb_image)

# Check if landmarks are detected
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        x_list, y_list, z_list = [], [], []
        
        for lm in face_landmarks.landmark:
            x_list.append(lm.x)
            y_list.append(lm.y)
            z_list.append(lm.z)

        # Convert lists to NumPy arrays
        x_list = np.array(x_list)
        y_list = np.array(y_list)
        z_list = np.array(z_list)

        # Plot 3D Face Mesh
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_list, y_list, z_list, c=z_list, cmap='jet', marker='o')

        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Depth (Z Axis)")
        ax.set_title("3D Face Mesh from Static Image")

        plt.show()
else:
    print("No face landmarks detected.")
