import torch
from decalib.deca import DECA
from decalib.utils import util
import cv2
import numpy as np
import open3d as o3d  # Import Open3D for 3D visualization

# Load DECA model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
deca = DECA(config=None, device=device)

# Load an image
image_path = r"C:\Users\parul\Face_Rec_Env\known_faces\Parul.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image
inputs = {'image': torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0).to(device)}
with torch.no_grad():
    codedict = deca.encode(inputs)

# Generate 3D face mesh
opdict, _ = deca.decode(codedict)

# Extract vertices & faces
vertices = opdict['vertices'][0].cpu().numpy()
faces = opdict['faces'][0].cpu().numpy()

# Convert faces to Open3D format
faces = np.asarray(faces, dtype=np.int32)

# Create Open3D Mesh
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(faces)

# Apply basic color & smooth shading
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.6, 0.6, 0.8])  # Light blue color

# Create Open3D visualization window
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.add_geometry(mesh)

# Rotation angle (in degrees)
rotation_angle = 5  

# Define rotation function
def rotate_left(vis):
    R = mesh.get_rotation_matrix_from_axis_angle([0, np.radians(rotation_angle), 0])  # Rotate left
    mesh.rotate(R, center=(0, 0, 0))
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

def rotate_right(vis):
    R = mesh.get_rotation_matrix_from_axis_angle([0, -np.radians(rotation_angle), 0])  # Rotate right
    mesh.rotate(R, center=(0, 0, 0))
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

# Assign key callbacks
vis.register_key_callback(ord("A"), rotate_left)   # Press 'A' to rotate left
vis.register_key_callback(ord("D"), rotate_right)  # Press 'D' to rotate right

print("Press 'A' to rotate left, 'D' to rotate right, and 'Esc' to exit.")

# Run Open3D visualization
vis.run()
vis.destroy_window()
