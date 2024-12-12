"""
This Streamlit application is designed to analyze and visualize the evolution 
of car designs by comparing key design features between two images. 
The application uses pre-trained deep learning models to:
- Detect bounding boxes and keypoints on car images (Model 1: Keypoints + Bounding Box).
- Predict keypoints only for car images (Model 2: Keypoints Only).

**Features:**
1. Upload and compare two car images via the sidebar.
2. Display bounding boxes and keypoints for each image using Model 1.
3. Display keypoints only for each image using Model 2.
4. Calculate and display the "Design Difference" score for both models, 
   which quantifies the changes in design between the two uploaded images.

**File Dependencies:**
1. `models.py` - Contains the model architectures:
   - `KeypointAndBboxModel` (Model 1).
   - `EnhancedKeypointsModel` (Model 2).
2. `keypoints_boundingbox_model.pth` - Pre-trained weights for Model 1.
3. `keypoints_only_model2.pth` - Pre-trained weights for Model 2.

**How to Use:**
1. Place `Front_end.py` and `models.py` in the same directory.
2. Ensure the model files are stored in the `Models` subdirectory.
3. Run the application:
   ```bash
   streamlit run Front_end.py
4. Upload two car images via the sidebar.
5. View the results, including visualizations and design difference scores, in the main interface.

Important Notes:

The application automatically rescales uploaded images to a resolution of 1280x855.
Bounding boxes are used to ensure keypoints are constrained within the detected region (Model 1 only).
For training details, refer to Backend.ipynb, where the models were trained using a dataset of 17 car images. """


import streamlit as st
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from models import KeypointAndBboxModel, EnhancedKeypointsModel

# Set up Streamlit page
st.set_page_config(page_title="Tracing Car Design Evolution with AI", layout="wide")

# GIF integration at the top of the page
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://usagif.com/wp-content/uploads/gifs/race-car-11.gif" alt="Car Animation" style="width: 50%; border-radius: 10px;">
    </div>
    <h1 style="color: #ff5722; text-align: center;">Tracing Car Design Evolution with AI</h1>
    <h3 style="color: #ffccbc; text-align: center;">Detecting change, one keypoint at a time.</h3>
    <hr style="border: 1px solid #ff5722;">
    """,
    unsafe_allow_html=True
)

# Device configuration for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to resize images
def resize_image(image_file, target_size=(1280, 855)):
    image = Image.open(image_file)
    return image.resize(target_size)

# Function to load pre-trained models
def load_saved_model(model_path):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

# Transform for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Sidebar: Upload images
st.sidebar.header("Upload Images for Comparison")
file1 = st.sidebar.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
file2 = st.sidebar.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])

# Load pre-trained models
model1_path = "Models/keypoints_boundingbox_model.pth"
model2_path = "Models/keypoints_only_model2.pth"
model1 = load_saved_model(model1_path)
model2 = load_saved_model(model2_path)

# Enforce keypoints within bounding box
def enforce_keypoint_within_bbox(bbox, keypoints):
    x_min, y_min, width, height = bbox
    x_max, y_max = x_min + width, y_min + height
    keypoints[:, 0] = np.clip(keypoints[:, 0], x_min, x_max)
    keypoints[:, 1] = np.clip(keypoints[:, 1], y_min, y_max)
    return keypoints

# Function for Model 1 inference
def predict_keypoints_and_bbox(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(image_tensor)
    bbox = predictions[0, :4].cpu().numpy()
    keypoints = predictions[0, 4:].cpu().numpy().reshape(-1, 2)
    
    # Scale predictions back to original dimensions
    bbox[0] *= 1280
    bbox[1] *= 850
    bbox[2] *= 1280
    bbox[3] *= 850
    keypoints[:, 0] *= 1280
    keypoints[:, 1] *= 850
    
    # Enforce keypoints within bounding box (only for Model 1)
    keypoints = enforce_keypoint_within_bbox(bbox, keypoints)
    
    return bbox, keypoints

# Function for Model 2 inference
def predict_keypoints(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(image_tensor).cpu().numpy().reshape(-1, 2)
    predictions[:, 0] *= 1280  # Scale x coordinates
    predictions[:, 1] *= 850   # Scale y coordinates
    return predictions


# Function to overlay keypoints and bounding boxes
def plot_keypoints_and_bbox(image_path, bbox, keypoints):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1280, 855))
    if bbox is not None:
        x_min, y_min, width, height = bbox
        cv2.rectangle(image, (int(x_min), int(y_min)), 
                      (int(x_min + width), int(y_min + height)), 
                      (255, 0, 0), 2)
    for x, y in keypoints:
        cv2.circle(image, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to compute normalized Euclidean distance
def compute_design_difference_from_arrays(keypoints1, keypoints2):
    length1 = keypoints1[:, 0].max() - keypoints1[:, 0].min()
    height1 = keypoints1[:, 1].max() - keypoints1[:, 1].min()
    length2 = keypoints2[:, 0].max() - keypoints2[:, 0].min()
    height2 = keypoints2[:, 1].max() - keypoints2[:, 1].min()
    keypoints1_normalized = np.copy(keypoints1)
    keypoints2_normalized = np.copy(keypoints2)
    keypoints1_normalized[:, 0] = (keypoints1[:, 0] - keypoints1[:, 0].min()) / length1
    keypoints1_normalized[:, 1] = (keypoints1[:, 1] - keypoints1[:, 1].min()) / height1
    keypoints2_normalized[:, 0] = (keypoints2[:, 0] - keypoints2[:, 0].min()) / length2
    keypoints2_normalized[:, 1] = (keypoints2[:, 1] - keypoints2[:, 1].min()) / height2
    distances = np.sqrt(np.sum((keypoints1_normalized - keypoints2_normalized) ** 2, axis=1))
    return np.sum(distances)

# Main app logic
if file1 and file2:
    st.markdown("<div style='text-align: center;'><h2>Inference Results</h2></div>", unsafe_allow_html=True)

    # Resize and process uploaded images
    temp_path1 = "temp_image1.jpg"
    temp_path2 = "temp_image2.jpg"
    img1 = resize_image(file1, target_size=(1280, 855))  # Resize Image 1
    img1.save(temp_path1)  # Save resized Image 1
    img2 = resize_image(file2, target_size=(1280, 855))  # Resize Image 2
    img2.save(temp_path2)  # Save resized Image 2

    # Model 1 Predictions
    bbox1_m1, keypoints1_m1 = predict_keypoints_and_bbox(temp_path1, model1)
    bbox2_m1, keypoints2_m1 = predict_keypoints_and_bbox(temp_path2, model1)

    # Model 2 Predictions
    keypoints1_m2 = predict_keypoints(temp_path1, model2)
    keypoints2_m2 = predict_keypoints(temp_path2, model2)

    # Compute design differences
    design_diff_m1 = compute_design_difference_from_arrays(keypoints1_m1, keypoints2_m1)
    design_diff_m2 = compute_design_difference_from_arrays(keypoints1_m2, keypoints2_m2)

    # Display Model 1 results
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>Model 1 (Bounding Box + Keypoints)</h2>", unsafe_allow_html=True)
        st.image(plot_keypoints_and_bbox(temp_path1, bbox1_m1, keypoints1_m1), caption="Image 1 (Model 1)", use_container_width=True)
        st.image(plot_keypoints_and_bbox(temp_path2, bbox2_m1, keypoints2_m1), caption="Image 2 (Model 1)", use_container_width=True)
        st.markdown(f"<h4 style='text-align: center;'>Design Difference: {design_diff_m1:.4f}</h4>", unsafe_allow_html=True)

    # Display Model 2 results
    with col2:
        st.markdown("<h2 style='text-align: center;'>Model 2 (Keypoints Only)</h2>", unsafe_allow_html=True)
        st.image(plot_keypoints_and_bbox(temp_path1, None, keypoints1_m2), caption="Image 1 (Model 2)", use_container_width=True)
        st.image(plot_keypoints_and_bbox(temp_path2, None, keypoints2_m2), caption="Image 2 (Model 2)", use_container_width=True)
        st.markdown(f"<h4 style='text-align: center;'>Design Difference: {design_diff_m2:.4f}</h4>", unsafe_allow_html=True)
