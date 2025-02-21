import os
import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans

# ✅ Move this to the top
st.set_page_config(layout="wide")  # Mobile-friendly layout

# Load EfficientNetB0 for feature extraction
@st.cache_resource
def load_model():
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    return tf.keras.models.Model(inputs=base_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))

model = load_model()
IMG_SIZE = (224, 224)
GLCM_DISTANCES = [1, 2]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Extract GLCM texture features
def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=GLCM_DISTANCES, angles=GLCM_ANGLES, symmetric=True, normed=True)
    features = np.hstack([graycoprops(glcm, prop).flatten() for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']])
    return features

# Extract features from image
def extract_features(image):
    img_resized = cv2.resize(image, IMG_SIZE)
    img_preprocessed = preprocess_input(img_resized)
    cnn_feature = model.predict(np.expand_dims(img_preprocessed, axis=0))[0]
    glcm_feature = extract_glcm_features(cv2.resize(image, (128, 128)))
    return np.hstack([cnn_feature, glcm_feature])

# Corrected dataset path
dataset_path = r"mri-prediction-app/data/Alzheimer_s Dataset/test/ModerateDemented"

# Ensure dataset path exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

# Get list of image files
dataset_images = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png'))]

# Load and train clustering models
@st.cache_resource
def train_clustering_models():
    feature_vectors = []
    
    for img_path in dataset_images:
        image = cv2.imread(img_path)
        if image is not None:
            feature_vectors.append(extract_features(image))
    
    feature_vectors = np.array(feature_vectors)
    
    if feature_vectors.shape[0] > 0:
        gender_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(feature_vectors)
        age_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(feature_vectors)
        return gender_kmeans, age_kmeans
    else:
        raise ValueError("No valid images found in dataset for training.")

gender_kmeans, age_kmeans = train_clustering_models()

# Prediction function
def predict(image):
    features = extract_features(image)
    gender_cluster = gender_kmeans.predict([features])[0]
    age_cluster = age_kmeans.predict([features])[0]
    gender_label = "Male" if gender_cluster == 0 else "Female"
    age_label = ["Young", "Middle-aged", "Old"][age_cluster]
    return gender_label, age_label

# ✅ UI Components Start Here
st.title("MRI Gender & Age Prediction App")
uploaded_file = st.file_uploader("Upload an MRI JPG image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)
    
    try:
        gender, age = predict(image)
        st.write(f"**Predicted Gender:** {gender}")
        st.write(f"**Predicted Age Group:** {age}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
