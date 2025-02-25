import os
import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
import joblib  # For saving/loading models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ✅ Set Streamlit Layout
st.set_page_config(layout="wide")

# ✅ Load EfficientNetB0 Model for Feature Extraction
@st.cache_resource
def load_model():
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    return tf.keras.models.Model(inputs=base_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))

model = load_model()

# ✅ Constants
IMG_SIZE = (224, 224)
GLCM_DISTANCES = [1, 2]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# ✅ Extract GLCM Features
def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=GLCM_DISTANCES, angles=GLCM_ANGLES, symmetric=True, normed=True)
    features = np.hstack([graycoprops(glcm, prop).flatten() for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']])
    return features

# ✅ Extract Features (CNN + GLCM)
def extract_features(image):
    img_resized = cv2.resize(image, IMG_SIZE)
    img_preprocessed = preprocess_input(img_resized)
    cnn_feature = model.predict(np.expand_dims(img_preprocessed, axis=0))[0]
    glcm_feature = extract_glcm_features(cv2.resize(image, (128, 128)))
    return np.hstack([cnn_feature, glcm_feature])

# ✅ Define Dataset Path
dataset_path = r"C:\Users\NABIRA\OneDrive\Desktop\madhuri\MRI\Alzheimer_s Dataset\test\ModerateDemented"

# ✅ Ensure Dataset Path Exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

# ✅ Get List of Image Files
dataset_images = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png'))]

# ✅ Load or Train SVM Models
@st.cache_resource
def train_svm_models():
    feature_vectors = []
    gender_labels = []
    age_labels = []

    for img_path in dataset_images:
        image = cv2.imread(img_path)
        if image is not None:
            feature = extract_features(image)
            feature_vectors.append(feature)
            
            # Sample Labels (Update as per your dataset)
            if "Male" in img_path:
                gender_labels.append(0)  # Male → 0
            else:
                gender_labels.append(1)  # Female → 1
            
            if "Young" in img_path:
                age_labels.append(0)  # Young → 0
            elif "Middle" in img_path:
                age_labels.append(1)  # Middle-aged → 1
            else:
                age_labels.append(2)  # Old → 2

    feature_vectors = np.array(feature_vectors)
    gender_labels = np.array(gender_labels)
    age_labels = np.array(age_labels)

    if feature_vectors.shape[0] > 0:
        gender_svm = make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True))
        age_svm = make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True))

        gender_svm.fit(feature_vectors, gender_labels)
        age_svm.fit(feature_vectors, age_labels)

        joblib.dump(gender_svm, "gender_svm_model.pkl")
        joblib.dump(age_svm, "age_svm_model.pkl")

        return gender_svm, age_svm
    else:
        raise ValueError("No valid images found in dataset for training.")

# ✅ Load or Train Models
if os.path.exists("gender_svm_model.pkl") and os.path.exists("age_svm_model.pkl"):
    gender_svm = joblib.load("gender_svm_model.pkl")
    age_svm = joblib.load("age_svm_model.pkl")
else:
    gender_svm, age_svm = train_svm_models()

# ✅ Prediction Function
def predict(image):
    features = extract_features(image)
    gender_pred = gender_svm.predict([features])[0]
    age_pred = age_svm.predict([features])[0]

    gender_label = "Male" if gender_pred == 0 else "Female"
    age_label = ["Young", "Middle-aged", "Old"][age_pred]

    return gender_label, age_label

# ✅ UI Components
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
