import os
import requests
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from skimage.feature import hog
from ultralytics import YOLO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Traffic Sign Recognition & Driver Assistance",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# MODEL PATHS
# ============================================================

YOLO_MODEL_PATH = "traffic_sign_detector.pt"
SVM_MODEL_PATH = "traffic_sign_final_model.pkl"

YOLO_MODEL_URL = (
    "https://github.com/aasthamittal01/"
    "traffic-sign-recognition/releases/download/v1.0/"
    "traffic_sign_detector.pt"
)

SVM_MODEL_URL = (
    "https://github.com/aasthamittal01/"
    "traffic-sign-recognition/releases/download/v1.0/"
    "traffic_sign_final_model.pkl"
)


# ============================================================
# DOWNLOAD MODEL
# ============================================================

def download_file(url, path):
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    with open(path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    if not os.path.exists(YOLO_MODEL_PATH):
        with st.spinner("Downloading YOLO11n model..."):
            download_file(YOLO_MODEL_URL, YOLO_MODEL_PATH)

    if not os.path.exists(SVM_MODEL_PATH):
        with st.spinner("Downloading SVM model..."):
            download_file(SVM_MODEL_URL, SVM_MODEL_PATH)

    yolo_model = YOLO(YOLO_MODEL_PATH)
    svm_model = joblib.load(SVM_MODEL_PATH)

    return yolo_model, svm_model


# ============================================================
# GTSRB SIGN NAMES
# ============================================================

SIGN_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 tons"
}


# ============================================================
# SIGN CATEGORIES
# ============================================================

def get_category(sign_name):

    if (
        "Speed limit" in sign_name
        or "No passing" in sign_name
        or "prohibited" in sign_name
        or "No vehicles" in sign_name
        or "No entry" in sign_name
    ):
        return "Regulatory / Prohibitory"

    if sign_name in [
        "Turn right ahead",
        "Turn left ahead",
        "Ahead only",
        "Go straight or right",
        "Go straight or left",
        "Keep right",
        "Keep left",
        "Roundabout mandatory"
    ]:
        return "Mandatory"

    if sign_name in [
        "Yield",
        "Stop",
        "Priority road"
    ]:
        return "Priority"

    if sign_name in [
        "General caution",
        "Dangerous curve left",
        "Dangerous curve right",
        "Double curve",
        "Bumpy road",
        "Slippery road",
        "Road narrows on the right",
        "Road work",
        "Traffic signals",
        "Pedestrians",
        "Children crossing",
        "Bicycles crossing",
        "Beware of ice/snow",
        "Wild animals crossing"
    ]:
        return "Warning"

    if "End of" in sign_name:
        return "Derestriction"

    return "Regulatory"


# ============================================================
# DRIVER ASSISTANCE RULES
# ============================================================

DRIVER_ACTIONS = {

    "Stop":
        "Stop the vehicle and proceed only when it is safe.",

    "Yield":
        "Slow down and give way to other traffic.",

    "Priority road":
        "You have priority at upcoming intersections.",

    "Speed limit (20km/h)":
        "Reduce speed to 20 km/h.",

    "Speed limit (30km/h)":
        "Reduce speed to 30 km/h.",

    "Speed limit (50km/h)":
        "Reduce speed to 50 km/h.",

    "Speed limit (60km/h)":
        "Reduce speed to 60 km/h.",

    "Speed limit (70km/h)":
        "Reduce speed to 70 km/h.",

    "Speed limit (80km/h)":
        "Reduce speed to 80 km/h.",

    "Speed limit (100km/h)":
        "Reduce speed to 100 km/h.",

    "Speed limit (120km/h)":
        "Reduce speed to 120 km/h.",

    "Dangerous curve left":
        "Slow down and prepare for a left curve.",

    "Dangerous curve right":
        "Slow down and prepare for a right curve.",

    "Road work":
        "Slow down and proceed carefully.",

    "Pedestrians":
        "Slow down and watch for pedestrians.",

    "Children crossing":
        "Slow down and watch carefully for children.",

    "Slippery road":
        "Reduce speed and drive carefully.",

    "Turn right ahead":
        "Slow down and prepare to turn right.",

    "Turn left ahead":
        "Slow down and prepare to turn left.",

    "Keep right":
        "Keep the vehicle to the right.",

    "Keep left":
        "Keep the vehicle to the left.",

    "Roundabout mandatory":
        "Slow down and follow the roundabout direction."
}


def get_driver_action(sign_name):

    return DRIVER_ACTIONS.get(
        sign_name,
        "Follow the indicated traffic rule and drive carefully."
    )


# ============================================================
# HOG FEATURE EXTRACTION
# ============================================================

def extract_hog_features(image):

    image = image.convert("RGB")

    image = image.resize((32, 32))

    image = np.array(image)

    gray = np.mean(
        image,
        axis=2
    ).astype(np.uint8)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2)
    )

    return features


# ============================================================
# SVM RECOGNITION
# ============================================================

def recognize_crop(crop, svm_model):

    features = extract_hog_features(crop)

    features = np.array(features).reshape(1, -1)

    class_id = int(
        svm_model.predict(features)[0]
    )

    sign_name = SIGN_NAMES.get(
        class_id,
        f"Class {class_id}"
    )

    return class_id, sign_name


# ============================================================
# ROAD IMAGE ANALYSIS
# ============================================================

def analyze_road_image(image, yolo_model, svm_model):

    if image is None:
        return None, []

    image = image.convert("RGB")

    image_array = np.array(image)

    height, width = image_array.shape[:2]

    # YOLO detects traffic sign locations
    results = yolo_model(
        image_array,
        conf=0.25
    )

    detections = []

    # Draw detections on image
    annotated_image = image.copy()

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop detected sign
            crop = image.crop(
                (x1, y1, x2, y2)
            )

            # HOG + SVM recognition
            class_id, sign_name = recognize_crop(
                crop,
                svm_model
            )

            category = get_category(
                sign_name
            )

            action = get_driver_action(
                sign_name
            )

            detections.append({
                "sign": sign_name,
                "class_id": class_id,
                "detection_confidence": confidence,
                "category": category,
                "action": action,
                "box": (x1, y1, x2, y2)
            })

    return annotated_image, detections


# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "Traffic Sign Recognition & Driver Assistance System"
)

st.write(
    "Detect traffic signs in road scenes, "
    "recognize them using HOG + SVM, and "
    "generate driver-assistance guidance."
)

st.markdown(
    """
    **Pipeline**

    Road Image → YOLO11n Detection → Sign Crop →
    HOG Feature Extraction → SVM Recognition →
    Sign Interpretation → Driver Assistance
    """
)

st.divider()


# ============================================================
# LOAD MODELS
# ============================================================

try:

    yolo_model, svm_model = load_models()

except Exception as error:

    st.error(
        "Model loading failed."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Road Image",
        use_container_width=True
    )

    if st.button(
        "Analyze Road Image",
        type="primary"
    ):

        with st.spinner(
            "Detecting and recognizing traffic signs..."
        ):

            annotated_image, detections = analyze_road_image(
                image,
                yolo_model,
                svm_model
            )

        if not detections:

            st.warning(
                "No traffic sign detected with "
                "the current detection threshold."
            )

        else:

            st.success(
                f"{len(detections)} traffic sign(s) detected."
            )

            st.subheader(
                "Traffic Sign Analysis"
            )

            for index, item in enumerate(
                detections,
                start=1
            ):

                st.markdown(
                    f"### Sign {index}"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        f"**Sign:** {item['sign']}"
                    )

                    st.write(
                        f"**GTSRB Class:** "
                        f"{item['class_id']}"
                    )

                    st.write(
                        f"**Category:** "
                        f"{item['category']}"
                    )

                with col2:

                    st.write(
                        f"**YOLO Detection Confidence:** "
                        f"{item['detection_confidence'] * 100:.2f}%"
                    )

                    st.write(
                        "**Driver Assistance:**"
                    )

                    st.info(
                        item["action"]
                    )

                st.divider()
