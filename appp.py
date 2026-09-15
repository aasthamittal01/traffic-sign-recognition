import os
import requests
import joblib
import numpy as np
import gradio as gr

from PIL import Image
from skimage.feature import hog
from ultralytics import YOLO


# ============================================================
# MODEL PATHS
# ============================================================

YOLO_MODEL_PATH = "traffic_sign_detector.pt"
SVM_MODEL_PATH = "traffic_sign_final_model.pkl"


# ============================================================
# DOWNLOAD LARGE YOLO MODEL IF NEEDED
# ============================================================

def download_file(url, path):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


# Later we will put the real hosted model URL here.
YOLO_MODEL_URL = os.getenv("YOLO_MODEL_URL")

if not os.path.exists(YOLO_MODEL_PATH):
    if not YOLO_MODEL_URL:
        raise RuntimeError(
            "YOLO model is missing. Set the YOLO_MODEL_URL environment variable."
        )

    print("Downloading YOLO model...")
    download_file(YOLO_MODEL_URL, YOLO_MODEL_PATH)
    print("YOLO model downloaded.")


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

print("Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)

print("Loading SVM model...")
svm_model = joblib.load(SVM_MODEL_PATH)

print("Models loaded successfully.")


# ============================================================
# GTSRB 43 SIGN NAMES
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

    if (
        sign_name in [
            "Turn right ahead",
            "Turn left ahead",
            "Ahead only",
            "Go straight or right",
            "Go straight or left",
            "Keep right",
            "Keep left",
            "Roundabout mandatory"
        ]
    ):
        return "Mandatory"

    if sign_name in [
        "Yield",
        "Stop",
        "Priority road"
    ]:
        return "Priority"

    if (
        sign_name in [
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
        ]
    ):
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
# RECOGNIZE ONE DETECTED SIGN
# ============================================================

def recognize_crop(crop):

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
# FINAL ROAD-SCENE ANALYSIS
# ============================================================

def analyze_road_image(image):

    if image is None:
        return "Please upload a road image."

    image = image.convert("RGB")

    image_array = np.array(image)

    height, width = image_array.shape[:2]

    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = yolo_model(
        image_array,
        conf=0.25
    )

    detections = []

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

            # Safety bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop(
                (x1, y1, x2, y2)
            )

            # ------------------------------------------------
            # HOG + SVM RECOGNITION
            # ------------------------------------------------

            class_id, sign_name = recognize_crop(
                crop
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

    # --------------------------------------------------------
    # NO SIGN FOUND
    # --------------------------------------------------------

    if not detections:

        return (
            "### No traffic sign detected\n\n"
            "YOLO11n did not find a traffic sign "
            "with the current detection threshold."
        )

    # --------------------------------------------------------
    # FORMAT RESULT
    # --------------------------------------------------------

    output = "## Traffic Sign Analysis\n\n"

    for i, item in enumerate(detections, 1):

        output += f"### Sign {i}\n\n"

        output += (
            f"**Sign:** {item['sign']}\n\n"
        )

        output += (
            f"**YOLO Detection Confidence:** "
            f"{item['detection_confidence'] * 100:.2f}%\n\n"
        )

        output += (
            f"**GTSRB Class:** "
            f"{item['class_id']}\n\n"
        )

        output += (
            f"**Category:** "
            f"{item['category']}\n\n"
        )

        output += (
            f"**Driver Assistance:** "
            f"{item['action']}\n\n"
        )

        output += "---\n\n"

    return output


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="Traffic Sign Recognition & Driver Assistance"
) as demo:

    gr.Markdown(
        """
        # Traffic Sign Recognition & Driver Assistance

        Upload a road image containing traffic signs.

        **YOLO11n → Detection → HOG → SVM → Recognition → Driver Assistance**
        """
    )

    with gr.Row():

        input_image = gr.Image(
            type="pil",
            label="Upload Road Image"
        )

        output = gr.Markdown(
            label="Analysis Result"
        )

    analyze_button = gr.Button(
        "Analyze Road Image",
        variant="primary"
    )

    analyze_button.click(
        fn=analyze_road_image,
        inputs=input_image,
        outputs=output
    )


# ============================================================
# LAUNCH
# ============================================================

demo.launch()
