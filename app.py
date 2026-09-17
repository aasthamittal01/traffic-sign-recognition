import os
import requests
import joblib
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from PIL import Image
from skimage.feature import hog
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Traffic Sign Recognition & Driver Assistance",
    layout="wide"
)

YOLO_MODEL_PATH = "traffic_sign_detector.pt"
SVM_MODEL_PATH = "traffic_sign_final_model.pkl"

YOLO_MODEL_URL = (
    "https://github.com/aasthamittal01/"
    "traffic-sign-recognition/releases/download/"
    "v1.0/traffic_sign_detector.pt"
)

SVM_MODEL_URL = (
    "https://github.com/aasthamittal01/"
    "traffic-sign-recognition/releases/download/"
    "v1.0/traffic_sign_final_model.pkl"
)


# ============================================================
# DOWNLOAD MODELS
# ============================================================

def download_file(url, path):
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


@st.cache_resource
def load_models():

    if not os.path.exists(YOLO_MODEL_PATH):
        with st.spinner("Downloading YOLO11n model..."):
            download_file(
                YOLO_MODEL_URL,
                YOLO_MODEL_PATH
            )

    if not os.path.exists(SVM_MODEL_PATH):
        with st.spinner("Downloading SVM model..."):
            download_file(
                SVM_MODEL_URL,
                SVM_MODEL_PATH
            )

    detector = YOLO(YOLO_MODEL_PATH)
    classifier = joblib.load(SVM_MODEL_PATH)

    return detector, classifier


detector, classifier = load_models()


# ============================================================
# GTSRB SIGN NAMES
# ============================================================

SIGN_NAMES = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
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
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing"
]


# ============================================================
# SIGN CATEGORY
# ============================================================

def get_category(sign_name):

    if "Speed limit" in sign_name:
        return "Regulatory"

    if (
        "No passing" in sign_name
        or "No entry" in sign_name
        or "prohibited" in sign_name
        or sign_name == "No vehicles"
    ):
        return "Prohibitory"

    warning_signs = [
        "General caution",
        "Dangerous curve to the left",
        "Dangerous curve to the right",
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

    if sign_name in warning_signs:
        return "Warning"

    priority_signs = [
        "Right-of-way at the next intersection",
        "Priority road",
        "Yield",
        "Stop"
    ]

    if sign_name in priority_signs:
        return "Priority"

    if "End of" in sign_name:
        return "Derestriction"

    return "Mandatory"


# ============================================================
# DRIVER ASSISTANCE DATABASE
# ============================================================

DRIVER_ACTIONS = {

    "Speed limit (20km/h)":
        "Maintain speed at or below 20 km/h.",

    "Speed limit (30km/h)":
        "Maintain speed at or below 30 km/h.",

    "Speed limit (50km/h)":
        "Maintain speed at or below 50 km/h.",

    "Speed limit (60km/h)":
        "Maintain speed at or below 60 km/h.",

    "Speed limit (70km/h)":
        "Maintain speed at or below 70 km/h.",

    "Speed limit (80km/h)":
        "Maintain speed at or below 80 km/h.",

    "Speed limit (100km/h)":
        "Maintain speed at or below 100 km/h.",

    "Speed limit (120km/h)":
        "Maintain speed at or below 120 km/h.",

    "End of speed limit (80km/h)":
        "The previous 80 km/h speed restriction ends.",

    "No passing":
        "Do not overtake other vehicles.",

    "No passing for vehicles over 3.5 metric tons":
        "Heavy vehicles must not overtake.",

    "Right-of-way at the next intersection":
        "Prepare to follow the right-of-way rule at the next intersection.",

    "Priority road":
        "You have priority on the upcoming road.",

    "Yield":
        "Slow down and give way to other road users.",

    "Stop":
        "Stop completely and proceed only when safe.",

    "No vehicles":
        "Do not enter with a vehicle.",

    "Vehicles over 3.5 metric tons prohibited":
        "Heavy vehicles above the specified weight are prohibited.",

    "No entry":
        "Do not enter this road from this direction.",

    "General caution":
        "Proceed carefully and remain alert.",

    "Dangerous curve to the left":
        "Slow down and prepare for a left curve.",

    "Dangerous curve to the right":
        "Slow down and prepare for a right curve.",

    "Double curve":
        "Reduce speed and prepare for successive curves.",

    "Bumpy road":
        "Reduce speed and prepare for uneven road conditions.",

    "Slippery road":
        "Reduce speed and avoid sudden braking or steering.",

    "Road narrows on the right":
        "Reduce speed and prepare for a narrower roadway.",

    "Road work":
        "Slow down and watch for road work and workers.",

    "Traffic signals":
        "Watch for upcoming traffic signals.",

    "Pedestrians":
        "Slow down and watch carefully for pedestrians.",

    "Children crossing":
        "Slow down and watch carefully for children.",

    "Bicycles crossing":
        "Slow down and watch for crossing bicycles.",

    "Beware of ice/snow":
        "Reduce speed and drive cautiously on potentially icy roads.",

    "Wild animals crossing":
        "Reduce speed and watch for animals crossing.",

    "End of all speed and passing limits":
        "Previous speed and passing restrictions end.",

    "Turn right ahead":
        "Prepare to turn right ahead.",

    "Turn left ahead":
        "Prepare to turn left ahead.",

    "Ahead only":
        "Continue straight ahead.",

    "Go straight or right":
        "Continue straight or turn right.",

    "Go straight or left":
        "Continue straight or turn left.",

    "Keep right":
        "Keep to the right side of the road.",

    "Keep left":
        "Keep to the left side of the road.",

    "Roundabout mandatory":
        "Enter the roundabout according to the indicated direction.",

    "End of no passing":
        "The previous no-passing restriction ends."
}


# ============================================================
# PRIORITY SYSTEM
# ============================================================

HIGH_PRIORITY = [
    "Stop",
    "Yield",
    "No entry",
    "No vehicles",
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)"
]

MEDIUM_PRIORITY = [
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
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


def get_priority(sign_name):

    if sign_name in HIGH_PRIORITY:
        return "HIGH"

    if sign_name in MEDIUM_PRIORITY:
        return "MEDIUM"

    return "LOW"


def priority_score(priority):

    if priority == "HIGH":
        return 3

    if priority == "MEDIUM":
        return 2

    return 1


# ============================================================
# IMAGE LOCATION
# ============================================================

def get_location(x1, y1, x2, y2, width, height):

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    if center_x < width / 3:
        horizontal = "Left"

    elif center_x < 2 * width / 3:
        horizontal = "Center"

    else:
        horizontal = "Right"

    if center_y < height / 3:
        vertical = "Upper"

    elif center_y < 2 * height / 3:
        vertical = "Middle"

    else:
        vertical = "Lower"

    return f"{horizontal} - {vertical}"


# ============================================================
# HOG FEATURE EXTRACTION
# ============================================================

def extract_hog_features(crop):

    crop = cv2.resize(crop, (32, 32))

    gray = np.mean(
        crop,
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
# ATTENTION LEVEL
# ============================================================

def get_attention_level(detections):

    if not detections:
        return "NO SIGN DETECTED"

    priorities = [
        d["Priority"]
        for d in detections
    ]

    if "HIGH" in priorities:
        return "HIGH ATTENTION"

    if "MEDIUM" in priorities:
        return "CAUTION"

    return "NORMAL"


# ============================================================
# SCENE INSTRUCTION ENGINE
# ============================================================

def generate_scene_instruction(detections):

    if not detections:
        return (
            "No traffic sign was detected. "
            "Continue normal visual observation."
        )

    instructions = []

    # STOP
    if any(
        d["Detected Sign"] == "Stop"
        for d in detections
    ):
        instructions.append(
            "STOP: Come to a complete stop and proceed only when safe."
        )

    # Speed limits
    speed_values = []

    for d in detections:

        if "Speed limit" in d["Detected Sign"]:

            try:

                value = int(
                    d["Detected Sign"]
                    .split("(")[1]
                    .split("km/h")[0]
                )

                speed_values.append(value)

            except:
                pass

    if speed_values:

        lowest_speed = min(speed_values)

        instructions.append(
            f"Speed control: Maintain speed at or below "
            f"{lowest_speed} km/h."
        )

    # Yield
    if any(
        d["Detected Sign"] == "Yield"
        for d in detections
    ):
        instructions.append(
            "Give way to other road users before proceeding."
        )

    # Curves
    if any(
        "curve" in d["Detected Sign"].lower()
        for d in detections
    ):
        instructions.append(
            "Curve warning: Reduce speed and prepare "
            "for changing road direction."
        )

    # Road work
    if any(
        d["Detected Sign"] == "Road work"
        for d in detections
    ):
        instructions.append(
            "Road work detected: Slow down and remain "
            "alert for workers or changes in road layout."
        )

    # Slippery
    if any(
        d["Detected Sign"] == "Slippery road"
        for d in detections
    ):
        instructions.append(
            "Slippery-road warning: Avoid sudden braking "
            "or steering."
        )

    # Pedestrians / children
    if any(
        d["Detected Sign"] in [
            "Pedestrians",
            "Children crossing"
        ]
        for d in detections
    ):
        instructions.append(
            "Pedestrian warning: Reduce speed and watch "
            "carefully for people crossing."
        )

    # Overtaking
    if any(
        "No passing" in d["Detected Sign"]
        for d in detections
    ):
        instructions.append(
            "Overtaking restriction: Do not overtake."
        )

    if not instructions:

        instructions.append(
            "Follow the detected road signs and "
            "continue with appropriate caution."
        )

    return "\n".join(
        f"• {instruction}"
        for instruction in instructions
    )


# ============================================================
# ROAD-SCENE ANALYSIS
# ============================================================

def recognize_road_image(input_image):

    if input_image is None:

        return (
            None,
            "Please upload a road image.",
            pd.DataFrame(),
            None
        )

    image = np.array(input_image)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    original = image.copy()

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    yolo_results = detector.predict(
        source=image,
        conf=0.15,
        verbose=False
    )

    boxes = yolo_results[0].boxes

    detections = []

    # --------------------------------------------------------
    # PROCESS DETECTED SIGNS
    # --------------------------------------------------------

    for box in boxes:

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        detection_confidence = float(
            box.conf[0]
            .cpu()
            .numpy()
        )

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # ----------------------------------------------------
        # EXPANDED CROP
        # ----------------------------------------------------

        box_width = x2 - x1
        box_height = y2 - y1

        pad_x = int(box_width * 0.10)
        pad_y = int(box_height * 0.10)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(width, x2 + pad_x)
        cy2 = min(height, y2 + pad_y)

        crop = original[
            cy1:cy2,
            cx1:cx2
        ]

        if crop.size == 0:
            continue

        # ----------------------------------------------------
        # HOG
        # ----------------------------------------------------

        features = extract_hog_features(crop)

        # ----------------------------------------------------
        # SVM
        # ----------------------------------------------------

        predicted_class = int(
            classifier.predict([features])[0]
        )

        sign_name = SIGN_NAMES[predicted_class]

        category = get_category(sign_name)

        priority = get_priority(sign_name)

        action = DRIVER_ACTIONS.get(
            sign_name,
            "Proceed carefully and follow the traffic sign."
        )

        location = get_location(
            x1,
            y1,
            x2,
            y2,
            width,
            height
        )

        detections.append({

            "Class ID":
                predicted_class,

            "Detected Sign":
                sign_name,

            "Category":
                category,

            "Priority":
                priority,

            "Detection Confidence (%)":
                round(
                    detection_confidence * 100,
                    1
                ),

            "Location":
                location,

            "Driver Assistance":
                action,

            "Crop":
                crop,

            "Box":
                (x1, y1, x2, y2)
        })

    # ========================================================
    # SORT BY PRIORITY
    # ========================================================

    detections.sort(
        key=lambda d:
        -priority_score(d["Priority"])
    )

    # ========================================================
    # ANNOTATED IMAGE
    # ========================================================

    annotated = original.copy()

    for d in detections:

        x1, y1, x2, y2 = d["Box"]

        confidence = d[
            "Detection Confidence (%)"
        ]

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        label = (
            f"{d['Detected Sign']} "
            f"| {confidence:.0f}%"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.55

        thickness = 2

        text_size, baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        text_width = text_size[0]

        text_height = text_size[1]

        label_y = max(
            y1 - 8,
            text_height + 10
        )

        cv2.rectangle(
            annotated,
            (
                x1,
                label_y - text_height - baseline - 5
            ),
            (
                x1 + text_width + 8,
                label_y + 5
            ),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            annotated,
            label,
            (x1 + 3, label_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    annotated_rgb = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    # ========================================================
    # NO DETECTION
    # ========================================================

    if not detections:

        summary = """
## Road Scene Analysis

### Detection Overview

**Traffic signs detected:** 0

**Attention Level:** NO SIGN DETECTED

No traffic sign was detected above the current
detection threshold.

Continue normal visual observation of the road.
"""

        return (
            annotated_rgb,
            summary,
            pd.DataFrame(),
            None
        )

    # ========================================================
    # SCENE STATISTICS
    # ========================================================

    total_signs = len(detections)

    high_count = sum(
        d["Priority"] == "HIGH"
        for d in detections
    )

    medium_count = sum(
        d["Priority"] == "MEDIUM"
        for d in detections
    )

    low_count = sum(
        d["Priority"] == "LOW"
        for d in detections
    )

    categories = sorted(
        set(
            d["Category"]
            for d in detections
        )
    )

    attention_level = get_attention_level(
        detections
    )

    highest_priority = detections[0]

    scene_instruction = generate_scene_instruction(
        detections
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = f"""
## Road Scene Analysis

### Detection Overview

**Traffic signs detected:** {total_signs}

**Overall attention level:** {attention_level}

**High priority:** {high_count}

**Medium priority:** {medium_count}

**Low priority:** {low_count}

**Categories present:** {", ".join(categories)}

---

### Highest-Priority Sign

**{highest_priority["Detected Sign"]}**

**Category:** {highest_priority["Category"]}

**Priority:** {highest_priority["Priority"]}

**Location:** {highest_priority["Location"]}

**Detection confidence:** {highest_priority["Detection Confidence (%)"]}%

---

### Combined Driver Assistance

{scene_instruction}

---

### Processing Pipeline

**1. YOLO11n**

Detects and localizes traffic signs in the road scene.

↓

**2. Sign Cropping**

Each detected sign is isolated from the road image.

↓

**3. HOG Feature Extraction**

Extracts visual shape and edge features.

↓

**4. SVM Classification**

Classifies the cropped sign into one of the 43 GTSRB classes.

↓

**5. Metadata Analysis**

Adds category, priority, location and driving guidance.

↓

**6. Scene-Level Interpretation**

Combines multiple detected signs into a single road-safety summary.
"""

    # ========================================================
    # RESULT TABLE
    # ========================================================

    table_data = []

    for d in detections:

        table_data.append({

            "Class ID":
                d["Class ID"],

            "Detected Sign":
                d["Detected Sign"],

            "Category":
                d["Category"],

            "Priority":
                d["Priority"],

            "Detection Confidence (%)":
                d["Detection Confidence (%)"],

            "Location":
                d["Location"],

            "Driver Assistance":
                d["Driver Assistance"]
        })

    result_table = pd.DataFrame(
        table_data
    )

    # ========================================================
    # SIGN CROP CONTACT SHEET
    # ========================================================

    crop_images = []

    for i, d in enumerate(detections):

        crop = d["Crop"].copy()

        crop = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB
        )

        crop = cv2.resize(
            crop,
            (180, 180)
        )

        crop = cv2.copyMakeBorder(
            crop,
            45,
            5,
            5,
            5,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )

        cv2.putText(
            crop,
            f"Sign {i + 1}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        sign_text = d["Detected Sign"]

        if len(sign_text) > 24:
            sign_text = sign_text[:24] + "..."

        cv2.putText(
            crop,
            sign_text,
            (8, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        crop_images.append(crop)

    if crop_images:

        crop_sheet = np.hstack(
            crop_images
        )

    else:

        crop_sheet = None

    return (
        annotated_rgb,
        summary,
        result_table,
        crop_sheet
    )


# ============================================================
# SIMULATOR
# ============================================================

SCENARIOS = {

    "STOP": {
        "name": "STOP",
        "action": "Vehicle slows down and stops",
        "target": 0,
        "behavior": "stop"
    },

    "Speed Limit 50": {
        "name": "Speed Limit 50",
        "action": "Vehicle reduces speed to 50 km/h",
        "target": 50,
        "behavior": "slow"
    },

    "Dangerous Curve Right": {
        "name": "Dangerous Curve Right",
        "action": "Vehicle slows and follows the right curve",
        "target": 35,
        "behavior": "curve-right"
    },

    "YIELD": {
        "name": "YIELD",
        "action": "Vehicle slows and continues carefully",
        "target": 25,
        "behavior": "yield"
    },

    "Road Work": {
        "name": "Road Work",
        "action": "Vehicle slows and proceeds carefully",
        "target": 30,
        "behavior": "slow"
    },

    "Turn Right Ahead": {
        "name": "Turn Right Ahead",
        "action": "Vehicle slows and takes the right turn",
        "target": 30,
        "behavior": "turn-right"
    },

    "Turn Left Ahead": {
        "name": "Turn Left Ahead",
        "action": "Vehicle slows and takes the left turn",
        "target": 30,
        "behavior": "turn-left"
    }
}


def simulator():

    st.markdown("---")

    st.header(
        "Driving Assistance Simulation"
    )

    col1, col2 = st.columns(2)

    with col1:

        scenario = st.selectbox(
            "Traffic Sign",
            list(SCENARIOS.keys())
        )

    with col2:

        initial_speed = st.slider(
            "Initial Speed (km/h)",
            min_value=20,
            max_value=100,
            value=65,
            step=5
        )

    data = SCENARIOS[scenario]

    st.markdown(
        f"""
        ### Selected Scenario

        **Traffic Sign:** {data["name"]}

        **Driver Assistance:** {data["action"]}

        **Target Speed:** {data["target"]} km/h
        """
    )

    if st.button(
        "Start Simulation",
        type="primary"
    ):

        placeholder = st.empty()

        current_speed = float(initial_speed)

        steps = 30

        for step in range(steps):

            progress = step / (steps - 1)

            if data["behavior"] == "stop":

                if progress < 0.65:

                    current_speed = initial_speed

                else:

                    current_speed = (
                        initial_speed *
                        (1 - (progress - 0.65) / 0.35)
                    )

                    current_speed = max(
                        0,
                        current_speed
                    )

            else:

                if progress < 0.45:

                    current_speed = initial_speed

                else:

                    reduction = (
                        initial_speed - data["target"]
                    )

                    current_speed = (
                        initial_speed -
                        reduction *
                        ((progress - 0.45) / 0.55)
                    )

                    current_speed = max(
                        data["target"],
                        current_speed
                    )

            car_position = int(
                5 + progress * 90
            )

            if data["behavior"] == "stop":

                status = (
                    "Vehicle approaching STOP sign..."
                    if current_speed > 0
                    else
                    "STOP detected — Vehicle has stopped."
                )

            elif data["behavior"] == "curve-right":

                status = (
                    "Vehicle approaching dangerous curve..."
                    if progress < 0.5
                    else
                    "Dangerous curve detected — "
                    "Vehicle slowing and following the right curve."
                )

            elif data["behavior"] == "turn-right":

                status = (
                    "Vehicle approaching turn..."
                    if progress < 0.5
                    else
                    "Right turn ahead — Vehicle slowing "
                    "and taking the right turn."
                )

            elif data["behavior"] == "turn-left":

                status = (
                    "Vehicle approaching turn..."
                    if progress < 0.5
                    else
                    "Left turn ahead — Vehicle slowing "
                    "and taking the left turn."
                )

            elif data["behavior"] == "yield":

                status = (
                    "Vehicle approaching YIELD sign..."
                    if progress < 0.5
                    else
                    "YIELD detected — Vehicle slowing "
                    "and continuing carefully."
                )

            else:

                status = (
                    "Vehicle approaching traffic sign..."
                    if progress < 0.5
                    else
                    f"{data['name']} detected — "
                    "Vehicle reducing speed."
                )

            # ====================================================
            # SIMULATION VISUAL
            # ====================================================

            simulation_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    body {{
        margin: 0;
        padding: 0;
        background: transparent;
        overflow: hidden;
    }}

    .scene {{
        width: 100%;
        height: 180px;
        background: linear-gradient(
            #9ed8ff 0%,
            #dff3ff 55%,
            #78ad62 55%
        );
        border-radius: 14px;
        position: relative;
        overflow: hidden;
        border: 1px solid #ccc;
        box-sizing: border-box;
    }}

    .road {{
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 65px;
        background: #404348;
    }}

    .road-line {{
        position: absolute;
        top: 30px;
        left: 0;
        width: 100%;
        border-top: 3px dashed white;
    }}

    .sign {{
        position: absolute;
        right: 12%;
        bottom: 70px;
        font-size: 45px;
    }}

    .car {{
        position: absolute;
        left: {car_position}%;
        bottom: 25px;
        font-size: 45px;
        transform: translateX(-50%);
    }}
</style>
</head>

<body>

<div class="scene">

    <div class="road">
        <div class="road-line"></div>
    </div>

    <div class="sign">🛑</div>

    <div class="car">🚗</div>

</div>

</body>
</html>
"""

            with placeholder.container():

                st.markdown("### Road Simulation")

                st.markdown(
                    f"**{scenario}**"
                )

                components.html(
                    simulation_html,
                    height=190,
                    scrolling=False
                )

                st.markdown(
                    f"**Current Vehicle Speed:** "
                    f"{round(current_speed)} km/h"
                )

                st.markdown(
                    f"**Status:** {status}"
                )

            import time
            time.sleep(0.12)


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

st.title(
    "Traffic Sign Recognition & Driver Assistance System"
)

st.write(
    "YOLO11n → Detection → HOG → SVM → "
    "Recognition → Metadata → Driver Assistance"
)

tab1, tab2 = st.tabs(
    [
        "Road Scene Analysis",
        "Driving Assistance Simulation"
    ]
)


# ============================================================
# TAB 1 — ROAD SCENE
# ============================================================

with tab1:

    st.subheader(
        "YOLO11n + HOG + SVM"
    )

    st.write(
        "Upload a road-scene image containing "
        "one or more traffic signs."
    )

    input_image = st.file_uploader(
        "Upload Road Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if input_image is not None:

        image = Image.open(
            input_image
        ).convert("RGB")

        st.image(
            image,
            caption="Input Road Image",
            use_container_width=True
        )

        if st.button(
            "Analyze Road Scene",
            type="primary"
        ):

            with st.spinner(
                "Analyzing road scene..."
            ):

                (
                    annotated,
                    summary,
                    result_table,
                    crop_sheet
                ) = recognize_road_image(
                    image
                )

            st.markdown(
                "## Detected Traffic Signs"
            )

            st.image(
                annotated,
                use_container_width=True
            )

            st.markdown(
                "## Road Scene Intelligence"
            )

            st.markdown(summary)

            if not result_table.empty:

                st.markdown(
                    "## Detected Sign Details"
                )

                st.dataframe(
                    result_table,
                    use_container_width=True,
                    hide_index=True
                )

            if crop_sheet is not None:

                st.markdown(
                    "## Individual Sign Analysis"
                )

                st.image(
                    crop_sheet,
                    use_container_width=True
                )


# ============================================================
# TAB 2 — SIMULATION
# ============================================================

with tab2:

    simulator()
