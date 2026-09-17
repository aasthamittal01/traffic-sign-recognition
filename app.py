import os
import json
import base64
import requests
import joblib
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from skimage.feature import hog
from ultralytics import YOLO

st.set_page_config(
    page_title="Traffic Sign Recognition & Driver Assistance",
    layout="wide"
)

# ============================================================
# MODEL DOWNLOAD / LOADING
# ============================================================

YOLO_MODEL_PATH = "traffic_sign_detector.pt"
SVM_MODEL_PATH = "traffic_sign_final_model.pkl"

YOLO_MODEL_URL = (
    "https://github.com/aasthamittal01/traffic-sign-recognition/"
    "releases/download/v1.0/traffic_sign_detector.pt"
)

SVM_MODEL_URL = (
    "https://github.com/aasthamittal01/traffic-sign-recognition/"
    "releases/download/v1.0/traffic_sign_final_model.pkl"
)


def download_file(url, path):
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()

    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


@st.cache_resource
def load_models():
    if not os.path.exists(YOLO_MODEL_PATH):
        download_file(YOLO_MODEL_URL, YOLO_MODEL_PATH)

    if not os.path.exists(SVM_MODEL_PATH):
        download_file(SVM_MODEL_URL, SVM_MODEL_PATH)

    return YOLO(YOLO_MODEL_PATH), joblib.load(SVM_MODEL_PATH)


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
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]


# ============================================================
# SIGN CATEGORIES
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
# DRIVER ASSISTANCE ACTIONS
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
        "The previous no-passing restriction ends.",

    "End of no passing by vehicles over 3.5 metric tons":
        "The previous no-passing restriction for heavy vehicles ends."
}


# ============================================================
# PRIORITY
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
    return {
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1
    }[priority]


# ============================================================
# LOCATION
# ============================================================

def get_location(x1, y1, x2, y2, width, height):

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    horizontal = (
        "Left"
        if center_x < width / 3
        else "Center"
        if center_x < 2 * width / 3
        else "Right"
    )

    vertical = (
        "Upper"
        if center_y < height / 3
        else "Middle"
        if center_y < 2 * height / 3
        else "Lower"
    )

    return f"{horizontal} - {vertical}"


# ============================================================
# HOG FEATURE EXTRACTION
# ============================================================

def extract_hog_features(crop):

    crop = cv2.resize(crop, (32, 32))

    gray = np.mean(crop, axis=2).astype(np.uint8)

    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2)
    )


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
# SCENE-LEVEL DRIVER INSTRUCTION
# ============================================================

def generate_scene_instruction(detections):

    if not detections:
        return (
            "No traffic sign was detected. "
            "Continue normal visual observation."
        )

    instructions = []

    if any(
        d["Detected Sign"] == "Stop"
        for d in detections
    ):
        instructions.append(
            "STOP: Come to a complete stop and proceed only when safe."
        )

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

            except Exception:
                pass

    if speed_values:

        instructions.append(
            f"Speed control: Maintain speed at or below "
            f"{min(speed_values)} km/h."
        )

    if any(
        d["Detected Sign"] == "Yield"
        for d in detections
    ):
        instructions.append(
            "Give way to other road users before proceeding."
        )

    if any(
        "curve" in d["Detected Sign"].lower()
        for d in detections
    ):
        instructions.append(
            "Curve warning: Reduce speed and prepare "
            "for changing road direction."
        )

    if any(
        d["Detected Sign"] == "Road work"
        for d in detections
    ):
        instructions.append(
            "Road work detected: Slow down and remain "
            "alert for workers or changes in road layout."
        )

    if any(
        d["Detected Sign"] == "Slippery road"
        for d in detections
    ):
        instructions.append(
            "Slippery-road warning: Avoid sudden braking or steering."
        )

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

    if any(
        "No passing" in d["Detected Sign"]
        for d in detections
    ):
        instructions.append(
            "Overtaking restriction: Do not overtake."
        )

    if not instructions:

        instructions.append(
            "Follow the detected road signs and continue "
            "with appropriate caution."
        )

    return "\n".join(
        f"• {x}"
        for x in instructions
    )


# ============================================================
# ROAD-SCENE RECOGNITION
# ============================================================

def recognize_road_image(
    input_image,
    detector,
    classifier
):

    image = np.array(input_image)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    original = image.copy()

    height, width = image.shape[:2]

    # YOLO detects traffic-sign locations
    yolo_results = detector.predict(
        source=image,
        conf=0.15,
        verbose=False
    )

    boxes = yolo_results[0].boxes

    detections = []

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

        box_width = x2 - x1
        box_height = y2 - y1

        # 10% crop expansion
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

        # HOG feature extraction
        features = extract_hog_features(crop)

        # SVM classification
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
            "Class ID": predicted_class,
            "Detected Sign": sign_name,
            "Category": category,
            "Priority": priority,
            "Detection Confidence (%)":
                round(
                    detection_confidence * 100,
                    1
                ),
            "Location": location,
            "Driver Assistance": action,
            "Crop": crop,
            "Box": (
                x1,
                y1,
                x2,
                y2
            )
        })

    # Highest priority signs first
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
            f"{d['Detected Sign']} | "
            f"{confidence:.0f}%"
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

        text_width, text_height = text_size

        label_y = max(
            y1 - 8,
            text_height + 10
        )

        cv2.rectangle(
            annotated,
            (
                x1,
                label_y
                - text_height
                - baseline
                - 5
            ),
            (
                x1
                + text_width
                + 8,
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
    # NO DETECTIONS
    # ========================================================

    if not detections:

        summary = """## Road Scene Analysis

### Detection Overview

**Traffic signs detected:** 0

**Attention Level:** NO SIGN DETECTED

No traffic sign was detected above the current detection threshold.

Continue normal visual observation of the road."""

        return (
            annotated_rgb,
            summary,
            pd.DataFrame(),
            None
        )

    # ========================================================
    # SCENE SUMMARY
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

    summary = f"""## Road Scene Analysis

### Detection Overview

**Traffic signs detected:** {total_signs}

**Overall attention level:** {attention_level}

**High priority:** {high_count}

**Medium priority:** {medium_count}

**Low priority:** {low_count}

**Categories present:** {", ".join(categories)}

---

### Highest-Priority Sign

**{highest_priority['Detected Sign']}**

**Category:** {highest_priority['Category']}

**Priority:** {highest_priority['Priority']}

**Location:** {highest_priority['Location']}

**Detection confidence:** {highest_priority['Detection Confidence (%)']}%

---

### Combined Driver Assistance

{scene_instruction}

---

### Processing Pipeline

**1. YOLO11n** — Detects and localizes traffic signs in the road scene.

↓

**2. Sign Cropping** — Each detected sign is isolated from the road image.

↓

**3. HOG Feature Extraction** — Extracts visual shape and edge features.

↓

**4. SVM Classification** — Classifies the cropped sign into one of the 43 GTSRB classes.

↓

**5. Metadata Analysis** — Adds category, priority, location and driving guidance.

↓

**6. Scene-Level Interpretation** — Combines multiple detected signs into a single road-safety summary.
"""

    # ========================================================
    # RESULT TABLE
    # ========================================================

    table_data = [
        {
            k: d[k]
            for k in [
                "Class ID",
                "Detected Sign",
                "Category",
                "Priority",
                "Detection Confidence (%)",
                "Location",
                "Driver Assistance"
            ]
        }
        for d in detections
    ]

    result_table = pd.DataFrame(
        table_data
    )

    # ========================================================
    # DETECTED SIGN CROP SHEET
    # ========================================================

    crop_images = []

    for i, d in enumerate(detections):

        crop = cv2.cvtColor(
            d["Crop"].copy(),
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

        crop = cv2.cvtColor(
            crop,
            cv2.COLOR_RGB2BGR
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
            sign_text = (
                sign_text[:24]
                + "..."
            )

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

        crop_images.append(
            cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            )
        )

    crop_sheet = (
        np.hstack(crop_images)
        if crop_images
        else None
    )

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

    14: {
        "name": "STOP",
        "action": "Vehicle slows down and stops",
        "target": 0,
        "behavior": "stop"
    },

    2: {
        "name": "Speed Limit 50",
        "action": "Vehicle reduces speed to 50 km/h",
        "target": 50,
        "behavior": "slow"
    },

    20: {
        "name": "Dangerous Curve Right",
        "action": "Vehicle slows and follows the right curve",
        "target": 35,
        "behavior": "curve-right"
    },

    13: {
        "name": "YIELD",
        "action": "Vehicle slows and continues carefully",
        "target": 25,
        "behavior": "yield"
    },

    25: {
        "name": "Road Work",
        "action": "Vehicle slows and proceeds carefully",
        "target": 30,
        "behavior": "slow"
    },

    33: {
        "name": "Turn Right Ahead",
        "action": "Vehicle slows and takes the right turn",
        "target": 30,
        "behavior": "turn-right"
    },

    34: {
        "name": "Turn Left Ahead",
        "action": "Vehicle slows and takes the left turn",
        "target": 30,
        "behavior": "turn-left"
    }
}


# ============================================================
# SIMULATOR HTML
# ============================================================

def make_simulator_html():

    scenario_data = json.dumps(
        SCENARIOS
    )

    # Self-contained sign graphics.
    # These are embedded so the Streamlit deployment does not
    # depend on Colab's /content/traffic_signs directory.

    sign_svgs = {

        14:
        """<svg xmlns='http://www.w3.org/2000/svg'
        viewBox='0 0 100 100'>
        <rect width='100' height='100' rx='12' fill='white'/>
        <polygon points='30,8 70,8 92,30 92,70 70,92 30,92 8,70 8,30'
        fill='#d71920'/>
        <polygon points='35,18 65,18 82,35 82,65 65,82 35,82 18,65 18,35'
        fill='white'/>
        <text x='50' y='57' text-anchor='middle'
        font-family='Arial' font-size='17'
        font-weight='bold' fill='#111'>STOP</text>
        </svg>""",

        2:
        """<svg xmlns='http://www.w3.org/2000/svg'
        viewBox='0 0 100 100'>
        <rect width='100' height='100' rx='50' fill='#d71920'/>
        <circle cx='50' cy='50' r='40' fill='white'/>
        <text x='50' y='58' text-anchor='middle'
        font-family='Arial' font-size='30'
        font-weight='bold' fill='#111'>50</text>
        </svg>""",

        20:
        """<svg xmlns='http://www.w3.org/2000/svg'
        viewBox='0 0 100 100'>
        <polygon points='50,5 95,90 5,90'
        fill='#d71920'/>
        <polygon points='50,16 82,80 18,80'
        fill='white'/>
        <path d='M35 67 C42 55,48 55,52 44
        C56 34,61 34,67 25'
        fill='none' stroke='#111'
        stroke-width='7'/>
        </svg>""",

        13:
        """<svg xmlns='http://www.w3.org/2000/svg'
        viewBox='0 0 100 100'>
        <polygon points='50,5 95,92 5,92'
        fill='#d71920'/>
        <polygon points='50,18 79,76 21,76'
        fill='white'/>
        </svg>""",

        25:
        """<svg xmlns='http://www.w3.org/2000/svg'
        viewBox='0 0 100 100'>
        <polygon points='50,5 95,90 5,90'
        fill='#d71920'/>
        <polygon points='50,17 80,78 20,78'
        fill='#f5c542'/>
        <path d='M35 69 L42 53 L49 69
        L56 53 L64 69'
        fill='none' stroke='#111'
        stroke-width='5'/>
        </svg>""",

        33:
        """<svg xmlns='http://www.w3.org/2000/svg'
        viewBox='0 0 100 100'>
        <polygon points='50,5 95,90 5,90'
        fill='#d71920'/>
        <polygon points='50,17 80,78 20,78'
        fill='white'/>
        <path d='M38 68 L55 50 L55 62 L70 62'
        fill='none' stroke='#111'
        stroke-width='7'/>
        </svg>""",

        34:
        """<svg xmlns='http://www.w3.org/2000/svg'
        viewBox='0 0 100 100'>
        <polygon points='50,5 95,90 5,90'
        fill='#d71920'/>
        <polygon points='50,17 80,78 20,78'
        fill='white'/>
        <path d='M62 68 L45 50 L45 62 L30 62'
        fill='none' stroke='#111'
        stroke-width='7'/>
        </svg>"""
    }

    sign_images = {
        class_id:
        "data:image/svg+xml;base64,"
        + base64.b64encode(
            svg.encode()
        ).decode()
        for class_id, svg
        in sign_svgs.items()
    }

    image_data = json.dumps(
        sign_images
    )

    # ========================================================
    # HTML + CSS
    # ========================================================

    html = f'''
<!DOCTYPE html>
<html>
<head>

<style>

body {{
    margin: 0;
    font-family: Arial, sans-serif;
    background: #eef1f5;
    color: #111827;
}}

.dashboard {{
    width: 1100px;
    max-width: 96%;
    margin: 20px auto;
    background: white;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,.12);
}}

.header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}}

.title {{
    font-size: 25px;
    font-weight: bold;
    color: #111827;
}}

.controls {{
    display: flex;
    align-items: center;
    gap: 10px;
}}

select,
button {{
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid #999;
    font-size: 15px;
    color: #111827;
    background: white;
}}

button {{
    cursor: pointer;
    font-weight: bold;
    background: #f3f4f6;
}}

.speed-control {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 15px;
    color: #111827;
    font-weight: bold;
}}

.speed-control input {{
    width: 230px;
}}

#speedValue {{
    min-width: 70px;
}}

.scene {{
    position: relative;
    height: 430px;
    overflow: hidden;
    border-radius: 14px;
    background:
        linear-gradient(
            #9ed8ff 0%,
            #dff3ff 58%,
            #78ad62 58%
        );
}}

.road {{
    position: absolute;
    bottom: 0;
    width: 100%;
    height: 165px;
    background: #404348;
}}

.road-line {{
    position: absolute;
    top: 78px;
    width: 100%;
    border-top: 5px dashed #f5f5f5;
}}

.sign-pole {{
    position: absolute;
    right: 150px;
    bottom: 105px;
    width: 8px;
    height: 145px;
    background: #555;
}}

.sign {{
    position: absolute;
    right: 115px;
    bottom: 230px;
    width: 75px;
    height: 75px;
    background: white;
    border-radius: 10px;
    padding: 5px;
    object-fit: contain;
    filter: contrast(1.4) saturate(1.4);
    box-shadow: 0 4px 12px rgba(0,0,0,.35);
}}

.car {{
    position: absolute;
    left: 40px;
    bottom: 52px;
    width: 120px;
    height: 55px;
    background: #26364a;
    border-radius: 18px 28px 10px 10px;
    box-shadow: 0 7px 10px rgba(0,0,0,.25);
    transition: bottom .7s ease;
}}

.car:before {{
    content: "";
    position: absolute;
    width: 60px;
    height: 30px;
    left: 28px;
    top: -22px;
    background: #344e68;
    border-radius: 20px 20px 5px 5px;
}}

.wheel {{
    position: absolute;
    bottom: -10px;
    width: 25px;
    height: 25px;
    background: #151515;
    border-radius: 50%;
}}

.wheel.left {{
    left: 15px;
}}

.wheel.right {{
    right: 15px;
}}

.info {{
    display: flex;
    gap: 15px;
    margin-top: 15px;
}}

.card {{
    flex: 1;
    padding: 15px;
    border-radius: 12px;
    background: #f1f4f8;
}}

.label {{
    font-size: 13px;
    color: #374151;
}}

.value {{
    margin-top: 5px;
    font-size: 18px;
    font-weight: bold;
    color: #111827;
}}

.status {{
    margin-top: 15px;
    padding: 14px;
    border-radius: 10px;
    text-align: center;
    font-size: 17px;
    font-weight: bold;
    color: #111827;
    background: #e5edff;
}}

</style>

</head>

<body>

<div class="dashboard">

<div class="header">

<div class="title">
Traffic Sign Recognition & Driver Assistance
</div>

<div class="controls">

<select id="scenario">

<option value="14">
STOP
</option>

<option value="2">
Speed Limit 50
</option>

<option value="20">
Dangerous Curve Right
</option>

<option value="13">
YIELD
</option>

<option value="25">
Road Work
</option>

<option value="33">
Turn Right Ahead
</option>

<option value="34">
Turn Left Ahead
</option>

</select>

<button id="startButton">
Start Simulation
</button>

</div>

</div>


<div class="speed-control">

<span>
Initial Speed:
</span>

<input
type="range"
id="initialSpeed"
min="20"
max="100"
value="65"
step="5"
>

<span id="speedValue">
65 km/h
</span>

</div>


<div class="scene">

<div class="road">
<div class="road-line"></div>
</div>

<div class="sign-pole"></div>

<img
id="sign"
class="sign"
>

<div
id="car"
class="car"
>

<div class="wheel left"></div>
<div class="wheel right"></div>

</div>

</div>


<div class="info">

<div class="card">

<div class="label">
Detected Traffic Sign
</div>

<div
id="signName"
class="value"
>
STOP
</div>

</div>


<div class="card">

<div class="label">
Current Vehicle Speed
</div>

<div
id="speed"
class="value"
>
65 km/h
</div>

</div>


<div class="card">

<div class="label">
Driver Assistance
</div>

<div
id="action"
class="value"
>
Vehicle slows down and stops
</div>

</div>

</div>


<div
id="status"
class="status"
>
Select a scenario, choose the starting speed and start the simulation.
</div>

</div>


<script>

const scenarios = {scenario_data};

const images = {image_data};


const scenarioSelect =
    document.getElementById('scenario');

const speedSlider =
    document.getElementById('initialSpeed');

const speedValue =
    document.getElementById('speedValue');

const sign =
    document.getElementById('sign');

const signName =
    document.getElementById('signName');

const speedDisplay =
    document.getElementById('speed');

const action =
    document.getElementById('action');

const status =
    document.getElementById('status');

const car =
    document.getElementById('car');

const startButton =
    document.getElementById('startButton');


let animationFrame = null;


speedSlider.addEventListener(
    'input',
    () => {{
        speedValue.innerText =
            speedSlider.value + ' km/h';
    }}
);


function updateScenario() {{

    const id =
        scenarioSelect.value;

    const data =
        scenarios[id];

    sign.src =
        images[id];

    signName.innerText =
        data.name;

    action.innerText =
        data.action;

    speedDisplay.innerText =
        speedSlider.value + ' km/h';

    status.innerText =
        'Select a scenario, choose the starting speed and start the simulation.';

    car.style.left =
        '40px';

    car.style.bottom =
        '52px';
}}


scenarioSelect.addEventListener(
    'change',
    updateScenario
);


function startSimulation() {{

    cancelAnimationFrame(
        animationFrame
    );

    const id =
        scenarioSelect.value;

    const data =
        scenarios[id];

    const initialSpeed =
        Number(speedSlider.value);

    const targetSpeed =
        data.target;

    let currentSpeed =
        initialSpeed;

    let position =
        40;

    let lastTime =
        null;

    let reachedSign =
        false;


    car.style.left =
        '40px';

    car.style.bottom =
        '52px';

    speedDisplay.innerText =
        initialSpeed + ' km/h';

    status.innerText =
        'Vehicle approaching the traffic sign...';


    function animate(timestamp) {{

        if (!lastTime)
            lastTime = timestamp;

        const elapsed =
            (timestamp - lastTime) / 1000;

        lastTime =
            timestamp;


        if (
            position >= 540
            &&
            !reachedSign
        ) {{

            reachedSign = true;

            status.innerText =
                data.name
                + ' detected — '
                + data.action;
        }}


        if (reachedSign) {{

            if (
                currentSpeed > targetSpeed
            ) {{

                currentSpeed -=
                    25 * elapsed;

                if (
                    currentSpeed < targetSpeed
                )
                    currentSpeed =
                        targetSpeed;

            }}

            else if (
                currentSpeed < targetSpeed
            ) {{

                currentSpeed +=
                    20 * elapsed;

                if (
                    currentSpeed > targetSpeed
                )
                    currentSpeed =
                        targetSpeed;
            }}
        }}


        speedDisplay.innerText =
            Math.round(currentSpeed)
            + ' km/h';


        position +=
            currentSpeed
            * elapsed
            * 3.0;


        car.style.left =
            position + 'px';


        if (
            data.behavior === 'stop'
            &&
            position >= 700
        ) {{

            speedDisplay.innerText =
                '0 km/h';

            car.style.left =
                '700px';

            status.innerText =
                'STOP detected — Vehicle has stopped.';

            return;
        }}


        if (
            data.behavior === 'curve-right'
            &&
            position >= 570
        ) {{

            car.style.bottom =
                '95px';

            status.innerText =
                'Dangerous curve detected — Vehicle slowing and following the right curve.';
        }}


        if (
            data.behavior === 'turn-right'
            &&
            position >= 570
        ) {{

            car.style.bottom =
                '125px';

            status.innerText =
                'Right turn ahead — Vehicle slowing and taking the right turn.';
        }}


        if (
            data.behavior === 'turn-left'
            &&
            position >= 570
        ) {{

            car.style.bottom =
                '125px';

            status.innerText =
                'Left turn ahead — Vehicle slowing and taking the left turn.';
        }}


        if (
            data.behavior !== 'stop'
            &&
            position > 1150
        ) {{

            status.innerText =
                data.name
                + ' handled — Vehicle continues driving.';

            return;
        }}


        animationFrame =
            requestAnimationFrame(
                animate
            );
    }}


    animationFrame =
        requestAnimationFrame(
            animate
        );
}}


startButton.addEventListener(
    'click',
    startSimulation
);


updateScenario();

</script>

</body>
</html>
'''

    return html


# ============================================================
# MAIN APP UI
# ============================================================

st.title(
    "Traffic Sign Recognition & Driver Assistance System"
)

st.caption(
    "YOLO11n + HOG + SVM + Metadata + Scene Intelligence"
)


# ============================================================
# LOAD MODELS
# ============================================================

try:

    detector, classifier = load_models()

except Exception as e:

    st.error(
        "Model loading failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# TABS
# ============================================================

road_tab, sim_tab = st.tabs(
    [
        "Road Scene Analysis",
        "Driving Assistance Simulation"
    ]
)


# ============================================================
# ROAD SCENE ANALYSIS
# ============================================================

with road_tab:

    st.markdown(
        "### Traffic Sign Recognition & Road Scene Analysis"
    )

    st.write(
        "Upload a road-scene image containing one or more "
        "traffic signs. The system detects signs, classifies "
        "them, analyzes their importance and provides "
        "scene-level driver assistance."
    )

    uploaded = st.file_uploader(
        "Upload Road Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="road_image"
    )

    if uploaded is not None:

        image = Image.open(
            uploaded
        ).convert("RGB")

        if st.button(
            "Analyze Road Scene",
            type="primary"
        ):

            with st.spinner(
                "Analyzing road scene..."
            ):

                annotated, summary, table, crops = (
                    recognize_road_image(
                        image,
                        detector,
                        classifier
                    )
                )

            st.image(
                annotated,
                caption="Detected Traffic Signs",
                use_container_width=True
            )

            st.markdown(
                "## Road Scene Intelligence"
            )

            st.markdown(
                summary
            )

            st.markdown(
                "## Detected Sign Details"
            )

            if table.empty:

                st.info(
                    "No detected signs to display."
                )

            else:

                st.dataframe(
                    table,
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown(
                "## Individual Sign Analysis"
            )

            if crops is not None:

                st.image(
                    crops,
                    caption="Detected Sign Crops",
                    use_container_width=True
                )


# ============================================================
# DRIVING ASSISTANCE SIMULATION
# ============================================================

with sim_tab:

    st.markdown(
        "### Traffic Sign Recognition & Driver Assistance Simulation"
    )

    st.write(
        "Multiple scenarios with adjustable starting speed, "
        "using the same simulation behaviour as the original notebook."
    )

    components.html(
        make_simulator_html(),
        height=720,
        scrolling=False
    )
