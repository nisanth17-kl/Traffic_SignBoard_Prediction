from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "traffic_sign_model.h5"
model = None

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully.")
else:
    print(
        f"[WARNING] Model file '{MODEL_PATH}' not found. "
        "Place it in the project root."
    )


# ============================================================
# GTSRB 43 CLASS LABELS
# ============================================================

CLASSES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}


# ============================================================
# SUGGESTED ACTION FOR EACH TRAFFIC SIGN
# ============================================================

SOLUTIONS = {

    0: "Reduce your speed to 20 km/h or below.",
    1: "Reduce your speed to 30 km/h or below.",
    2: "Reduce your speed to 50 km/h or below.",
    3: "Reduce your speed to 60 km/h or below.",
    4: "Reduce your speed to 70 km/h or below.",
    5: "Reduce your speed to 80 km/h or below.",

    6: "The 80 km/h restriction has ended. Follow the next applicable speed limit.",

    7: "Reduce your speed to 100 km/h or below.",
    8: "The maximum permitted speed is 120 km/h. Drive according to road conditions.",

    9: "Do not overtake other vehicles.",
    10: "Do not overtake restricted heavy vehicles.",

    11: "Give priority according to the intersection traffic rules.",
    12: "You are travelling on a priority road. Continue carefully.",
    13: "Slow down and give way to other vehicles.",

    14: "Stop the vehicle completely and proceed only when it is safe.",

    15: "Vehicles are not allowed on this road.",
    16: "Vehicles over 3.5 metric tons are prohibited.",

    17: "Do not enter this road. Find an alternate route.",

    18: "Be alert. A hazard may be present ahead and you should reduce speed.",

    19: "Dangerous curve to the left ahead. Slow down and steer carefully.",
    20: "Dangerous curve to the right ahead. Slow down and steer carefully.",
    21: "Multiple curves ahead. Reduce speed and drive carefully.",

    22: "Bumpy road ahead. Reduce speed and maintain vehicle control.",
    23: "Road may be slippery. Reduce speed and avoid sudden braking.",

    24: "Road narrows on the right. Slow down and maintain a safe distance.",

    25: "Road work ahead. Slow down and watch for workers and diversions.",

    26: "Traffic signals ahead. Reduce speed and follow the signal.",

    27: "Pedestrians may be crossing. Slow down and give them priority.",

    28: "Children may be crossing. Reduce speed and drive with extra caution.",

    29: "Bicycles may be crossing. Slow down and watch carefully.",

    30: "Ice or snow may be present. Reduce speed and avoid sudden movements.",

    31: "Wild animals may cross the road. Slow down and stay alert.",

    32: "Previous speed and passing restrictions have ended. Follow current signs.",

    33: "Turn right ahead. Reduce speed and prepare for the turn.",
    34: "Turn left ahead. Reduce speed and prepare for the turn.",

    35: "Continue straight ahead.",

    36: "You may continue straight or turn right.",
    37: "You may continue straight or turn left.",

    38: "Keep to the right side of the road.",
    39: "Keep to the left side of the road.",

    40: "Roundabout ahead. Follow the indicated direction.",

    41: "The no-passing restriction has ended. Overtake only when safe and legal.",

    42: "The restriction for passing heavy vehicles has ended."
}


# ============================================================
# RISK LEVEL
# ============================================================

RISK_LEVELS = {

    # HIGH RISK
    13: "HIGH",
    14: "HIGH",
    17: "HIGH",
    18: "HIGH",
    19: "HIGH",
    20: "HIGH",
    21: "HIGH",
    23: "HIGH",
    25: "HIGH",
    26: "HIGH",
    27: "HIGH",
    28: "HIGH",
    30: "HIGH",
    31: "HIGH",

    # MEDIUM RISK
    0: "MEDIUM",
    1: "MEDIUM",
    2: "MEDIUM",
    3: "MEDIUM",
    4: "MEDIUM",
    5: "MEDIUM",
    6: "MEDIUM",
    7: "MEDIUM",
    8: "MEDIUM",
    9: "MEDIUM",
    10: "MEDIUM",
    11: "MEDIUM",
    12: "MEDIUM",
    15: "MEDIUM",
    16: "MEDIUM",
    22: "MEDIUM",
    24: "MEDIUM",
    29: "MEDIUM",
    32: "MEDIUM",
    33: "MEDIUM",
    34: "MEDIUM",
    35: "MEDIUM",
    36: "MEDIUM",
    37: "MEDIUM",
    38: "MEDIUM",
    39: "MEDIUM",
    40: "MEDIUM",
    41: "MEDIUM",
    42: "MEDIUM"
}


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(img_array):
    """
    Resize and normalize image
    according to model input requirements.
    """

    img = cv2.resize(img_array, (64, 64))

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    return img


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# PREDICT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "error": "Model not loaded. Place traffic_sign_model.h5 in the project root."
        }), 500

    try:

        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({
                "error": "No image data received."
            }), 400

        # ------------------------------------------------
        # DECODE BASE64 IMAGE
        # ------------------------------------------------

        image_data = data["image"]

        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)

        img_array = np.frombuffer(
            img_bytes,
            dtype=np.uint8
        )

        img = cv2.imdecode(
            img_array,
            cv2.IMREAD_COLOR
        )

        if img is None:
            return jsonify({
                "error": "Could not decode image."
            }), 400

        # ------------------------------------------------
        # PREPROCESS
        # ------------------------------------------------

        processed = preprocess_image(img)

        # ------------------------------------------------
        # MODEL PREDICTION
        # ------------------------------------------------

        predictions = model.predict(
            processed,
            verbose=0
        )[0]

        # ------------------------------------------------
        # TOP 3 PREDICTIONS
        # ------------------------------------------------

        top3_indices = predictions.argsort()[-3:][::-1].tolist()

        top3 = []

        for idx in top3_indices:

            confidence = float(predictions[idx]) * 100

            top3.append({
                "class_id": int(idx),
                "label": CLASSES[int(idx)],
                "confidence": round(confidence, 2)
            })

        # ------------------------------------------------
        # MAIN DETECTION
        # ------------------------------------------------

        detected_class = top3[0]["class_id"]

        detected_label = top3[0]["label"]

        confidence = top3[0]["confidence"]

        solution = SOLUTIONS.get(
            detected_class,
            "Follow the traffic rules and drive carefully."
        )

        risk = RISK_LEVELS.get(
            detected_class,
            "MEDIUM"
        )

        # ------------------------------------------------
        # RETURN RESULT
        # ------------------------------------------------

        return jsonify({

            "class_id": detected_class,

            "label": detected_label,

            "confidence": confidence,

            "risk_level": risk,

            "solution": solution,

            "top3": top3
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "ok",

        "model_loaded": model is not None

    })


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
