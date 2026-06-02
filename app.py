from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

disease_classes = [
    "Aphid",
    "Black Rust",
    "Blast",
    "Brown Rust",
    "Common Root Rot",
    "Fusarium Head Blight",
    "Healthy",
    "Leaf Blight",
    "Mildew",
    "Mite",
    "Septoria",
    "Smut",
    "Stem fly",
    "Tan spot",
    "Yellow Rust"
]

severity_classes = ["Moderate", "Severe"]

# Confidence threshold — if top disease prediction is below this,
# treat as Healthy. Tune between 50-70 based on your testing.
CONFIDENCE_THRESHOLD = 50.0

def load_disease_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 15)
    model.load_state_dict(
        torch.load("best_disease_model.pth", map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model

def load_severity_model():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(
        torch.load("best_severity_model.pth", map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model

disease_model  = load_disease_model()
severity_model = load_severity_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_bytes):
    image  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        disease_probs = torch.softmax(disease_model(tensor), dim=1)[0]
        disease_pred  = torch.argmax(disease_probs).item()
        disease_conf  = round(float(disease_probs[disease_pred]) * 100, 1)

    disease_name = disease_classes[disease_pred]

    # If confidence is below threshold and it's not already Healthy,
    # the model is unsure — default to Healthy
    if disease_name != "Healthy" and disease_conf < CONFIDENCE_THRESHOLD:
        disease_name = "Healthy"

    all_scores = {
        cls: round(float(p) * 100, 1)
        for cls, p in zip(disease_classes, disease_probs)
    }

    if disease_name == "Healthy":
        return {
            "disease":             "Healthy",
            "disease_confidence":  disease_conf,
            "severity":            "None",
            "severity_confidence": None,
            "all_disease_scores":  all_scores
        }

    with torch.no_grad():
        severity_probs = torch.softmax(severity_model(tensor), dim=1)[0]
        severity_pred  = torch.argmax(severity_probs).item()
        severity_conf  = round(float(severity_probs[severity_pred]) * 100, 1)

    return {
        "disease":             disease_name,
        "disease_confidence":  disease_conf,
        "severity":            severity_classes[severity_pred],
        "severity_confidence": severity_conf,
        "all_disease_scores":  all_scores
    }

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def run_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        result = predict(file.read())
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(f"Running on http://localhost:5000  |  device: {device}")
    app.run(debug=True, port=5000)