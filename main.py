import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# 1. Load Models
# -----------------------------

device = torch.device("cpu")

# Rust classification model
rust_model = models.resnet18(pretrained=False)
rust_model.fc = nn.Linear(rust_model.fc.in_features, 4)
rust_model.load_state_dict(torch.load("rust_model.pth", map_location=device))
rust_model.eval()

# Severity model
severity_model = models.resnet18(pretrained=False)
severity_model.fc = nn.Linear(severity_model.fc.in_features, 2)
severity_model.load_state_dict(torch.load("severity_model.pth", map_location=device))
severity_model.eval()

# -----------------------------
# 2. Class Labels
# -----------------------------

rust_classes = ["black_rust", "brown_rust", "healthy", "yellow_rust"]
severity_classes = ["moderate", "severe"]

# -----------------------------
# 3. Preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# 4. Recommendation System
# -----------------------------

def get_recommendation(rust, severity):
    recommendations = {
        "black_rust": {
            "moderate": {
                "action": "Apply systemic fungicide (triazole-based) at recommended dose.",
                "monitoring": "Inspect field every 3–5 days for spread.",
                "notes": "Ensure proper field sanitation and avoid excess irrigation.",
                "urgency": "medium"
            },
            "severe": {
                "action": "Immediate fungicide application required. Repeat spray cycle if necessary.",
                "monitoring": "Monitor neighboring plants closely and isolate heavily infected areas.",
                "notes": "Severe stem rust can significantly reduce yield.",
                "urgency": "high"
            }
        },
        "brown_rust": {
            "moderate": {
                "action": "Apply preventive fungicide and maintain crop health.",
                "monitoring": "Regularly check leaves for increase in infection.",
                "notes": "Common disease; manage before it spreads widely.",
                "urgency": "medium"
            },
            "severe": {
                "action": "Apply systemic fungicide across affected area.",
                "monitoring": "Reassess after 5–7 days and consider reapplication.",
                "notes": "Heavy infection can reduce photosynthesis.",
                "urgency": "high"
            }
        },
        "yellow_rust": {
            "moderate": {
                "action": "Apply fungicide early to prevent rapid spread.",
                "monitoring": "Check for stripe formation on leaves.",
                "notes": "Spreads quickly in cool and moist conditions.",
                "urgency": "medium"
            },
            "severe": {
                "action": "Urgent fungicide application across the field.",
                "monitoring": "Frequent monitoring required due to rapid spread.",
                "notes": "Can spread very quickly under favorable conditions.",
                "urgency": "high"
            }
        },
        "healthy": {
            "moderate": {
                "action": "No treatment required.",
                "monitoring": "Continue regular field inspection.",
                "notes": "Maintain good agricultural practices.",
                "urgency": "low"
            },
            "severe": {
                "action": "No treatment required.",
                "monitoring": "Continue regular field inspection.",
                "notes": "Plant appears healthy.",
                "urgency": "low"
            }
        }
    }

    return recommendations[rust][severity]
    

# -----------------------------
# 5. Prediction Pipeline
# -----------------------------

def predict(image_path):
    # Load image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    # Rust prediction
    with torch.no_grad():
        rust_output = rust_model(image)
        _, rust_pred = torch.max(rust_output, 1)
        rust = rust_classes[rust_pred.item()]

    # Severity prediction
    with torch.no_grad():
        severity_output = severity_model(image)
        _, severity_pred = torch.max(severity_output, 1)
        severity = severity_classes[severity_pred.item()]

    # Recommendation
    recommendation = get_recommendation(rust, severity)

    return rust, severity, recommendation



print("Disease:", rust)
print("Severity:", severity)
print("Recommendation:", recommendation)
