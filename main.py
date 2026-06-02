import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

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

disease_model = models.resnet18(weights=None)
disease_model.fc = nn.Linear(disease_model.fc.in_features, 15)
disease_model.load_state_dict(
    torch.load("best_disease_model.pth", map_location=device, weights_only=True)
)
disease_model.to(device)
disease_model.eval()

severity_model = models.mobilenet_v3_small(weights=None)
severity_model.classifier[3] = nn.Linear(
    severity_model.classifier[3].in_features, 2
)
severity_model.load_state_dict(
    torch.load("best_severity_model.pth", map_location=device, weights_only=True)
)
severity_model.to(device)
severity_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_path):
    image  = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        disease_probs = torch.softmax(disease_model(tensor), dim=1)[0]
        disease_pred  = torch.argmax(disease_probs).item()
        disease_conf  = round(float(disease_probs[disease_pred]) * 100, 1)

    disease_name = disease_classes[disease_pred]

    if disease_name == "Healthy":
        return {
            "Disease":            "Healthy",
            "Disease Confidence": disease_conf,
            "Severity":           "None",
            "Severity Confidence": None
        }

    with torch.no_grad():
        severity_probs = torch.softmax(severity_model(tensor), dim=1)[0]
        severity_pred  = torch.argmax(severity_probs).item()
        severity_conf  = round(float(severity_probs[severity_pred]) * 100, 1)

    return {
        "Disease":             disease_name,
        "Disease Confidence":  disease_conf,
        "Severity":            severity_classes[severity_pred],
        "Severity Confidence": severity_conf
    }

if __name__ == "__main__":
    result = predict("test.jpg")
    print("\n===== RESULT =====")
    print(f"Disease             : {result['Disease']}")
    print(f"Disease Confidence  : {result['Disease Confidence']}%")
    print(f"Severity            : {result['Severity']}")
    if result['Severity Confidence']:
        print(f"Severity Confidence : {result['Severity Confidence']}%")
