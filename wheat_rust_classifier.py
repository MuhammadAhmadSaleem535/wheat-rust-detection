import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -----------------------------
# 1. Config
# -----------------------------
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2. Strong Transforms
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 3. Dataset
# -----------------------------
train_dataset = datasets.ImageFolder(
    "dataset_wheat/data/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    "dataset_wheat/data/valid",
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Class Mapping:")
print(train_dataset.class_to_idx)

# -----------------------------
# 4. Model (Fine-tuned ResNet18)
# -----------------------------
model = models.resnet18(pretrained=True)

# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last block (VERY IMPORTANT)
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(device)

# -----------------------------
# 5. Loss Function (Label Smoothing)
# -----------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# -----------------------------
# 6. Optimizer (Differential LR)
# -----------------------------
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)

# -----------------------------
# 7. Scheduler
# -----------------------------
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# -----------------------------
# 8. Training Loop
# -----------------------------
best_val_acc = 0

for epoch in range(EPOCHS):

    # -------------------------
    # TRAINING
    # -------------------------
    model.train()

    total_loss = 0
    correct = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

    train_loss = total_loss / len(train_dataset)
    train_acc = correct / len(train_dataset)

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()

    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()

    val_acc = val_correct / len(val_dataset)

    # -------------------------
    # Scheduler step
    # -------------------------
    scheduler.step()

    # -------------------------
    # Save best model
    # -------------------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_disease_model.pth")

    # -------------------------
    # Print stats
    # -------------------------
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"Loss      : {train_loss:.4f}")
    print(f"Train Acc : {train_acc:.4f}")
    print(f"Val Acc   : {val_acc:.4f}")
    print("-" * 30)

# -----------------------------
# 9. Done
# -----------------------------
print("\nTraining Complete!")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print("Model saved as best_disease_model.pth")
