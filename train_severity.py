import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


def main():
    # -----------------------------
    # 1. CONFIG
    # -----------------------------
    BATCH_SIZE  = 32
    EPOCHS      = 20
    NUM_CLASSES = 2  # moderate, severe

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # 2. VERIFY PATHS
    # -----------------------------
    train_path = "dataset_wheat/data/severity/train"
    val_path   = "dataset_wheat/data/severity/val"

    print("\n--- PATH VERIFICATION ---")
    print(f"Train absolute path : {os.path.abspath(train_path)}")
    print(f"Val   absolute path : {os.path.abspath(val_path)}")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train path not found: {os.path.abspath(train_path)}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Val path not found: {os.path.abspath(val_path)}")

    print(f"Train folder contents : {os.listdir(train_path)}")
    print(f"Val   folder contents : {os.listdir(val_path)}")
    print("-------------------------\n")

    # -----------------------------
    # 3. TRANSFORMS
    # -----------------------------
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # -----------------------------
    # 4. DATASETS
    # -----------------------------
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_path,   transform=val_transform)

    print("Class Mapping:", train_dataset.class_to_idx)

    expected_classes = {"moderate", "severe"}
    found_classes    = set(train_dataset.class_to_idx.keys())
    if found_classes != expected_classes:
        raise ValueError(
            f"Wrong classes detected: {found_classes}. "
            f"Expected {expected_classes}."
        )

    # num_workers=0 and pin_memory=False  →  safe on Windows / CPU
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)

    # -----------------------------
    # 5. CLASS WEIGHTS
    # -----------------------------
    class_counts  = torch.bincount(torch.tensor(train_dataset.targets))
    print("Class counts:", class_counts)

    class_weights = class_counts.sum() / (class_counts.float() + 1e-6)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    print("Class weights:", class_weights)

    # -----------------------------
    # 6. MODEL
    # -----------------------------
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")

    for param in model.features[:-4].parameters():
        param.requires_grad = False
    for param in model.features[-4:].parameters():
        param.requires_grad = True

    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    model = model.to(device)

    # -----------------------------
    # 7. LOSS
    # -----------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # -----------------------------
    # 8. OPTIMIZER
    # -----------------------------
    optimizer = optim.AdamW([
        {'params': model.features[-4:].parameters(), 'lr': 1e-4},
        {'params': model.classifier[3].parameters(), 'lr': 3e-4}
    ], weight_decay=1e-4)

    # -----------------------------
    # 9. SCHEDULER
    # -----------------------------
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # -----------------------------
    # 10. TRAINING LOOP
    # -----------------------------
    best_val_acc = 0.0

    for epoch in range(EPOCHS):

        # --- TRAIN ---
        model.train()
        train_loss    = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / len(train_dataset)

        # --- VALIDATION ---
        model.eval()
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs       = model(images)
                val_correct  += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_dataset)
        scheduler.step()

        # --- SAVE BEST ---
        saved_tag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_severity_model.pth")
            saved_tag = "  <-- saved"

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}]  "
              f"Loss: {train_loss:.4f}  "
              f"Train Acc: {train_acc:.4f}  "
              f"Val Acc: {val_acc:.4f}{saved_tag}")

    print("\nTraining Complete!")
    print(f"Best Validation Accuracy : {best_val_acc:.4f}")
    print("Saved model              : best_severity_model.pth")


# ↓↓ THIS LINE IS CRITICAL ON WINDOWS — do not remove ↓↓
if __name__ == '__main__':
    main()