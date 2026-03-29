import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from sklearn.metrics import accuracy_score
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# TRANSFORMS
# ----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
val_dataset = datasets.ImageFolder("dataset/val", transform=transform)
test_dataset = datasets.ImageFolder("dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)
test_loader = DataLoader(test_dataset, batch_size=4)

num_classes = 2

print("Class Mapping:", train_dataset.class_to_idx)

# ----------------------------
# EVALUATION FUNCTION
# ----------------------------

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)


# ----------------------------
# CNN TRAINING
# ----------------------------

def train_cnn():

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_val_acc = 0

    for epoch in range(10):

        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc = evaluate(model, val_loader)

        print(f"[CNN] Epoch {epoch+1}, Loss: {total_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "cnn_model.pth")

    print("Best CNN Validation Accuracy:", best_val_acc)

    test_acc = evaluate(model, test_loader)
    print("CNN Test Accuracy:", test_acc)


# ----------------------------
# ViT TRAINING
# ----------------------------

def train_vit():

    model = ViTForImageClassification.from_pretrained(
        "google/vit-small-patch16-224",
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    best_val_acc = 0

    for epoch in range(5):

        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # evaluation
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(pixel_values=images)
                preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)

        print(f"[ViT] Epoch {epoch+1}, Loss: {total_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "vit_model.pth")

    print("Best ViT Validation Accuracy:", best_val_acc)


if __name__ == "__main__":
    #train_cnn()
    train_vit()