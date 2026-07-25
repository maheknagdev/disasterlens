import os
import torch
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor

from dataset import AiderDataset
from model import ClipClassifier
from severity_mapping import CLASS_NAMES

DATA_ROOT = "../../data/aider"
OUTPUT_PATH = "../../outputs/models/clip_classifier.pt"
BATCH_SIZE = 32
EPOCHS = 3
LR = 1e-3

# MPS = Apple Silicon GPU acceleration, falls back to CPU if unavailable.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load data using CLIP's own preprocessor so images match its pretraining distribution.
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
train_ds = AiderDataset(os.path.join(DATA_ROOT, "Train"), processor)
val_ds = AiderDataset(os.path.join(DATA_ROOT, "Val"), processor)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Freeze CLIP's vision encoder (linear probing) — only the classifier head trains.
model = ClipClassifier(num_classes=len(CLASS_NAMES)).to(device)
for param in model.vision_encoder.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()


def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for pixel_values, labels in val_loader:
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            preds = model(pixel_values).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for pixel_values, labels in train_loader:
        pixel_values, labels = pixel_values.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(pixel_values), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    val_acc = evaluate()
    print(f"epoch {epoch+1}/{EPOCHS} - loss: {running_loss/len(train_loader):.4f} - val_acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        torch.save(model.state_dict(), OUTPUT_PATH)

print(f"best val accuracy: {best_acc:.4f} - saved to {OUTPUT_PATH}")
