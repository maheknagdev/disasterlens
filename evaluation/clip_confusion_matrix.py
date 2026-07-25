import sys
import os
import torch
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("../vision/clip_finetune")
from dataset import AiderDataset
from model import ClipClassifier
from severity_mapping import CLASS_NAMES

DATA_ROOT = "../data/aider"
CHECKPOINT_PATH = "../outputs/models/clip_classifier.pt"
OUTPUT_FIG = "../outputs/models/clip_confusion_matrix.png"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the trained classifier head + frozen CLIP backbone from the training run.
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = ClipClassifier(num_classes=len(CLASS_NAMES)).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

test_ds = AiderDataset(os.path.join(DATA_ROOT, "Test"), processor)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

# Run inference over the held-out test set to get predictions vs. true labels.
all_preds, all_labels = [], []
with torch.no_grad():
    for pixel_values, labels in test_loader:
        pixel_values = pixel_values.to(device)
        preds = model(pixel_values).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("CLIP Classifier — Test Set Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_FIG)
print(f"saved confusion matrix to {OUTPUT_FIG}")
