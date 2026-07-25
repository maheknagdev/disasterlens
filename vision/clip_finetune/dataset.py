import os
from PIL import Image
from torch.utils.data import Dataset

from severity_mapping import CLASS_NAMES


class AiderDataset(Dataset):
    # Expects root/<ClassName>/*.png|jpg, matching the AIDERv2 folder layout.
    def __init__(self, root_dir, processor):
        self.processor = processor
        self.samples = []
        for label_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return pixel_values, label
