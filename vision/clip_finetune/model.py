import torch.nn as nn
from transformers import CLIPVisionModel


class ClipClassifier(nn.Module):
    def __init__(self, num_classes, checkpoint="openai/clip-vit-base-patch32"):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(checkpoint)
        hidden_size = self.vision_encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        # pooler_output is CLIP's summary embedding for the whole image.
        outputs = self.vision_encoder(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        return self.classifier(pooled)
