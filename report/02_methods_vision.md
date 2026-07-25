# Methods: Vision Module

The vision module predicts one of four ordered severity labels—`none`, `mild`,
`moderate`, or `severe`—from a single image. The implementation uses the pretrained
`openai/clip-vit-base-patch32` image encoder as a frozen feature extractor. Each
image is transformed by the associated CLIP processor, embedded as a
512-dimensional normalized feature vector, and passed to a trainable classification
head.

The classification head comprises a 512-to-128 linear projection, a rectified
linear activation, dropout with probability 0.3, and a 128-to-4 output projection.
Training ran for 15 epochs on the stratified AIDER training split with seed 42.
Validation accuracy determined checkpoint selection, whereas the held-out test split
was reserved for final evaluation.

The AIDER remapping prioritizes a stable severity contract over preservation of
event names. Normal scenes represent `none`; minor
traffic accidents represent `mild`; fires and partial flooding represent
`moderate`; and collapsed buildings and severe flooding represent `severe`.

During inference, `predict_severity(image_path)` loads the frozen encoder and
fine-tuned head, computes class probabilities with softmax, and returns the selected
severity and its probability. The response also includes an empty damage-type list,
which reserves the field for future multilabel damage recognition, and a model
identifier. This interface decouples the downstream fusion layer from the AIDER
training taxonomy.
