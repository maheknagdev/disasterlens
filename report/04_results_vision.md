# Results: Vision

On 647 held-out AIDER images, the fine-tuned CLIP head achieved **98.15% test
accuracy**. Macro precision, recall, and F1 were 0.9601, 0.9781, and 0.9689,
respectively. The best validation accuracy was 98.29%.

| Severity | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| none | 0.9931 | 0.9818 | 0.9874 | 439 |
| mild | 0.9216 | 0.9592 | 0.9400 | 49 |
| moderate | 0.9815 | 0.9907 | 0.9860 | 107 |
| severe | 0.9444 | 0.9808 | 0.9623 | 52 |

The confusion matrix, ordered `none`, `mild`, `moderate`, `severe`, is:

| Actual \ Predicted | none | mild | moderate | severe |
|---|---:|---:|---:|---:|
| none | 431 | 4 | 2 | 2 |
| mild | 1 | 47 | 0 | 1 |
| moderate | 1 | 0 | 106 | 0 |
| severe | 1 | 0 | 0 | 51 |

The saved visualization is
`disasterlens_modules/models/clip_vision/confusion_matrix.png`. Errors were sparse:
the model correctly classified 431 of 439 `none` images, 47 of 49 `mild` images,
106 of 107 `moderate` images, and 51 of 52 `severe` images. The `mild` class had the
lowest F1 (0.9400), largely because four normal scenes were assigned to that class.

These results measure discrimination under the processed AIDER label mapping. They
do not establish robustness to out-of-domain social-media imagery, calibration
under distribution shift, or contradictory textual evidence. The separate conflict
fixture addresses the final condition at the fusion-system level.
