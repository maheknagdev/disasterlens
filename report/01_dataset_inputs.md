# Data and Inputs

## Data sources

The vision analysis used AIDER, an image dataset containing normal scenes and
disaster events. The processed sample comprised 6,433 images. Original event labels
were remapped to the four-level DisasterLens severity scale: normal scenes to
`none`, minor traffic accidents to `mild`, fires and partially flooded scenes to
`moderate`, and collapsed buildings or severe flooding to `severe`. This mapping
converted heterogeneous event labels into a common output scale for multimodal
fusion.

The language analysis used CrisisMMD, a multimodal collection of disaster-related
tweets and images. Direct inspection of the downloaded humanitarian annotations found
18,082 image–text pairs, 16,058 unique tweets, 18,082 unique images, and eight
original humanitarian categories. The original fine-grained categories were
retained because distinctions among affected individuals, infrastructure damage,
injuries, missing persons, and rescue activity are relevant to urgency assessment.
After reserving the conflict-evaluation IDs, 15,637 records remained for model
development.

## Preprocessing and splits

Both datasets were divided using deterministic, stratified 80/10/10
training/validation/test splits with seed 42. AIDER contains 5,144 training, 642
validation, and 647 test images. Its complete severity distribution is 4,390
`none`, 485 `mild`, 1,047 `moderate`, and 511 `severe`.

CrisisMMD contains 12,506 training, 1,561 validation, and 1,570 test records after
conflict-fixture exclusion. The retained fine-category counts are:

| Fine category | Records |
|---|---:|
| affected individuals | 465 |
| infrastructure and utility damage | 937 |
| injured or dead people | 481 |
| missing or found people | 40 |
| not humanitarian | 4,531 |
| other relevant information | 5,872 |
| rescue, volunteering, or donation effort | 3,257 |
| vehicle damage | 54 |

The contract-facing resource vocabulary is intentionally narrower than the training
taxonomy. `affected_individuals` and `injured_or_dead_people` map to medical;
`infrastructure_and_utility_damage` maps to shelter; and
`rescue_volunteering_or_donation_effort` maps to food and water. Categories without
a direct resource interpretation produce an empty list. This conservative mapping
avoids inferring material needs from labels that do not encode them.

## Image–text conflict fixture

To evaluate disagreement between modalities, a separate 500-row fusion fixture was
constructed and excluded from every model-development split. It contains 150 real
severity mismatches, 150 real resource mismatches, 100 real agreement controls, and
100 synthetic location mismatches. Four hundred rows therefore preserve naturally
paired CrisisMMD content. The remaining 100 deliberately combine content from
different events to provide controlled location-conflict coverage. In total, 421
unique tweet identifiers represented in the fixture were excluded from the general
splits.

Image severity was derived from the image annotation. Text severity was assigned by
a deterministic rubric using explicit damage and affected-population signals. A
60-record review sheet was then prepared for manual checking and adjudication.

The original text and image humanitarian labels differ in 10,003 CrisisMMD pairs.
That count identifies candidates for analysis; it does not establish 10,003
confirmed severity conflicts. Accordingly, the curated fixture supports controlled
behavioral evaluation but does not estimate disagreement prevalence in operational
disaster reporting.

## EDA artifacts

Exploratory analysis covered sample counts, class balance, image dimensions, tweet
lengths, and conflict-fixture composition:

- `disasterlens_modules/notebooks/plots/sample_counts.png`
- `disasterlens_modules/notebooks/plots/class_distribution.png`
- `disasterlens_modules/notebooks/plots/image_size_distribution.png`
- `disasterlens_modules/notebooks/plots/tweet_length_histogram.png`
- `disasterlens_modules/notebooks/plots/conflict_distribution.png`

The resulting distributions reveal substantial imbalance. AIDER is dominated by
`none`; CrisisMMD is concentrated in `other_relevant_information`,
`not_humanitarian`, and rescue/donation content. Because missing/found people and
vehicle damage are particularly rare, evaluation emphasizes macro-averaged metrics
alongside accuracy.
