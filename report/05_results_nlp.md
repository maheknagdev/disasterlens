# Results: NLP

## Fine-category classification

The selected RoBERTa checkpoint achieved a validation macro-F1 of 0.2755. On 1,570
held-out records, accuracy was 52.74%, weighted F1 was 0.5014, and eight-category
macro-F1 was **0.2746**.

| Fine category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| affected individuals | 0.1667 | 0.0213 | 0.0377 | 47 |
| infrastructure and utility damage | 1.0000 | 0.0105 | 0.0208 | 95 |
| injured or dead people | 0.3889 | 0.5714 | 0.4628 | 49 |
| missing or found people | 0.0000 | 0.0000 | 0.0000 | 4 |
| not humanitarian | 0.5989 | 0.4802 | 0.5330 | 454 |
| other relevant information | 0.5310 | 0.5833 | 0.5559 | 588 |
| rescue, volunteering, or donation effort | 0.4927 | 0.7248 | 0.5866 | 327 |
| vehicle damage | 0.0000 | 0.0000 | 0.0000 | 6 |

Performance differed substantially by category. Rescue, volunteering, or donation
effort produced the highest F1 (0.5866), followed by other relevant information
(0.5559) and not-humanitarian content (0.5330). Missing/found people and vehicle
damage had only four and six test examples, respectively, and neither category
produced a correct prediction. Retaining them preserves distinctions relevant to an
urgency rubric, but the observed results do not support autonomous operational use.

## Contract-facing resource classification

After the fine categories were mapped to the four-resource contract, macro-F1 was
**0.4020** and micro-F1 was 0.5419.

| Resource | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| food | 0.4927 | 0.7248 | 0.5866 | 327 |
| water | 0.4927 | 0.7248 | 0.5866 | 327 |
| shelter | 1.0000 | 0.0105 | 0.0208 | 95 |
| medical | 0.4615 | 0.3750 | 0.4138 | 96 |

Food and water have identical results because both derive from the same
rescue/donation source category. Shelter behaves differently: its precision is
1.0000, but recall is only 0.0105, indicating that the model almost never emits the
corresponding infrastructure category. High precision therefore does not imply
useful shelter detection.

The integration sentence, “Heavy flooding in Dhaka. 10,000 people displaced, need
food,” illustrates the practical consequence. RoBERTa assigned
`other_relevant_information`, producing an empty resource list despite the explicit
mention of food. Better task-specific annotations, improved imbalance handling, or
a hybrid lexical rule for explicit resource mentions could address this failure.

## Entity-extraction example

For the same integration sentence, the deterministic entity extractor returned:

```json
{
  "location": "Dhaka",
  "population_estimate": 10000,
  "event_type": "flood",
  "urgency_keywords": ["displaced"]
}
```

This example verifies contract compliance rather than extraction accuracy. A labeled
entity corpus is required to estimate exact-match accuracy or field-level precision,
recall, and F1. Until such an evaluation is performed, the rule-based fields should
be treated as provisional signals.
