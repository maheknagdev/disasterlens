# Methods: NLP Module

## Humanitarian classification

The NLP classifier is based on `FacebookAI/roberta-base` and retains all eight
CrisisMMD humanitarian categories. Although DisasterLens exposes a multi-resource
output, each source annotation contains one mutually exclusive category. The model
was therefore formulated as eight-class, single-label classification rather than as
eight independent binary decisions. This choice aligns the loss function with the
annotation structure.

Input text was tokenized to a maximum sequence length of 128 tokens. Fine-tuning ran
for three epochs with seed 42 and weighted cross-entropy. Inverse-square-root class
weights, capped at 5, reduced the influence of the observed class imbalance without
allowing the rarest categories to dominate the objective. Validation macro-F1
determined checkpoint selection.

The evaluation interface `classify_resources_fine_scores(text)` exposes all eight
softmax probabilities, and `classify_resources_fine(text)` returns the
highest-probability category. The contract-facing function
`classify_resources(text)` then maps that category to the resource vocabulary of
food, water, shelter, and medical. Fine-category scores remain outside
`analyze_text()`, preserving the fusion-layer interface.

## Entity extraction

Entity extraction uses deterministic local patterns. The rules identify event
names, affected-population expressions, a predefined urgency vocabulary, and
locations introduced by common spatial prepositions. Deterministic extraction makes
the integration test reproducible, but it cannot capture the lexical and syntactic
variation expected in unrestricted social-media text.

Finally, `analyze_text(snippet)` combines the mapped resource output with the
extracted entities. Fields without textual evidence remain null or empty, preventing
unsupported values from entering the fusion layer.
