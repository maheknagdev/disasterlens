# DisasterLens: Multimodal Disaster Relief Resource Mapping System

## Project Overview
DisasterLens is a multimodal AI system that ingests disaster zone photos and news text snippets, then produces structured outputs: damage severity assessment, estimated resource needs (food, water, shelter, medical), and a prioritized resource allocation map.

**Core research question:** Can a multimodal pipeline combining vision and language models reliably infer localized relief priorities from unstructured real-world inputs, without requiring structured sensor data or official government reports?

## Team
- **Mahek Nagdev** — MS Computer Science, Year 1 — nagdev.m@northeastern.edu
- **Pratyusha Jaitly** — MS Computer Science, Year 2 — jaitly.p@northeastern.edu

## Division of Labor

**Mahek:**
- Streamlit app
- GPT-4o fusion layer
- Folium map
- CLIP fine-tuning
- GPT-4o entity extraction
- All evaluation metrics + confusion matrices
- Integration + final demo

**Pratyusha:**
- AIDER + CrisisMMD dataset download, cleaning, EDA
- RoBERTa resource classification
- Data + Methods report sections
- Her portion of the presentation

## Datasets
- **AIDER** (Aerial Image Dataset for Emergency Response) — labeled aerial/ground-level photos across disaster categories (floods, fires, collapsed structures). [Nia et al., ECCV Workshops, 2020]
- **CrisisMMD** — multimodal benchmark of tweets paired with disaster images, labeled for informativeness and humanitarian categories (infrastructure damage, affected individuals, rescue efforts). [Alam et al., ICWSM, 2018]

Inputs are (image, text snippet) pairs representing a single disaster report.

## Pipeline / Methods
1. **Vision encoder** (fine-tuned CLIP or lightweight ViT) — classifies image-level damage severity into 4 levels: none, mild, moderate, severe.
2. **Text module** (RoBERTa or GPT-4o via API) — extracts structured fields: affected population estimate, resource types mentioned, location entities.
3. **Fusion layer** — combines both outputs, queries an LLM to generate a natural-language relief priority summary and populate a resource-need vector (food, water, shelter, medical) on a 1–5 urgency scale.
4. **Visualization** — interactive map via Folium, markers color-coded by severity.

## Known Challenges (from proposal)
- Handling low-quality/ambiguous images common in real disaster scenarios
- Fusing modalities that may contradict each other (e.g., mild-looking photo + high-casualty text report)
- Preventing LLM hallucination of resource needs
- Mitigations planned: confidence thresholding, modality-conflict flags, grounding outputs to structured label sets

## Advisor Feedback (Dr. Mohammad Toutiaee) & Resulting Action Items

### 1. Conflicting-modality test set (owned by Pratyusha, as part of EDA)
Feedback: need a concrete test set for cases where image and text severity signals disagree, rather than assuming the fusion layer handles it gracefully.

Action items:
- Mine real conflict cases from CrisisMMD (text severity vs. image severity mismatch)
- Supplement with synthetic conflicts (e.g., pair a "none/mild" AIDER image with a hand-written severe caption, and vice versa)
- Tag each case using the schema below
- Keep as a separate labeled subset, not mixed into general training data
- Decide (jointly, Mahek + Pratyusha) what "handling gracefully" means for the fusion layer: flag to user / default to more severe estimate / average — this decision should be made before building the fusion logic, not patched in after

**Conflict-case tagging schema:**
| Field | Values | Notes |
|---|---|---|
| `pair_id` | unique ID | ties back to original CrisisMMD tweet/image pair |
| `conflict_type` | `severity_mismatch`, `resource_mismatch`, `location_mismatch`, `none` | core tag |
| `image_severity_signal` | `none`/`mild`/`moderate`/`severe` | image-only read |
| `text_severity_signal` | `none`/`mild`/`moderate`/`severe` | text-only read |
| `conflict_direction` | `image>text` or `text>image` | informs fusion layer's caution policy |
| `source` | `real` or `synthetic` | real CrisisMMD case vs. hand-constructed |

### 2. Evaluating urgency scores without ground truth (owned by Mahek, evaluation)
Feedback: urgency scores (1–5) aren't directly labeled in AIDER/CrisisMMD; need a defined scoring rubric or proxy labels.

Options considered, roughly by rigor vs. effort:
- **Proxy labels from CrisisMMD categories** (primary approach) — deterministic rubric mapping existing humanitarian categories (infrastructure damage, affected individuals, rescue efforts) to a 1–5 urgency scale
- **Small human-labeled subset** (sanity check) — Mahek + Pratyusha independently rate 30–50 examples, check agreement (percent agreement or Cohen's kappa)
- **Relative/ranking evaluation** — Spearman's rank correlation instead of exact-score matching, since triage is fundamentally about relative ordering

**Chosen approach:** proxy labels from CrisisMMD categories as primary evaluation method, plus a small human-labeled subset as qualitative sanity check.

**Coordination note:** the granularity of Pratyusha's RoBERTa classifier output categories needs to align with what the urgency rubric requires — sync needed on whether current categories are sufficient or need refinement.

## Tech Stack

**Core environment:**
- Python 3.10+ (3.11 recommended)
- Conda (preferred over venv — handles CUDA deps better if GPU is used)
- VS Code or PyCharm
- Jupyter Notebook/Lab for exploratory work

**Vision (Stage 1 — CLIP/ViT on AIDER):**
- `torch`, `torchvision`
- `transformers` (Hugging Face — pretrained CLIP/ViT checkpoints)
- `timm` (lightweight ViT variants, if used)
- `Pillow`, `opencv-python`

**Text (Stage 2 — RoBERTa / GPT-4o):**
- `transformers` (RoBERTa)
- `openai` Python SDK (GPT-4o API)
- `python-dotenv` (API key management)
- `spacy` (NER for location extraction, if not fully delegated to GPT-4o)

**Fusion (Stage 3):**
- `openai` SDK (reused)
- `pydantic` (structured/validated output schema for resource-need vector)

**Visualization:**
- `folium`
- `pandas`

**Dataset handling:**
- `datasets` (Hugging Face)
- `numpy`

**Dev hygiene:**
- Git + public GitHub repo (required for report appendix)
- `requirements.txt` or `environment.yml`
- `.env` + `.gitignore` for API keys

## Key Deliverable Dates (course requirements)
- Project Proposal — Due June 28 (submitted)
- Project Report — Due July 26 (max 8 pages, scientific paper format, PDF)
- Pre-recorded Presentation — Due July 26 (5 ± 1 min, MP4/MOV)

## Report Requirements to Track
- **AI Prompts Used** section is mandatory — log all Claude/AI prompts used for coding, debugging, explanations, research
- Statement of Contributions required (per-member breakdown)
- Public GitHub repo link required in Appendix
- All claims must be supported by citations, results, or common knowledge — no unsupported statements

## Open Items / Next Steps
- [ ] Pratyusha: begin conflict-case mining from CrisisMMD during EDA
- [ ] Pratyusha: tag conflict subset using schema above
- [ ] Mahek + Pratyusha: jointly decide fusion layer's conflict-handling policy
- [ ] Mahek + Pratyusha: sync on RoBERTa category granularity vs. urgency rubric needs
- [ ] Mahek: draft proxy-label rubric mapping CrisisMMD categories to 1–5 urgency scale
- [ ] Both: maintain running log of AI prompts used (required for report)
