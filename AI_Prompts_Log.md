# AI Prompts Used — DisasterLens

**Format:** Member | Tool | Purpose | Prompt Summary | Output Used?

---

| Member | Tool | Purpose | Prompt Summary | Output Used? |
|---|---|---|---|---|
| Mahek | Claude Code | Project context recap | Asked Claude to summarize the DisasterLens CLAUDE.md project context (team, pipeline, advisor feedback, deadlines) | Used for orientation, not in report |
| Mahek | Claude Code | Environment setup | Asked Claude for steps to start building (Mahek's parts) incl. installations; Claude installed Miniconda, created `disasterlens` conda env, scaffolded project folders, installed requirements.txt | Used — env + scaffold in repo |
| Mahek | Claude Code | Git/GitHub setup | Walked through creating a personal GitHub repo, SSH key setup, fixing deploy-key read-only and divergent-branch push errors | Used — repo live at github.com/maheknagdev/disasterlens |
| Mahek | Claude Code | AIDER dataset research | Asked Claude to research the AIDER dataset (classes, image counts, format, license, citation) before downloading, to correct citation and flag category-vs-severity labeling mismatch | Used — informed dataset choice + severity-mapping design decision |
| Mahek | Claude Code | CLIP fine-tuning (linear probe) | Built dataset.py/model.py/train.py for CLIP vision-encoder linear probing on AIDERv2 (frozen backbone, trainable linear head, 4-class Earthquake/Fire/Flood/Normal); debugged a macOS multiprocessing DataLoader crash | Used — 98.56% val accuracy, checkpoint in outputs/models/ |
| Mahek | Claude Code | CLIP evaluation | Built evaluation/clip_confusion_matrix.py to run trained CLIP classifier on held-out test set and generate confusion matrix + classification report | Used — 99% test accuracy, figure saved for report |
| Mahek | Claude Code | Entity extraction (Gemini) | Built schema.py + extract.py for structured entity extraction; switched from GPT-4o to Gemini (free tier) for cost reasons, debugged deprecated model name via live model-list query | Used — verified correct extraction on sample text |
| Mahek | Claude Code | Fusion layer | Built fusion/fusion.py combining vision severity + text entities via Gemini structured output, encoding the "favor more severe reading on conflict" policy in the system prompt; caught a module-name collision bug (two files both named schema.py) before it silently broke | Used — verified correct severity escalation + resource vector on sample input |
| Mahek | Claude Code | Streamlit app + Folium map | Built app/app.py wiring CLIP + entity extraction + fusion into a UI with sample-image picker and severity-colored map marker; debugged a subtle Streamlit bug where the st_folium component's auto-rerun was wiping button-triggered results, fixed via st.session_state | Used — verified full pipeline renders correctly in browser end-to-end |
| Mahek | Claude Code | Merge Pratyusha's vision/NLP modules | Pulled her feature/vision-nlp-modules branch (git show, no commit), resolved a Git LFS pointer issue for the RoBERTa checkpoint via GitHub's media endpoint, wired her direct-severity CLIP classifier and RoBERTa resource classifier into app.py + fusion.py as additional signals alongside the Gemini pipeline | Used — verified 3-signal fusion (vision + Gemini + RoBERTa) renders correctly end-to-end |
| Mahek | Claude Code | Urgency proxy-label rubric | Built evaluation/urgency_rubric.py (CrisisMMD category -> 1-5 urgency proxy mapping) plus scripts to generate a stratified 39-example human-labeling sanity subset and compute Cohen's kappa/agreement once labeled | Used — rubric + tooling built; human labeling still pending |

---
