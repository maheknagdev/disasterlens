# DisasterLens

A multimodal pipeline that infers localized disaster relief priorities from a paired aerial image and text report — combining a fine-tuned CLIP vision classifier, two independent text-analysis systems (a Gemini-based LLM extractor and a fine-tuned RoBERTa classifier), and an LLM fusion layer, visualized through an interactive Streamlit + Folium map.

Built for a graduate ML course final project by Mahek Nagdev and Pratyusha Jaitly.

## What it does

Given a disaster-scene image and an accompanying text report, DisasterLens produces:
- A severity assessment (`none` / `mild` / `moderate` / `severe`) from the image
- Structured entities from the text (affected population, resource needs, locations) via two independent methods
- A fused final severity, a resource-need vector (food/water/shelter/medical, 1-5 urgency), and a natural-language priority summary
- An interactive map with a severity-colored marker at the extracted location

## Project structure

```
vision/clip_finetune/        Exploratory CLIP category classifier (Mahek)
text/entity_extraction/      Gemini-based structured entity extraction (Mahek)
fusion/                      Multi-signal fusion layer (Mahek)
app/                         Streamlit + Folium demo application (Mahek)
evaluation/                  Confusion matrices, urgency proxy-label rubric (Mahek)
disasterlens_modules/        Direct-severity CLIP classifier, RoBERTa resource
                              classifier, EDA, conflict-evaluation fixture (Pratyusha)
```

## Setup

1. Create the environment:
   ```bash
   conda create -n disasterlens python=3.11
   conda activate disasterlens
   pip install -r requirements.txt
   ```
2. Install [git-lfs](https://git-lfs.github.com/) (`brew install git-lfs`) before cloning — the RoBERTa checkpoint is stored via Git LFS.
3. Copy `.env.example` to `.env` and add a Gemini API key (from [aistudio.google.com](https://aistudio.google.com)).
4. Run the app:
   ```bash
   cd app
   streamlit run app.py
   ```

## Datasets

- **AIDER** (Aerial Image Dataset for Emergency Response Applications) — Kyrkou & Theocharides, 2019/2020
- **CrisisMMD** — multimodal Twitter dataset of disaster reports — Alam, Ofli, & Imran, 2018

See the final report for full methodology, results, and citations.
