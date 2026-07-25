import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))
from disasterlens_modules.vision_module import predict_severity  # noqa: E402
from disasterlens_modules.nlp_module import analyze_text as roberta_analyze_text  # noqa: E402

sys.path.append(str(BASE_DIR / "text" / "entity_extraction"))
from extract import extract_entities  # noqa: E402

sys.path.append(str(BASE_DIR / "fusion"))
from fusion import fuse  # noqa: E402

CLASS_NAMES = ["Earthquake", "Fire", "Flood", "Normal"]
SEVERITY_COLORS = {"none": "green", "mild": "beige", "moderate": "orange", "severe": "red"}

SAMPLE_TEXTS = {
    "Earthquake": "A magnitude 6.5 earthquake has collapsed several buildings in the old town district. Rescue teams report dozens trapped; medical aid and shelter are urgently needed.",
    "Fire": "A wildfire is rapidly spreading through the hillside community, forcing evacuation of about 800 residents. Firefighters are requesting additional water supplies.",
    "Flood": "Heavy flooding has displaced an estimated 3,000 residents in the riverside district of Millbrook. Local shelters are overwhelmed and volunteers are calling for clean water and medical supplies.",
    "Normal": "The city skyline was calm this evening with light traffic and clear weather reported across downtown.",
}


def get_sample_image_path(category):
    class_dir = BASE_DIR / "data" / "aider" / "Test" / category
    first_image = sorted(os.listdir(class_dir))[0]
    return class_dir / first_image


st.title("DisasterLens — Relief Priority Mapper")

input_mode = st.radio("Input source", ["Use a sample AIDER image", "Upload my own image"])

image = None
image_path = None
default_text = ""
if input_mode == "Use a sample AIDER image":
    sample_class = st.selectbox("Sample disaster category", CLASS_NAMES)
    image_path = get_sample_image_path(sample_class)
    image = Image.open(image_path).convert("RGB")
    default_text = SAMPLE_TEXTS[sample_class]
else:
    uploaded_image = st.file_uploader("Upload a disaster-scene image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        # predict_severity expects a file path, so persist the upload to disk.
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp.name)
        image_path = tmp.name

text_snippet = st.text_area("Related text report/snippet", value=default_text, height=100)

if image is not None:
    st.image(image, caption="Input image", width=300)

clicked = st.button("Run analysis")

# Compute once on click and stash in session_state — st_folium triggers its own
# rerun after first render, which would otherwise wipe a transient result block.
if clicked and image_path is not None and text_snippet:
    # Vision: Pratyusha's CLIP-based classifier predicts severity directly.
    vision_result = predict_severity(str(image_path))

    # Text: two independent signals — Gemini LLM extraction and RoBERTa classification.
    gemini_entities = extract_entities(text_snippet)
    roberta_result = roberta_analyze_text(text_snippet)

    # Fusion: combine vision + both text signals into a final priority assessment.
    result = fuse(
        vision_severity=vision_result["severity"],
        vision_confidence=vision_result["confidence"],
        gemini_entities=gemini_entities,
        roberta_output=roberta_result,
    )

    st.session_state.result_data = {
        "vision_result": vision_result,
        "gemini_entities": gemini_entities,
        "roberta_result": roberta_result,
        "result": result,
    }

if "result_data" in st.session_state:
    data = st.session_state.result_data
    vision_result = data["vision_result"]
    gemini_entities = data["gemini_entities"]
    roberta_result = data["roberta_result"]
    result = data["result"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Vision signal")
        st.write(f"Severity: **{vision_result['severity']}**")
        st.write(f"Confidence: {vision_result['confidence']:.2f}")
    with col2:
        st.subheader("Text (LLM extractor)")
        st.write(f"Population estimate: {gemini_entities.population_estimate}")
        st.write(f"Resources mentioned: {gemini_entities.resource_types_mentioned}")
        st.write(f"Locations: {gemini_entities.locations}")
    with col3:
        st.subheader("Text (RoBERTa)")
        st.write(f"Resources mentioned: {roberta_result['resources_mentioned']}")
        st.write(f"Location: {roberta_result['location']}")
        st.write(f"Population estimate: {roberta_result['population_estimate']}")

    st.subheader("Fusion result")
    st.write(f"Final severity: **{result.final_severity}**")
    st.bar_chart({
        "food": result.resource_needs.food,
        "water": result.resource_needs.water,
        "shelter": result.resource_needs.shelter,
        "medical": result.resource_needs.medical,
    })
    st.info(result.priority_summary)

    # Map: geocode the first extracted location and drop a severity-colored marker.
    location_name = (gemini_entities.locations[0] if gemini_entities.locations else roberta_result["location"])
    if location_name:
        geolocator = Nominatim(user_agent="disasterlens")
        location = geolocator.geocode(location_name)
        if location:
            m = folium.Map(location=[location.latitude, location.longitude], zoom_start=10)
            folium.Marker(
                [location.latitude, location.longitude],
                popup=result.priority_summary,
                icon=folium.Icon(color=SEVERITY_COLORS.get(result.final_severity, "gray")),
            ).add_to(m)
            st_folium(m, width=700, height=450)
        else:
            st.warning(f"Could not geocode location: {location_name}")
