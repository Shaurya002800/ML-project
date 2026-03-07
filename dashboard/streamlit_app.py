import sys
import os

# ── Mac fix: add project root to path so imports work ──
script_dir   = os.path.dirname(os.path.abspath(__file__))  # dashboard/
project_root = os.path.dirname(script_dir)                 # voc-agrigpt/
sys.path.insert(0, project_root)

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # stops Mac warning

import streamlit as st
from models.train_disease import predict_disease
from agrigpt.rag_pipeline import agrigpt_answer

st.set_page_config(page_title="AgriGPT — Disease Detector", page_icon="🌿", layout="wide")

# ── Header ────────────────────────────────────────────
st.title("🌿 Vivayu + AgriGPT Disease Detection Platform")
st.caption("Enter VOC sensor readings to detect crop disease and get treatment advice")
st.divider()

# ── Sidebar: Sensor inputs ────────────────────────────
st.sidebar.header("📡 VOC Sensor Readings")
crop        = st.sidebar.selectbox("Crop Type", ["Wheat", "Tomato", "Rice", "Onion"])
voc1        = st.sidebar.slider("VOC1 Reading", 0.0, 1.0, 0.32, 0.01)
voc2        = st.sidebar.slider("VOC2 Reading", 0.0, 1.0, 0.55, 0.01)
voc3        = st.sidebar.slider("VOC3 Reading", 0.0, 1.0, 0.21, 0.01)
humidity    = st.sidebar.number_input("Humidity (%)", 30, 100, 82)
temperature = st.sidebar.number_input("Temperature (°C)", 10, 50, 29)
run         = st.sidebar.button("🔬 Analyse Crop", use_container_width=True)

# ── Main panel: Results ────────────────────────────────
if run:
    with st.spinner("Analysing sensor data..."):
        disease, confidence = predict_disease(voc1, voc2, voc3, humidity, temperature, crop)

    col1, col2, col3 = st.columns(3)
    col1.metric("🔬 Disease Detected", disease.replace("_", " "))
    col2.metric("📊 Confidence",        f"{confidence}%")
    col3.metric("🌾 Crop",              crop)

    if disease == "Healthy":
        st.success("✅ Your crop appears healthy! No disease detected.")
    else:
        st.warning(f"⚠️ {disease.replace('_', ' ')} detected on your {crop}.")
        st.subheader("🤖 AgriGPT Treatment Recommendation")
        with st.spinner("Getting treatment advice..."):
            advice = agrigpt_answer(
                question=f"Treatment for {disease} on {crop}?",
                disease_context=f"{disease} on {crop} with {confidence}% confidence"
            )
        st.info(advice)

else:
    st.info("👈 Enter sensor readings in the sidebar and click Analyse Crop")

# ── AgriGPT standalone chat ───────────────────────────
st.divider()
st.subheader("💬 Ask AgriGPT Anything")
question = st.text_input("Ask about diseases, farming tips, government schemes, crop calendar...")
if question:
    with st.spinner("AgriGPT is thinking..."):
        answer = agrigpt_answer(question)
    st.write(answer)