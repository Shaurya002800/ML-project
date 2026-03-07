import sys
import os

# ── FIX: Stop C++ memory crashes (Segfaults) on macOS ──
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Allows PyTorch/FAISS to share memory safely
os.environ["OMP_NUM_THREADS"] = "1"          # Stops XGBoost/FAISS from spawning too many hidden threads

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import streamlit as st

st.set_page_config(page_title="AgriGPT — Disease Detector", page_icon="🌿", layout="wide")

# ── Cache heavy models so they load ONCE only ─────────
@st.cache_resource
def load_models():
    from models.train_disease import predict_disease
    return predict_disease

@st.cache_resource
def load_agrigpt():
    # Import the answer function AND the vectorstore getter
    from agrigpt.rag_pipeline import agrigpt_answer, get_vectorstore
    
    # FIX: "Warm up" the vectorstore here. 
    # This forces FAISS and HuggingFace to load exactly ONCE 
    # and locks them safely inside Streamlit's memory manager.
    get_vectorstore()
    
    return agrigpt_answer

predict_disease = load_models()
agrigpt_answer  = load_agrigpt()

# ── Header ─────────────────────────────────────────────
st.title("🌿 Vivayu + AgriGPT Disease Detection Platform")
st.caption("Enter VOC sensor readings to detect crop disease and get treatment advice")
st.divider()

# ── Health banner: show vectorstore / key status so users know why AgriGPT might fail
from agrigpt import rag_pipeline as _rp
_vec_ok = False
_key_ok = False
try:
    _vec_ok = __import__('os').path.exists(_rp.VECTORSTORE_DIR)
except Exception:
    _vec_ok = False

try:
    _key_ok = bool(__import__('os').environ.get('GROQ_API_KEY'))
except Exception:
    _key_ok = False

if not _vec_ok or not _key_ok:
    msg_parts = []
    if not _vec_ok:
        msg_parts.append("Vectorstore missing (run build_vectorstore())")
    if not _key_ok:
        msg_parts.append("GROQ_API_KEY not set in .env")
    st.warning(" • ".join(msg_parts))
    # Offer a one-click builder for convenience (may be slow; runs in-process)
    if not _vec_ok:
        if st.button("Build vectorstore now (may take a while)"):
            try:
                with st.spinner("Building vectorstore (this can take a few minutes)..."):
                    _rp.build_vectorstore()
                st.success("Vectorstore built — reload the app if needed.")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to build vectorstore: {e}")

# ── Sidebar ────────────────────────────────────────────
st.sidebar.header("📡 VOC Sensor Readings")

# FIX 3: Only Wheat is valid — model was trained on Wheat only
crop        = st.sidebar.selectbox("Crop Type", ["Wheat"])
voc1        = st.sidebar.slider("VOC1 Reading", 0.0, 1.0, 0.32, 0.01)
voc2        = st.sidebar.slider("VOC2 Reading", 0.0, 1.0, 0.55, 0.01)
voc3        = st.sidebar.slider("VOC3 Reading", 0.0, 1.0, 0.21, 0.01)
humidity    = st.sidebar.number_input("Humidity (%)", 30, 100, 82)
temperature = st.sidebar.number_input("Temperature (°C)", 10, 50, 29)
run         = st.sidebar.button("🔬 Analyse Crop", use_container_width=True)

# ── Sensor value guide in sidebar ─────────────────────
st.sidebar.divider()
st.sidebar.caption("**VOC guide for Wheat:**")
st.sidebar.caption("🟢 Healthy: VOC1 < 0.20")
st.sidebar.caption("🟡 Powdery Mildew: VOC1 0.20–0.41")
st.sidebar.caption("🔴 Rust: VOC1 > 0.34")

# ── Main panel ─────────────────────────────────────────
if run:
    # Step 1: Disease prediction
    with st.spinner("Analysing sensor data..."):
        try:
            disease, confidence = predict_disease(voc1, voc2, voc3, humidity, temperature, crop)
        except Exception as e:
            st.error(f"Disease model error: {e}")
            st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("🔬 Disease Detected", disease.replace("_", " "))
    col2.metric("📊 Confidence",        f"{confidence}%")
    col3.metric("🌾 Crop",              crop)

    if disease == "Healthy":
        st.success("✅ Your crop appears healthy! No disease detected.")
    else:
        st.warning(f"⚠️ {disease.replace('_', ' ')} detected on your {crop}.")

    # Step 2: AgriGPT treatment advice
    st.subheader("🤖 AgriGPT Treatment Recommendation")
    with st.spinner("Getting treatment advice..."):
        try:
            # Direct call - much safer in Streamlit
            advice = agrigpt_answer(
                question=f"What is the treatment and pesticide dosage for {disease.replace('_',' ')} on {crop}?",
                disease_context=f"{disease.replace('_',' ')} on {crop} detected with {confidence}% confidence"
            )
            st.info(advice)
        except Exception as e:
            st.error(f"AgriGPT error: {str(e)}")

else:
    st.info("👈 Set VOC readings in the sidebar and click **Analyse Crop**")

# ── AgriGPT Chat ───────────────────────────────────────
st.divider()
st.subheader("💬 Ask AgriGPT Anything")
st.caption("Ask about wheat diseases, farming tips, government schemes, or crop calendar")

question = st.text_input("Type your question here and press Enter...")

if question:
    placeholder = st.empty()
    placeholder.info("⏳ AgriGPT is thinking...")
    try:
        # Direct call instead of threading
        chat_answer = agrigpt_answer(question)
        placeholder.empty() # Clear the "thinking" message
        st.success(chat_answer)
    except Exception as e:
        placeholder.empty()
        st.error(f"AgriGPT error: {str(e)}")