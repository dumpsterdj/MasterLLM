import os
import difflib
import tempfile
import time
from typing import Dict, Any, Tuple

import streamlit as st
from ollama import Client

# ---------------- PDF SUPPORT ----------------
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="djGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- STYLES ----------------
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.model-header {
    font-weight: bold;
    color: #4DA6FF;
    border-bottom: 2px solid #333;
    padding-bottom: 5px;
    margin-bottom: 8px;
}
.diff-added {
    background-color: rgba(239,68,68,0.25);
    border-radius: 4px;
    padding: 0 3px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- ENV ----------------
DEFAULT_HOST = "https://ollama.com"

OLLAMA_HOST = (
    os.getenv("OLLAMA_HOST")
    or st.secrets.get("OLLAMA_HOST", DEFAULT_HOST)
)

OLLAMA_API_KEY = (
    os.getenv("OLLAMA_API_KEY")
    or st.secrets.get("OLLAMA_API_KEY", "")
)

RUNNING_ON_STREAMLIT = bool(os.getenv("STREAMLIT_SERVER_RUN_ON_SAVE"))

# ---------------- MODELS ----------------
ALL_MODELS = [
    "gpt-oss:120b-cloud",
    "gemma3:27b-cloud",
    "deepseek-v3.1:671b-cloud",
    "qwen3-vl:235b-cloud",
    "qwen3-coder:480b-cloud",
    "ministral-3:14b-cloud",
]

VISION_MODELS = {"qwen3-vl:235b-cloud"}
CODER_MODELS = {"qwen3-coder:480b-cloud"}

# ---------------- HELPERS ----------------
def supports_vision(m): return m in VISION_MODELS

def looks_like_code(p: str) -> bool:
    if not p:
        return False
    keywords = ["code", "python", "java", "loop", "function", "bug"]
    return any(k in p.lower() for k in keywords) or "```" in p

def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def approx_tokens(t: str) -> int:
    return int(len(t.split()) / 0.75) if t else 0

def highlight_diff(base: str, text: str) -> str:
    sm = difflib.SequenceMatcher(None, base.split(), text.split())
    out = []
    for tag, _, _, j1, j2 in sm.get_opcodes():
        chunk = " ".join(text.split()[j1:j2])
        if not chunk:
            continue
        if tag == "equal":
            out.append(chunk)
        else:
            out.append(f"<span class='diff-added'>{chunk}</span>")
    return " ".join(out)

def prepare_image_file(img):
    if not img:
        return None
    suffix = ".png"
    if img.type == "image/jpeg": suffix = ".jpg"
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(img.getbuffer())
    f.close()
    return f.name

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    backend_mode = st.radio(
        "Backend",
        ["Cloud (ollama.com)", "Local (localhost)"],
        index=0,
        disabled=RUNNING_ON_STREAMLIT,
        help="Local mode works only on your own machine"
    )

    uploaded_image = st.file_uploader(
        "📷 Attach image (optional)",
        type=["png", "jpg", "jpeg", "webp"]
    )

    models_available = (
        [m for m in ALL_MODELS if supports_vision(m)]
        if uploaded_image else ALL_MODELS
    )

    selected_models = st.multiselect(
        "Models",
        models_available,
        default=models_available[:3]
    )

    conversation_mode = st.radio(
        "Mode",
        ["Independent answers", "Critique baseline", "Debate"]
    )

    baseline_model = (
        st.selectbox("Baseline", selected_models)
        if selected_models else None
    )

    highlight_diffs = st.checkbox("Highlight differences", value=True)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- CLIENT INIT (SAFE) ----------------
if backend_mode.startswith("Cloud"):
    if not OLLAMA_API_KEY:
        st.error("Missing OLLAMA_API_KEY (set it in Streamlit Secrets)")
        st.stop()
    client = Client(
        host=OLLAMA_HOST,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    )
else:
    client = Client(host="http://localhost:11434")

# ---------------- MODEL CALL ----------------
def call_model(
    model: str,
    prompt: str,
    image_path=None,
    system_hint=None,
    stream=False,
    placeholder=None
) -> Tuple[str, float]:

    system = "Answer clearly and concisely."
    if system_hint:
        system += " " + system_hint

    user = {"role": "user", "content": prompt}
    if image_path:
        user["images"] = [image_path]

    messages = [
        {"role": "system", "content": system},
        user
    ]

    start = time.perf_counter()
    text = ""

    try:
        if stream and placeholder:
            for chunk in client.chat(model=model, messages=messages, stream=True):
                delta = getattr(chunk.message, "content", "")
                text += delta
                placeholder.markdown(text)
        else:
            r = client.chat(model=model, messages=messages)
            text = r.message.content
            if placeholder:
                placeholder.markdown(text)
    except Exception as e:
        text = f"❌ {e}"

    return text, (time.perf_counter() - start) * 1000

# ---------------- PDF EXPORT ----------------
def build_pdf_export(md: str):
    if not HAS_FPDF:
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.set_font("Helvetica", size=11)
    pdf.add_page()

    replacements = {
        "–": "-", "—": "-", "’": "'", "“": '"', "”": '"', "•": "*"
    }

    for k, v in replacements.items():
        md = md.replace(k, v)

    for line in md.split("\n"):
        pdf.write(6, line + "\n")

    return pdf.output(dest="S").encode("latin1")

# ---------------- UI ----------------
st.title("🤖 djGPT")
st.caption("Multi-Model Cloud Debate Engine")

prompt = st.chat_input("Ask something")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    answers, stats = {}, {}
    image_path = prepare_image_file(uploaded_image)

    with st.chat_message("assistant"):
        for m in selected_models:
            ph = st.empty()
            text, lat = call_model(m, prompt, image_path=image_path, placeholder=ph)
            answers[m] = text
            stats[m] = {"latency": lat, "tokens": approx_tokens(text)}

        for m, a in answers.items():
            st.markdown(f"### {m}")
            st.markdown(
                highlight_diff(answers[baseline_model], a)
                if highlight_diffs and baseline_model and m != baseline_model
                else a,
                unsafe_allow_html=True
            )

    md = "\n".join(f"## {k}\n{v}" for k, v in answers.items())
    if HAS_FPDF:
        pdf = build_pdf_export(md)
        if pdf:
            st.download_button("⬇️ Download PDF", pdf, "djgpt.pdf")
