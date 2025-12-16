
import os
import difflib
import tempfile
import time
from typing import Dict, Any, Tuple, List, Optional

import streamlit as st

# optional dotenv for local dev
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from ollama import Client

# PDF export
try:
    from fpdf import FPDF

    HAS_FPDF = True
except Exception:
    HAS_FPDF = False

# ---------------- Page setup ----------------
st.set_page_config(
    page_title="djGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Professional Dark UI CSS (Fixed Height & Layouts) ----------------
st.markdown(
    """
<style>
/* ===============================
   GLOBAL RESET
   =============================== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background-color: #000000 !important;
    color: #E5E5E5 !important;
}

html, body, .stMarkdown, p {
    font-family: 'Inter', system-ui, sans-serif;
    color: #E5E5E5 !important;
}

/* ===============================
   SIDEBAR
   =============================== */
[data-testid="stSidebar"] {
    background-color: #0A0A0A !important;
    border-right: 1px solid #262626;
}

/* ===============================
   MODEL CARD (CORE FIX)
   =============================== */
.model-card {
    background-color: #1A1A1A;
    border: 1px solid #333333;
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;

    /* IMPORTANT */
    overflow-x: auto;
    overflow-y: visible;
    max-width: 100%;
}

/* Prevent Streamlit phantom height */
.element-container:has(.model-card) {
    height: auto !important;
    min-height: 0 !important;
}

/* ===============================
   MODEL CONTENT
   =============================== */
.model-header {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    color: #A1A1AA;
    margin-bottom: 0.4rem;
    border-bottom: 1px solid #333;
    padding-bottom: 0.25rem;
}

.model-body {
    font-size: 0.95rem;
    line-height: 1.55;
    white-space: normal;
    max-width: 100%;
}

/* ===============================
   TABLE FIX (THIS IS THE MAGIC)
   =============================== */
.model-body table {
    width: max-content !important;
    max-width: 100% !important;
    border-collapse: collapse;
    margin: 0.5rem 0;
}

/* Wrap table in scroll */
.model-card {
    scrollbar-width: thin;
}

.model-body th,
.model-body td {
    padding: 6px 10px;
    border: 1px solid #2a2a2a;
    white-space: nowrap;
}

.model-body th {
    background-color: #111827;
    font-weight: 600;
}

/* ===============================
   CONSENSUS CARD (GREEN)
   =============================== */
.consensus-card {
    background: linear-gradient(145deg, #064E3B, #065F46);
    border: 1px solid #10B981;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;

    /* CRITICAL */
    overflow-x: auto;
    max-width: 100%;
}

.consensus-card table {
    width: max-content !important;
    max-width: 100% !important;
}

/* ===============================
   CHAT INPUT
   =============================== */
.stChatInput > div {
    background-color: #171717 !important;
    border-color: #404040 !important;
}

/* ===============================
   CLEANUP
   =============================== */
.stMarkdown {
    margin-bottom: 0.4rem !important;
}


</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Environment ----------------
DEFAULT_HOST = "https://ollama.com"
OLLAMA_HOST = os.getenv("OLLAMA_HOST") or st.secrets.get("OLLAMA_HOST", DEFAULT_HOST)
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY") or st.secrets.get("OLLAMA_API_KEY", "")

# ---------------- Models ----------------
ALL_MODELS = [
    "gpt-oss:120b-cloud",
    "ministral-3:14b-cloud",
    "deepseek-v3.1:671b-cloud",
    "qwen3-vl:235b-cloud",
    "qwen3-coder:480b-cloud",
    "gemma3:27b-cloud",
]

VISION_MODELS = {"qwen3-vl:235b-cloud"}
CODER_MODELS = {"qwen3-coder:480b-cloud"}


def supports_vision(model: str) -> bool:
    return model in VISION_MODELS


def looks_like_code(prompt: str) -> bool:
    if not prompt:
        return False
    p = prompt.lower()
    code_keywords = ["code", "python", "c++", "java", "loop", "function", "class", "bug", "error"]
    return any(k in p for k in code_keywords) or "```" in prompt


# ---------------- Session state ----------------
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = []

if "plain_history" not in st.session_state:
    st.session_state.plain_history: List[Dict[str, str]] = []


# ---------------- Sidebar / Settings ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    backend_mode = st.radio(
        "Backend",
        ["Cloud ", "Local Indexing"],
        index=0,
    )

    uploaded_image = st.file_uploader("📷 Attach image (optional)", type=["png", "jpg", "jpeg", "webp"])

    available_models = [m for m in ALL_MODELS if supports_vision(m)] if uploaded_image else ALL_MODELS

    selected_models = st.multiselect(
        "Select models",
        options=available_models,
        default=available_models[:3],
    )

    conversation_mode = st.radio(
        "Mode",
        options=["Independent answers", "Critique baseline", "Debate"],
        index=0,
    )

    baseline_model = st.selectbox("Baseline model", options=[""] + selected_models)
    baseline_model = baseline_model if baseline_model else None

    highlight_diffs = st.checkbox("Highlight differences", value=True)

    st.divider()
    st.caption("Debate Configuration")
    max_debate_rounds = st.number_input("Rounds", min_value=1, max_value=6, value=3)
    approval_threshold = st.slider("Approval Threshold (%)", 50, 100, 100, step=10)

    st.divider()
    if st.button("Clear History", type="primary"):
        st.session_state.messages = []
        st.session_state.plain_history = []
        st.rerun()

# ---------------- Client initialization ----------------
if backend_mode.startswith("Cloud"):
    if not OLLAMA_API_KEY:
        st.error("Cloud backend selected but OLLAMA_API_KEY is missing.")
        st.stop()
    client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"})
else:
    client = Client(host="http://localhost:11434")

# ---------------- Utilities ----------------
def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def approx_tokens(text: str) -> int:
    words = len(text.split())
    return int(words / 0.75) if words else 0


def highlight_diff(base: str, text: str) -> str:
    if similarity(base, text) < 0.4:
        return text
    base_words = base.split()
    other_words = text.split()
    sm = difflib.SequenceMatcher(None, base_words, other_words)
    out = []
    for tag, _, _, j1, j2 in sm.get_opcodes():
        words = other_words[j1:j2]
        chunk = " ".join(words)
        if not chunk:
            continue
        if tag == "equal":
            out.append(chunk)
        else:
            out.append(f"<span class='diff-added'>{chunk}</span>")
    return " ".join(out)


def prepare_image_file(image) -> Optional[str]:
    if image is None:
        return None
    suffix = ".png"
    if image.type == "image/jpeg":
        suffix = ".jpg"
    elif image.type == "image/webp":
        suffix = ".webp"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(image.getbuffer())
    tmp.close()
    return tmp.name


# ---------------- Model call (with STREAMING) ----------------
def call_model(
    model: str,
    prompt: str,
    image_path: Optional[str] = None,
    system_hint: Optional[str] = None,
    stream: bool = False,
    placeholder=None,
    messages_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, float]:
    if not isinstance(prompt, str):
        prompt = str(prompt or "")
    prompt = prompt.strip()

    system_content = "You are a helpful AI assistant. Answer clearly and concisely."
    if image_path:
        system_content += " Use the attached image if it is relevant."
    if system_hint:
        system_content += " " + system_hint

    messages = [{"role": "system", "content": system_content}]

    if messages_history:
        for m in messages_history:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": prompt or " "})
    if image_path:
        messages[-1]["images"] = [image_path]

    start = time.perf_counter()
    full_text = ""

    try:
        if stream and placeholder:
            stream_response = client.chat(model=model, messages=messages, stream=True)
            for chunk in stream_response:
                if chunk.get("message") and chunk["message"].get("content"):
                    content = chunk["message"]["content"]
                    full_text += content
                    placeholder.markdown(full_text + "▌")
            placeholder.markdown(full_text)
        else:
            r = client.chat(model=model, messages=messages)
            try:
                full_text = getattr(r.message, "content", r["message"]["content"])
            except Exception:
                full_text = str(r)
            if placeholder is not None:
                placeholder.markdown(full_text)

    except Exception as e:
        full_text = f"❌ Error: {e}"
        if placeholder is not None:
            placeholder.markdown(full_text)

    latency_ms = (time.perf_counter() - start) * 1000
    return full_text, latency_ms


# ---------------- Analysis Helpers ----------------
def compute_winner_scores(answers: Dict[str, str]) -> Dict[str, float]:
    names = list(answers.keys())
    if len(names) <= 1:
        return {names[0]: 1.0} if names else {}
    scores = {n: 0.0 for n in names}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            scores[a] += similarity(answers[a], answers[b])
    avg_scores = {n: scores[n] / (len(names) - 1) for n in names}
    return avg_scores


def generate_analysis(judge_model: str, prompt: str, answers: Dict[str, str], baseline: Optional[str]) -> str:
    if not answers:
        return ""
    names = list(answers.keys())
    base_name = baseline if baseline in names else names[0]

    analysis_prompt = f"User question:\n{prompt}\n\nAnswers:\n"
    for name, text in answers.items():
        analysis_prompt += f"\n---\nModel: {name}\n{text}\n"

    analysis_prompt += (
        "\n---\nWrite a structured analysis:\n"
        "1) Key differences\n"
        "2) Contradictions\n"
        f"3) Which is most accurate (baseline: {base_name})\n"
        "Keep it concise."
    )

    text, _ = call_model(
        judge_model,
        analysis_prompt,
        system_hint="You are a strict debate judge.",
    )
    return text

# ---------------- Rendering Logic (Layout Switcher) ----------------
def render_comparison(
    answers: Dict[str, str],
    stats: Dict[str, Dict[str, Any]],
    baseline: Optional[str],
    highlight: bool,
    mode: str, 
):
    if not answers:
        return
    model_names = list(answers.keys())
    base_name = baseline if baseline in model_names else model_names[0]
    base_text = answers.get(base_name, "")

    # Layout Logic: 
    # Debate = Side-by-Side (Columns)
    # Independent/Critique = Vertical Stack (Sequential)
    
    if mode == "Debate":
        cols = st.columns(len(model_names))
        # Iterate through columns + names
        iter_obj = zip(cols, model_names)
        is_columnar = True
    else:
        # Just use the names, no columns (Vertical stack)
        iter_obj = zip([None]*len(model_names), model_names)
        is_columnar = False

    for col, name in iter_obj:
        # Context manager for layout:
        # If columnar, use 'with col:'. If vertical, use 'st.container():' or just direct.
        
        container = col if is_columnar else st.container()
        
        with container:
            text = answers[name]
            s = stats.get(name, {})
            
            # Highlight Logic
            display_text = text
            if highlight and name != base_name:
                display_text = highlight_diff(base_text, text)
                
            # Meta info
            latency = s.get("latency_ms", 0)
            tokens = s.get("tokens", 0)
            meta_str = f"{latency:.0f}ms · {tokens} tokens"
            
            card_html = f"""
<div class="model-card">
    <div class="model-header">{name}</div>
    <div class="model-body">{display_text}</div>
    <div class="model-meta">{meta_str}</div>
</div>"""
            st.markdown(card_html, unsafe_allow_html=True)


# ---------------- Markdown & PDF Export ----------------
def build_markdown_export(prompt, mode, baseline, answers, stats, winner_scores, judge_analysis) -> str:
    lines = [f"# djGPT Report: {mode}", f"**Prompt:** {prompt}", ""]
    if winner_scores:
        lines.append("## Consensus Scores")
        for k, v in sorted(winner_scores.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- **{k}**: {v*100:.1f}%")
        lines.append("")
    for name, text in answers.items():
        lines.append(f"## {name}")
        lines.append(text)
        lines.append("")
    if judge_analysis:
        lines.append("## Judge Analysis")
        lines.append(judge_analysis)
    return "\n".join(lines)


def build_pdf_export(markdown_text: str) -> Optional[bytes]:
    if not HAS_FPDF:
        return None
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=11)
    pdf.add_page()
    text = markdown_text.replace("’", "'").replace("–", "-").replace("—", "-")
    for line in text.split("\n"):
        pdf.write(6, line + "\n")
    return pdf.output(dest="S").encode("latin1")


# ---------------- Main UI ----------------
st.title("🤖 djGPT")

# --- History Rendering ---
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]

    with st.chat_message(role):
        if role == "user":
            st.markdown(content)
        elif role == "assistant":
            if isinstance(content, dict) and content.get("is_debate_result"):
                # Debate Result -> Show Summary Card + Expandable Details
                best_model = content.get("best_model")
                best_score = content.get("best_score", 0)
                final_text = content.get("final_text", "")

                st.markdown(
                    f"""
                    <div class="consensus-card">
                        <div class="consensus-badge">🏆 Final Consensus • {best_score*100:.0f}% Agreement</div>
                        <h3 style="margin:0 0 1rem 0;">{best_model}</h3>
                        <div style="font-size:1.1rem; line-height:1.6; color:#ECFDF5;">
                            {final_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                with st.expander("Show Debate Process (Horizontal View)", expanded=False):
                    st.markdown('<div class="debate-grid">', unsafe_allow_html=True)
                    render_comparison(
                        content.get("answers", {}),
                        content.get("stats", {}),
                        content.get("baseline"),
                        highlight_diffs,
                        mode="Debate" # Force Horizontal for history detail view
                    )
                    
                    judge_analysis = content.get("judge_analysis")
                    if judge_analysis:
                        st.markdown("---")
                        st.markdown("#### ⚖️ Judge Analysis")
                        st.markdown(f"<div style='color:#A1A1AA;'>{judge_analysis}</div>", unsafe_allow_html=True)

            elif isinstance(content, dict):
                # Standard independent comparison -> Render based on stored mode
                # If mode is missing from old history, default to "Independent" (Vertical)
                stored_mode = content.get("mode", "Independent answers")
                render_comparison(
                    content.get("answers", {}),
                    content.get("stats", {}),
                    content.get("baseline"),
                    highlight_diffs,
                    mode=stored_mode
                )
            else:
                st.markdown(content)

# --- Chat Input ---
prompt = st.chat_input("Ask your question")

if prompt:
    if not selected_models:
        st.error("Select at least one model.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.plain_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    image_path = prepare_image_file(uploaded_image)
    history_to_send = st.session_state.plain_history[-10:]

    live_container = st.container()

    if conversation_mode == "Debate":
        if not baseline_model:
            st.error("Baseline model required for debate.")
            st.stop()

        with live_container:
            # Init Data
            debate_data = {
                "is_debate_result": True,
                "baseline": baseline_model,
                "mode": "Debate",
                "answers": {},
                "stats": {}
            }

            # Phase 1: Baseline
            st.caption("Phase 1: Generating Baseline")
            ph_base = st.empty()
            base_text, base_lat = call_model(baseline_model, prompt, image_path, messages_history=history_to_send)
            ph_base.markdown(base_text)
            
            debate_data["answers"][baseline_model] = base_text
            debate_data["stats"][baseline_model] = {"latency_ms": base_lat, "tokens": approx_tokens(base_text)}
            current_answer = base_text
            
            critics = [m for m in selected_models if m != baseline_model]
            
            # Loop
            if critics:
                for round_idx in range(1, max_debate_rounds + 1):
                    # Phase 2: Critiques (Horizontal Columns for Live View)
                    st.markdown("---")
                    st.caption(f"Phase 2: Critiques (Round {round_idx})")
                    
                    crit_cols = st.columns(len(critics)) # HORIZONTAL
                    critiques_map = {}
                    
                    for idx, model in enumerate(critics):
                        with crit_cols[idx]:
                            st.markdown(f"**{model}**")
                            ph_crit = st.empty()
                            c_prompt = f"Original Q: {prompt}\nBaseline A: {current_answer}\nCritique this. Be concise."
                            c_text, c_lat = call_model(model, c_prompt, image_path, system_hint="You are a critic.", messages_history=history_to_send, stream=True, placeholder=ph_crit)
                            critiques_map[model] = c_text
                            
                            key = f"{model} (R{round_idx})"
                            debate_data["answers"][key] = c_text
                            debate_data["stats"][key] = {"latency_ms": c_lat, "tokens": approx_tokens(c_text)}

                    # Phase 3: Rebuttal
                    st.caption(f"Phase 3: Rebuttal (Round {round_idx})")
                    r_prompt = f"User Q: {prompt}\nYour Prev A: {current_answer}\nCritiques: {critiques_map}\nProvide a final corrected answer."
                    ph_reb = st.empty()
                    reb_text, reb_lat = call_model(baseline_model, r_prompt, image_path, system_hint="Improve your answer based on critiques.", messages_history=history_to_send, stream=True, placeholder=ph_reb)
                    
                    current_answer = reb_text
                    final_rebuttal = reb_text
                    debate_data["answers"][f"{baseline_model} (R{round_idx} Rebuttal)"] = reb_text
                    
                    # Phase 4: Approval
                    votes = 0
                    for model in critics:
                        a_prompt = f"New Answer: {current_answer}\nDo you APPROVE or REJECT? Start with keyword."
                        a_text, _ = call_model(model, a_prompt, system_hint="Vote APPROVE or REJECT.")
                        if "APPROVE" in a_text.upper():
                            votes += 1
                    
                    ratio = (votes / len(critics)) * 100
                    if ratio >= approval_threshold:
                        break
            else:
                 final_rebuttal = base_text

            # Finalize
            final_pool = {k: v for k, v in debate_data["answers"].items() if "Rebuttal" in k or k in critics}
            if not final_pool: final_pool = {baseline_model: final_rebuttal}
            
            scores = compute_winner_scores(final_pool)
            best_model, best_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)[0] if scores else (baseline_model, 1.0)
            
            debate_data["best_model"] = baseline_model
            debate_data["best_score"] = best_score
            debate_data["final_text"] = final_rebuttal
            debate_data["judge_analysis"] = generate_analysis(baseline_model, prompt, debate_data["answers"], baseline_model)
            
            st.session_state.messages.append({"role": "assistant", "content": debate_data})
            st.rerun()

    else:
        # Non-Debate Mode (Independent/Critique) -> VERTICAL STACK
        with live_container:
            answers = {}
            stats = {}
            
            # Iterate sequentially (no columns = vertical stack)
            for model in selected_models:
                st.markdown(f"### {model} ...")
                card_placeholder = st.empty()

                sys_hint = None
                if conversation_mode == "Critique baseline" and baseline_model and model != baseline_model:
                    sys_hint = f"Critique {baseline_model}."

                # Stream response
                text, lat = call_model(
                    model, 
                    prompt, 
                    image_path, 
                    system_hint=sys_hint, 
                    stream=True, 
                    placeholder=card_placeholder,
                    messages_history=history_to_send
                )
                
                answers[model] = text
                stats[model] = {"latency_ms": lat, "tokens": approx_tokens(text)}
                
                # Snap to Card Format
                display_text = text
                # Try to diff against baseline if available
                if highlight_diffs and baseline_model and model != baseline_model:
                     base_text = answers.get(baseline_model, "")
                     if base_text:
                         display_text = highlight_diff(base_text, text)

                meta_str = f"{lat:.0f}ms · {approx_tokens(text)} tokens"
                
                card_html = f"""
<div class="model-card">
    <div class="model-header">{model}</div>
    <div class="model-body">{display_text}</div>
    <div class="model-meta">{meta_str}</div>
</div>"""
                card_placeholder.empty()
                card_placeholder.markdown(card_html, unsafe_allow_html=True)
            
            # Save results
            scores = compute_winner_scores(answers)
            judge_res = generate_analysis(baseline_model or selected_models[0], prompt, answers, baseline_model)
            
            msg_payload = {
                "prompt": prompt,
                "answers": answers,
                "stats": stats,
                "baseline": baseline_model,
                "mode": conversation_mode, # Save mode to render correctly later
                "winner_scores": scores,
                "judge_analysis": judge_res
            }
            st.session_state.messages.append({"role": "assistant", "content": msg_payload})
            st.rerun()

