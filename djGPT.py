import os
import difflib
import tempfile
import time
from io import BytesIO
from typing import Dict, Any, Tuple

import streamlit as st
# from dotenv import load_dotenv
from ollama import Client

# PDF export
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# --- SETUP ---
# load_dotenv()

st.set_page_config(
    page_title="djGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
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

# --- ENV & CLIENT ---
# OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
# OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY") or st.secrets.get("OLLAMA_API_KEY")
# ---------- ENV ----------
DEFAULT_HOST = "https://ollama.com"

OLLAMA_HOST = (
    os.getenv("OLLAMA_HOST")
    or st.secrets.get("OLLAMA_HOST", DEFAULT_HOST)
)

OLLAMA_API_KEY = (
    os.getenv("OLLAMA_API_KEY")
    or st.secrets.get("OLLAMA_API_KEY", "")
)

if backend_mode.startswith("Cloud"):
    if not OLLAMA_API_KEY:
        st.error("Cloud backend selected but OLLAMA_API_KEY is missing.")
        st.stop()
    client = Client(
        host=OLLAMA_HOST,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
    )
else:
    client = Client(host="http://localhost:11434")

# if not OLLAMA_API_KEY:
#     st.error("❌ OLLAMA_API_KEY not found in .env")
#     st.stop()

# client = Client(
#     host=OLLAMA_HOST,
#     headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
# )

# --- MODELS & CAPABILITIES ---
ALL_MODELS = [
    "gpt-oss:120b-cloud",
    "gemma3:27b-cloud",
    "deepseek-v3.1:671b-cloud",
    "qwen3-vl:235b-cloud",
    "qwen3-coder:480b-cloud",
    "ministral-3:14b-cloud",
]

VISION_MODELS = {
    "qwen3-vl:235b-cloud"
}

CODER_MODELS = {
    "qwen3-coder:480b-cloud"
}


def supports_vision(model: str) -> bool:
    return model in VISION_MODELS


def looks_like_code(prompt: str) -> bool:
    if not prompt:
        return False
    p = prompt.lower()
    code_keywords = ["code", "python", "c++", "java", "loop", "function", "class", "bug", "error"]
    return any(k in p for k in code_keywords) or "```" in prompt


# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_comparison" not in st.session_state:
    st.session_state.last_comparison = None

# --- SIDEBAR ---
with st.sidebar:
    # st.header("⚙️ Settings")
    # st.success(f"Connected to {OLLAMA_HOST}")

    # st.divider()

    uploaded_image = st.file_uploader(
        "📷 Attach image (optional)",
        type=["png", "jpg", "jpeg", "webp"]
    )

    if uploaded_image:
        st.info("Image detected — only vision-capable models shown")

    available_models = (
        [m for m in ALL_MODELS if supports_vision(m)]
        if uploaded_image
        else ALL_MODELS
    )

    selected_models = st.multiselect(
        "Select models",
        options=available_models,
        default=available_models[:1] if uploaded_image else available_models[:3],
    )

    conversation_mode = st.radio(
        "Conversation mode",
        options=["Independent answers", "Critique baseline", "Debate"],
        index=0,
        help=(
            "Independent: each model answers the user.\n"
            "Critique baseline: models critique the baseline.\n"
            "Debate: models critique baseline, then baseline rebuts."
        )
    )

    baseline_model = (
        st.selectbox("Baseline model", selected_models)
        if selected_models else None
    )

    highlight_diffs = st.checkbox(
        "Highlight differences vs baseline",
        value=True
    )

    # st.divider()
    # st.caption("Model capabilities")
    # for m in ALL_MODELS:
    #     caps = []
    #     if supports_vision(m):
    #         caps.append("🖼️ Vision")
    #     if m in CODER_MODELS:
    #         caps.append("💻 Code")
    #     if not caps:
    #         caps.append("💬 Text")
    #     st.markdown(f"- `{m}` · {' · '.join(caps)}")

    # st.divider()
    if st.button("Clear Chat History", type="primary"):
        st.session_state.messages = []
        st.session_state.last_comparison = None
        st.rerun()

# ---------- CLIENT (depends on backend) ----------
if backend_mode.startswith("Cloud"):
    if not OLLAMA_API_KEY:
        st.error("Cloud backend selected but OLLAMA_API_KEY is missing.")
        st.stop()
    client = Client(
        host=OLLAMA_HOST,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
    )
else:
    # Note: Local Ollama won't work on Streamlit Cloud,
    # this is only meaningful when you run djGPT on your own machine.
    client = Client(host="http://localhost:11434")


# --- UTILS ---
def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def approx_tokens(text: str) -> int:
    words = len(text.split())
    return int(words / 0.75) if words else 0


def highlight_diff(base: str, text: str) -> str:
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


def prepare_image_file(image) -> str | None:
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


# --- MODEL CALLS ---
def call_model(
    model: str,
    prompt: str,
    image_path: str | None = None,
    system_hint: str | None = None,
    stream: bool = False,
    placeholder=None,
) -> Tuple[str, float]:
    """Call Ollama model, optionally streaming into a placeholder."""
    if not isinstance(prompt, str):
        prompt = str(prompt or "")
    prompt = prompt.strip()

    system_content = "You are a helpful AI assistant. Answer clearly and concisely."
    if image_path:
        system_content += " Use the attached image if it is relevant."
    if system_hint:
        system_content += " " + system_hint

    user_msg: Dict[str, Any] = {
        "role": "user",
        "content": prompt or "Describe the image in detail.",
    }
    if image_path:
        user_msg["images"] = [image_path]

    messages = [
        {"role": "system", "content": system_content},
        user_msg,
    ]

    start = time.perf_counter()
    full_text = ""

    try:
        if stream and placeholder is not None:
            for chunk in client.chat(model=model, messages=messages, stream=True):
                try:
                    delta = getattr(chunk.message, "content", chunk["message"]["content"])
                except Exception:
                    delta = ""
                full_text += delta
                placeholder.markdown(full_text)
        else:
            r = client.chat(model=model, messages=messages)
            full_text = getattr(r.message, "content", r["message"]["content"])
            if placeholder is not None:
                placeholder.markdown(full_text)
    except Exception as e:
        full_text = f"❌ {e}"
        if placeholder is not None:
            placeholder.markdown(full_text)

    latency_ms = (time.perf_counter() - start) * 1000
    return full_text, latency_ms


# --- ANALYSIS HELPERS ---
def compute_winner_scores(answers: Dict[str, str]) -> Dict[str, float]:
    """Consensus-based score: average similarity vs other models."""
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


def generate_analysis(
    judge_model: str,
    prompt: str,
    answers: Dict[str, str],
    baseline: str | None
) -> str:
    """Ask a judge model to analyse differences, contradictions, and pick a winner."""
    if not answers:
        return ""
    names = list(answers.keys())
    if baseline in names:
        base_name = baseline
    else:
        base_name = names[0]

    analysis_prompt = (
        "You are comparing answers from multiple models.\n\n"
        f"User question:\n{prompt}\n\n"
        "Here are the answers:\n"
    )
    for name, text in answers.items():
        analysis_prompt += f"\n---\nModel: {name}\n{text}\n"

    analysis_prompt += (
        "\n---\nWrite a structured analysis with sections:\n"
        "1) Common points across models\n"
        "2) Key differences\n"
        "3) Explicit contradictions or disagreements\n"
        f"4) Which answer seems most accurate and why (refer to the model name, baseline is {base_name})\n"
        "Keep it concise but specific.\n"
    )

    text, _ = call_model(
        judge_model,
        analysis_prompt,
        image_path=None,
        system_hint="You are acting as a strict debate judge.",
        stream=False,
        placeholder=None,
    )
    return text


def render_comparison(
    answers: Dict[str, str],
    stats: Dict[str, Dict[str, Any]],
    baseline: str | None,
    highlight: bool,
):
    if not answers:
        return

    model_names = list(answers.keys())
    if baseline in model_names:
        base_name = baseline
    else:
        base_name = model_names[0]
    base_text = answers[base_name]

    cols = st.columns(len(model_names))

    for col, name in zip(cols, model_names):
        text = answers[name]
        s = stats.get(name, {})
        with col:
            with st.container(border=True):
                st.markdown(
                    f"<div class='model-header'>{name}</div>",
                    unsafe_allow_html=True
                )

                display = text
                if highlight and name != base_name:
                    display = highlight_diff(base_text, text)

                st.markdown(display, unsafe_allow_html=True)

                latency = s.get("latency_ms")
                tokens = s.get("tokens")
                meta_bits = []
                if latency is not None:
                    meta_bits.append(f"{latency:.0f} ms")
                if tokens is not None:
                    meta_bits.append(f"~{tokens} tokens")
                if meta_bits:
                    st.caption(" · ".join(meta_bits))

                if highlight and name != base_name:
                    sim = similarity(base_text, text)
                    st.caption(f"Similarity vs {base_name}: {sim*100:.1f}%")
                elif name == base_name:
                    st.caption("Baseline model")


def build_markdown_export(
    prompt: str,
    mode: str,
    baseline: str | None,
    answers: Dict[str, str],
    stats: Dict[str, Dict[str, Any]],
    winner_scores: Dict[str, float],
    judge_analysis: str | None,
) -> str:
    lines = []
    lines.append("# djGPT Comparison\n")
    lines.append(f"**Prompt:** {prompt}")
    lines.append(f"**Mode:** {mode}")
    if baseline:
        lines.append(f"**Baseline model:** `{baseline}`")
    lines.append("")

    # Winner scores
    if winner_scores:
        lines.append("## Winner Scoring (Consensus-based)")
        for name, score in sorted(winner_scores.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- **{name}**: {score*100:.1f}% consensus score")
        lines.append("")

    for name, text in answers.items():
        s = stats.get(name, {})
        latency = s.get("latency_ms")
        tokens = s.get("tokens")
        meta_bits = []
        if latency is not None:
            meta_bits.append(f"{latency:.0f} ms")
        if tokens is not None:
            meta_bits.append(f"~{tokens} tokens")
        meta = " · ".join(meta_bits) if meta_bits else ""
        lines.append(f"## {name}")
        if meta:
            lines.append(f"*{meta}*")
        lines.append("")
        lines.append(text)
        lines.append("")

    if judge_analysis:
        lines.append("## Judge Analysis (Differences, Contradictions, Winner)")
        lines.append(judge_analysis)
        lines.append("")

    return "\n".join(lines)


# def build_pdf_export(markdown_text: str) -> bytes | None:
#     if not HAS_FPDF:
#         return None

#     pdf = FPDF(orientation="P", unit="mm", format="A4")
#     pdf.set_auto_page_break(auto=True, margin=15)
#     pdf.set_font("Arial", size=12)

#     # Simple conversion: split on headings for slide-like pages
#     sections = markdown_text.split("\n## ")
#     for i, sec in enumerate(sections):
#         pdf.add_page()
#         if i == 0:
#             # first "slide" already has "# " maybe
#             text = sec
#         else:
#             text = "## " + sec
#         for line in text.split("\n"):
#             pdf.multi_cell(0, 6, line)

#     output = pdf.output(dest="S").encode("latin1")
#     return output

def build_pdf_export(markdown_text: str) -> bytes | None:
    """
    Bulletproof PDF export:
    - Uses write() instead of multi_cell()
    - Avoids line-width calculations
    - Cannot trigger FPDF horizontal space errors
    """
    if not HAS_FPDF:
        return None

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # ✅ Use core font only (no deprecations)
    pdf.set_font("Helvetica", size=11)

    pdf.add_page()

    # Normalize text
    text = markdown_text.replace("\t", " ").replace("\r", "")

    # ✅ FULL Unicode punctuation normalization
    unicode_replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",     # en dash
        "—": "-",     # em dash
        "-": "-",     # non-breaking hyphen (THIS was your crash)
        "•": "*",
        "→": "->",
        "←": "<-",
        "≥": ">=",
        "≤": "<=",
    }

    for bad, good in unicode_replacements.items():
        text = text.replace(bad, good)

    # ✅ write() NEVER fails due to width
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue
        pdf.write(6, line + "\n")

    return pdf.output(dest="S").encode("latin1")


# --- MAIN UI ---
st.title("🤖 djGPT")
st.caption("Cloud Multi-Model Comparator (Text + Image · Debate · Streaming · Scoring · PDF)")

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            payload = msg["content"]
            answers = payload.get("answers", {})
            stats = payload.get("stats", {})
            baseline = payload.get("baseline")
            mode = payload.get("mode")
            winner_scores = payload.get("winner_scores", {})
            judge_analysis = payload.get("judge_analysis")

            if mode:
                st.markdown(f"**Mode:** {mode}")
            render_comparison(answers, stats, baseline, highlight_diffs)

            if winner_scores:
                st.markdown("**Winner scoring (consensus):**")
                score_str = ", ".join(
                    f"{name}: {score*100:.1f}%"
                    for name, score in sorted(winner_scores.items(), key=lambda x: x[1], reverse=True)
                )
                st.markdown(score_str)

            if judge_analysis:
                st.markdown("**Judge analysis (differences & contradictions):**")
                st.markdown(judge_analysis)
        else:
            st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Ask your question")

if prompt:
    if not selected_models:
        st.error("Select at least one model.")
        st.stop()

    if looks_like_code(prompt) and not any(m in CODER_MODELS for m in selected_models):
        st.info(
            "This looks like a coding question. "
            "Consider including a code-specialised model "
            "(e.g. qwen3-coder:480b-cloud)."
        )

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answers: Dict[str, str] = {}
    stats: Dict[str, Dict[str, Any]] = {}

    # Prepare image path once
    image_path = prepare_image_file(uploaded_image)

    with st.chat_message("assistant"):
        if conversation_mode == "Debate":
            # -------- DEBATE MODE WITH STREAMING --------
            if not baseline_model:
                st.error("Baseline model must be selected for Debate mode.")
                st.stop()

            st.markdown(f"**Mode:** Debate (baseline: {baseline_model})")

            # Round 1: baseline initial answer (streaming)
            st.markdown("**Baseline initial answer:**")
            base_placeholder = st.empty()
            base_text, base_latency = call_model(
                baseline_model,
                prompt,
                image_path=image_path,
                system_hint=None,
                stream=True,
                placeholder=base_placeholder,
            )
            answers[baseline_model] = base_text
            stats[baseline_model] = {
                "latency_ms": base_latency,
                "tokens": approx_tokens(base_text),
            }

            # Round 2: critiques from other models (streaming)
            critiques: Dict[str, str] = {}
            if len(selected_models) > 1:
                st.markdown("**Critiques from other models:**")
                crit_cols = st.columns(len([m for m in selected_models if m != baseline_model]))
                idx = 0
                for model in selected_models:
                    if model == baseline_model:
                        continue
                    with crit_cols[idx]:
                        st.markdown(f"**{model} (critique)**")
                        ph = st.empty()
                        crit_prompt = (
                            f"The user asked:\n{prompt}\n\n"
                            f"The baseline model {baseline_model} answered:\n{base_text}\n\n"
                            "Your task is to critically evaluate this answer. "
                            "Point out what is correct, what is missing or wrong, "
                            "and how you would improve it. Be concise but specific."
                        )
                        crit_text, crit_latency = call_model(
                            model,
                            crit_prompt,
                            image_path=image_path,
                            system_hint="You are acting as a debate participant.",
                            stream=True,
                            placeholder=ph,
                        )
                        critiques[model] = crit_text
                        answers[model] = crit_text
                        stats[model] = {
                            "latency_ms": crit_latency,
                            "tokens": approx_tokens(crit_text),
                        }
                    idx += 1

            # Round 3: baseline rebuttal (streaming)
            st.markdown(f"**{baseline_model} rebuttal:**")
            rebut_placeholder = st.empty()
            rebut_prompt = (
                f"The user asked:\n{prompt}\n\n"
                f"Your original answer was:\n{base_text}\n\n"
                "Other models critiqued your answer as follows:\n"
            )
            for m, c in critiques.items():
                rebut_prompt += f"\n- {m}: {c}\n"

            rebut_prompt += (
                "\nRespond with a concise rebuttal or clarification. "
                "Address valid points, correct your mistakes if any, "
                "and provide a final improved answer."
            )

            rebut_text, rebut_latency = call_model(
                baseline_model,
                rebut_prompt,
                image_path=image_path,
                system_hint="You are closing the debate with a final response.",
                stream=True,
                placeholder=rebut_placeholder,
            )
            rebut_name = f"{baseline_model} (Rebuttal)"
            answers[rebut_name] = rebut_text
            stats[rebut_name] = {
                "latency_ms": rebut_latency,
                "tokens": approx_tokens(rebut_text),
            }

        else:
            # -------- NON-DEBATE MODES (no streaming) --------
            st.markdown(f"**Mode:** {conversation_mode}")
            cols = st.columns(len(selected_models))
            for col, model in zip(cols, selected_models):
                with col:
                    with st.container(border=True):
                        st.markdown(
                            f"<div class='model-header'>{model}</div>",
                            unsafe_allow_html=True
                        )
                        # Different hints depending on mode
                        system_hint = None
                        if conversation_mode == "Critique baseline" and baseline_model:
                            if model == baseline_model:
                                system_hint = None
                            else:
                                system_hint = (
                                    f"You are critiquing the baseline model {baseline_model}. "
                                    "Point out strengths, weaknesses, missing details, "
                                    "and correctness issues in its answer."
                                )
                        ph = st.empty()
                        ans_text, lat = call_model(
                            model,
                            prompt,
                            image_path=image_path,
                            system_hint=system_hint,
                            stream=False,
                            placeholder=ph,
                        )
                        answers[model] = ans_text
                        stats[model] = {
                            "latency_ms": lat,
                            "tokens": approx_tokens(ans_text),
                        }

        # --- Winner scoring (consensus) ---
        winner_scores = compute_winner_scores(answers)

        st.markdown("**Winner scoring (consensus-based):**")
        scoreboard = sorted(winner_scores.items(), key=lambda x: x[1], reverse=True)
        score_str = ", ".join(
            f"{name}: {score*100:.1f}%"
            for name, score in scoreboard
        )
        st.markdown(score_str)

        # --- Judge analysis: differences + contradictions + winner ---
        judge_model = baseline_model or list(answers.keys())[0]
        judge_analysis = generate_analysis(judge_model, prompt, answers, baseline_model)

        if judge_analysis:
            st.markdown("**Judge analysis (differences & contradictions):**")
            st.markdown(judge_analysis)

        # --- Export: Markdown & PDF ---
        md = build_markdown_export(
            prompt=prompt,
            mode=conversation_mode,
            baseline=baseline_model,
            answers=answers,
            stats=stats,
            winner_scores=winner_scores,
            judge_analysis=judge_analysis,
        )
        st.download_button(
            "⬇️ Download comparison (Markdown)",
            data=md,
            file_name="ollama_nexus_comparison.md",
            mime="text/markdown",
        )

        if HAS_FPDF:
            try:
                pdf_bytes = build_pdf_export(md)
                if pdf_bytes:
                    st.download_button(
                        "⬇️ Download debate as PDF slides",
                        data=pdf_bytes,
                        file_name="ollama_nexus_debate.pdf",
                        mime="application/pdf",
                    )
            except Exception as e:
                st.warning(f"PDF export failed safely: {e}")

        else:
            st.caption("Install `fpdf2` to enable PDF export: `pip install fpdf2`")

    # Save assistant turn
    st.session_state.last_comparison = {
        "prompt": prompt,
        "answers": answers,
        "stats": stats,
        "baseline": baseline_model,
        "mode": conversation_mode,
        "winner_scores": winner_scores,
        "judge_analysis": judge_analysis,
    }
    st.session_state.messages.append({
        "role": "assistant",
        "content": st.session_state.last_comparison,
    })


