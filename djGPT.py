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

st.markdown(
    """
<style>
.block-container { padding-top: 1.5rem; }
.model-header {
    font-weight: bold;
    color: #4DA6FF;
    border-bottom: 2px solid #333;
    padding-bottom: 5px;
    margin-bottom: 8px;
}
.diff-added {
    background-color: rgba(239,68,68,0.18);
    border-radius: 4px;
    padding: 0 3px;
}
.phase-heading {
    font-weight: 600;
    margin-top: 8px;
    margin-bottom: 4px;
}
.final-answer {
    background-color: rgba(34,197,94,0.06);
    border-left: 4px solid #22c55e;
    padding: 12px;
    border-radius: 6px;
}
.small-caption { color: #9ca3af; font-size: 0.9rem; }
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
    "gemma3:27b-cloud",
    "deepseek-v3.1:671b-cloud",
    "qwen3-vl:235b-cloud",
    "qwen3-coder:480b-cloud",
    "ministral-3:14b-cloud",
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

if "last_comparison" not in st.session_state:
    st.session_state.last_comparison = None

# Flag and storage for "collapse after debate"
if "collapse_after_debate" not in st.session_state:
    st.session_state.collapse_after_debate = False

if "debate_artifacts" not in st.session_state:
    st.session_state.debate_artifacts = {}

# ---------------- Sidebar / Settings ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    backend_mode = st.radio(
        "Backend",
        ["Cloud (ollama.com)", "Local Ollama (localhost)"],
        index=0,
        help="Cloud uses your Ollama Cloud key; Local talks to localhost:11434 (only for local runs).",
    )

    uploaded_image = st.file_uploader("📷 Attach image (optional)", type=["png", "jpg", "jpeg", "webp"])

    if uploaded_image:
        st.info("Image uploaded — vision-capable models will be highlighted in the list below.")

    available_models = [m for m in ALL_MODELS if supports_vision(m)] if uploaded_image else ALL_MODELS

    selected_models = st.multiselect(
        "Select models",
        options=available_models,
        default=available_models[:3],
        help="Choose which models to query in parallel.",
    )

    conversation_mode = st.radio(
        "Conversation mode",
        options=["Independent answers", "Critique baseline", "Debate"],
        index=0,
        help=(
            "Independent answers: each model replies to the prompt.\n"
            "Critique baseline: models critique a baseline model's answer.\n"
            "Debate: baseline answers, others critique, baseline rebuts; loop until approval."
        ),
    )

    baseline_model = st.selectbox("Baseline model (optional)", options=[""] + selected_models)
    baseline_model = baseline_model if baseline_model else None

    highlight_diffs = st.checkbox("Highlight differences vs baseline", value=True)

    st.divider()
    st.subheader("Debate options")
    max_debate_rounds = st.number_input("Max internal debate rounds", min_value=1, max_value=6, value=3, step=1)
    approval_threshold = st.slider("Approval requirement (percent of critics)", 50, 100, 100, step=10)
    st.caption("During debate, critics will return APPROVE / REJECT. Loop until threshold satisfied or max rounds reached.")

    # st.divider()
    # st.caption("Quick model capabilities")
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
        st.session_state.plain_history = []
        st.session_state.last_comparison = None
        st.session_state.collapse_after_debate = False
        st.session_state.debate_artifacts = {}
        st.rerun()

# ---------------- Client initialization ----------------
if backend_mode.startswith("Cloud"):
    if not OLLAMA_API_KEY:
        st.error("Cloud backend selected but OLLAMA_API_KEY is missing. Add it to Streamlit Secrets or environment.")
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


# ---------------- Model call (with history support) ----------------
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
            try:
                full_text = getattr(r.message, "content", r["message"]["content"])
            except Exception:
                full_text = str(r)
            if placeholder is not None:
                placeholder.markdown(full_text)
    except Exception as e:
        full_text = f"❌ {e}"
        if placeholder is not None:
            placeholder.markdown(full_text)

    latency_ms = (time.perf_counter() - start) * 1000
    return full_text, latency_ms


# ---------------- Analysis & rendering helpers ----------------
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

    analysis_prompt = "You are comparing answers from multiple models.\n\n"
    analysis_prompt += f"User question:\n{prompt}\n\nHere are the answers:\n"
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
        messages_history=None,
    )
    return text


def render_comparison(
    answers: Dict[str, str],
    stats: Dict[str, Dict[str, Any]],
    baseline: Optional[str],
    highlight: bool,
):
    if not answers:
        return

    model_names = list(answers.keys())
    base_name = baseline if baseline in model_names else model_names[0]
    base_text = answers.get(base_name, "")

    cols = st.columns(len(model_names))
    for col, name in zip(cols, model_names):
        text = answers[name]
        s = stats.get(name, {})
        with col:
            with st.container(border=True):
                st.markdown(f"<div class='model-header'>{name}</div>", unsafe_allow_html=True)
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


# ---------------- Markdown & PDF export ----------------
def build_markdown_export(
    prompt: str,
    mode: str,
    baseline: Optional[str],
    answers: Dict[str, str],
    stats: Dict[str, Dict[str, Any]],
    winner_scores: Dict[str, float],
    judge_analysis: Optional[str],
) -> str:
    lines: List[str] = []
    lines.append("# djGPT Comparison\n")
    lines.append(f"**Prompt:** {prompt}")
    lines.append(f"**Mode:** {mode}")
    if baseline:
        lines.append(f"**Baseline model:** `{baseline}`")
    lines.append("")

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


def build_pdf_export(markdown_text: str) -> Optional[bytes]:
    if not HAS_FPDF:
        return None

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=11)
    pdf.add_page()

    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "\u2011": "-",
        "•": "*",
        "→": "->",
        "←": "<-",
    }
    text = markdown_text.replace("\t", " ").replace("\r", "")
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue
        pdf.write(6, line + "\n")

    return pdf.output(dest="S").encode("latin1")


# ---------------- Main UI ----------------
st.title("🤖 djGPT")
st.caption("Cloud Multi-Model Comparator (Text + Image · Debate · Streaming · Scoring · PDF)")

# If collapse flag is set and we have stored artifacts, show just final + expander
if st.session_state.collapse_after_debate and st.session_state.debate_artifacts:
    artifacts = st.session_state.debate_artifacts
    # final chosen model and answer
    best_model = artifacts.get("best_model")
    best_score = artifacts.get("best_score")
    answers = artifacts.get("answers", {})
    stats = artifacts.get("stats", {})
    # show only the final answer prominently
    st.markdown("### Final (consensus) answer")
    st.markdown(f"<div class='final-answer'><strong>{best_model}</strong> — {best_score*100:.1f}%<br><br>{answers.get(best_model, '')}</div>", unsafe_allow_html=True)

    # hidden artifacts inside an expander
    with st.expander("Show debate artifacts (baseline, critiques, rebuttals, approvals)", expanded=False):
        st.markdown("#### All artifacts")
        st.markdown("**Saved artifacts from debate:**")
        # pretty-print answers
        for name, text in answers.items():
            st.markdown(f"**{name}**")
            st.markdown(text)
        st.markdown("---")
        if "judge_analysis" in artifacts and artifacts["judge_analysis"]:
            st.markdown("#### Judge analysis")
            st.markdown(artifacts["judge_analysis"])
        if "winner_scores" in artifacts:
            st.markdown("#### Winner scores")
            for nm, s in artifacts["winner_scores"].items():
                st.markdown(f"- {nm}: {s*100:.1f}%")

    st.stop()  # stop further rendering (so the rest of the UI isn't shown until next user action)

# Display prior UI history (compact)
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
                score_str = ", ".join(f"{name}: {score*100:.1f}%" for name, score in sorted(winner_scores.items(), key=lambda x: x[1], reverse=True))
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
        st.info("This looks like a coding question. Consider adding a code-specialised model (e.g. qwen3-coder:480b-cloud).")

    # Save user message to UI history and plain history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.plain_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    answers: Dict[str, str] = {}
    stats: Dict[str, Dict[str, Any]] = {}
    image_path = prepare_image_file(uploaded_image)

    # Truncate history to items
    MAX_ITEMS = 30
    history_to_send = st.session_state.plain_history[-MAX_ITEMS:] if st.session_state.plain_history else []

    # Prepare placeholders to show progress
    baseline_section = st.empty()
    critiques_section = st.empty()
    rebuttal_section = st.empty()
    approval_section = st.empty()

    # --- Debate mode with visible progress per phase ---
    if conversation_mode == "Debate":
        if not baseline_model:
            st.error("Baseline model must be selected for Debate mode.")
            st.stop()

        # Show phase heading immediately
        baseline_section.markdown(f"<div class='phase-heading'>Baseline initial answer (from {baseline_model})</div>", unsafe_allow_html=True)
        base_text, base_latency = call_model(
            baseline_model,
            prompt,
            image_path=image_path,
            stream=False,
            placeholder=None,
            messages_history=history_to_send,
        )
        baseline_section.markdown(base_text)
        st.session_state.plain_history.append({"role": "assistant", "content": base_text})

        final_baseline_answer = base_text
        final_critiques: Dict[str, str] = {}
        final_rebuttal = ""
        rounds_performed = 0
        required_approval_ratio = approval_threshold / 100.0
        critics = [m for m in selected_models if m != baseline_model]

        if not critics:
            final_critiques = {}
            final_rebuttal = final_baseline_answer
            rounds_performed = 0
        else:
            for round_idx in range(1, max_debate_rounds + 1):
                rounds_performed = round_idx

                # Critiques
                critiques_section.markdown(f"<div class='phase-heading'>Round {round_idx}: Critiques</div>", unsafe_allow_html=True)
                critiques: Dict[str, str] = {}
                crit_cols = critiques_section.columns(len(critics))
                for idx, model in enumerate(critics):
                    crit_prompt = (
                        f"The user asked:\n{prompt}\n\n"
                        f"The baseline ({baseline_model}) answered:\n{final_baseline_answer}\n\n"
                        "Critique this answer: what is correct, missing, or wrong? Be concise."
                    )
                    crit_text, crit_latency = call_model(
                        model,
                        crit_prompt,
                        image_path=image_path,
                        system_hint="You are a debate participant providing a critique.",
                        stream=False,
                        placeholder=None,
                        messages_history=st.session_state.plain_history[-MAX_ITEMS:],
                    )
                    critiques[model] = crit_text
                    st.session_state.plain_history.append({"role": "assistant", "content": crit_text})
                    with crit_cols[idx]:
                        st.markdown(f"**{model} (critique)**")
                        st.markdown(crit_text)

                # Baseline rebuttal
                rebuttal_section.markdown(f"<div class='phase-heading'>Round {round_idx}: Baseline rebuttal</div>", unsafe_allow_html=True)
                rebut_prompt = (
                    f"You are the baseline model ({baseline_model}). The user asked:\n{prompt}\n\n"
                    f"Your previous answer was:\n{final_baseline_answer}\n\n"
                    "Other models gave these critiques (listed):\n"
                )
                for m, c in critiques.items():
                    rebut_prompt += f"\n- {m}: {c}\n"
                rebut_prompt += "\nProvide a concise rebuttal and improved final answer."

                rebut_text, rebut_latency = call_model(
                    baseline_model,
                    rebut_prompt,
                    image_path=image_path,
                    system_hint="You are closing this debate round with a rebuttal and improved answer.",
                    stream=False,
                    placeholder=None,
                    messages_history=st.session_state.plain_history[-MAX_ITEMS:],
                )
                final_rebuttal = rebut_text
                st.session_state.plain_history.append({"role": "assistant", "content": rebut_text})
                rebuttal_section.markdown(rebut_text)

                # Critics approve/reject
                approval_section.markdown(f"<div class='phase-heading'>Round {round_idx}: Critics approval</div>", unsafe_allow_html=True)
                approvals: Dict[str, Dict[str, str]] = {}
                appr_cols = approval_section.columns(len(critics))
                for idx, model in enumerate(critics):
                    approval_prompt = (
                        f"The user asked:\n{prompt}\n\n"
                        f"The baseline ({baseline_model})'s rebuttal / improved answer is:\n{final_rebuttal}\n\n"
                        "Do you APPROVE this rebuttal as a satisfactory final answer? "
                        "Answer in one line starting with APPROVE or REJECT, then a brief reason."
                    )
                    appr_text, appr_lat = call_model(
                        model,
                        approval_prompt,
                        image_path=image_path,
                        system_hint="Answer with APPROVE or REJECT followed by a short reason.",
                        stream=False,
                        placeholder=None,
                        messages_history=st.session_state.plain_history[-MAX_ITEMS:],
                    )
                    first_line = (appr_text or "").strip().splitlines()[0] if appr_text else ""
                    token = first_line.strip().upper().split()[0] if first_line else "REJECT"
                    decision = "APPROVE" if token.startswith("APP") else "REJECT"
                    approvals[model] = {"decision": decision, "raw": appr_text or ""}
                    st.session_state.plain_history.append({"role": "assistant", "content": appr_text or ""})
                    with appr_cols[idx]:
                        st.markdown(f"**{model} approval**: {decision}")
                        st.markdown(appr_text or "No reply")

                approved_count = sum(1 for v in approvals.values() if v["decision"] == "APPROVE")
                total_critics = len(critics)
                approval_ratio = approved_count / total_critics if total_critics else 1.0
                approval_summary = f"Approved: {approved_count}/{total_critics} ({approval_ratio*100:.0f}%) — required {required_approval_ratio*100:.0f}%"
                approval_section.markdown(f"**Approval summary:** {approval_summary}")

                final_critiques = critiques

                if approval_ratio >= required_approval_ratio:
                    # threshold satisfied -> finish debate loop
                    break
                else:
                    # prepare for next round
                    final_baseline_answer = final_rebuttal
                    continue

        # Build answers/stats for export and scoring
        answers[baseline_model] = final_baseline_answer
        stats[baseline_model] = {"latency_ms": 0.0, "tokens": approx_tokens(final_baseline_answer)}
        for m, ct in final_critiques.items():
            answers[m] = ct
            stats[m] = {"latency_ms": 0.0, "tokens": approx_tokens(ct)}
        rebut_name = f"{baseline_model} (Rebuttal)"
        answers[rebut_name] = final_rebuttal
        stats[rebut_name] = {"latency_ms": 0.0, "tokens": approx_tokens(final_rebuttal)}

        # Winner scoring (consensus)
        winner_scores = compute_winner_scores(answers)
        if winner_scores:
            sorted_scores = sorted(winner_scores.items(), key=lambda x: x[1], reverse=True)
            best_model, best_score = sorted_scores[0]
        else:
            best_model, best_score = baseline_model, 1.0

        # Judge analysis (optional)
        judge_model = baseline_model or (list(answers.keys())[0] if answers else None)
        judge_analysis = generate_analysis(judge_model, prompt, answers, baseline_model) if judge_model else None

        # Save artifacts to session_state so they can be shown inside an expander after collapse
        st.session_state.debate_artifacts = {
            "answers": answers,
            "stats": stats,
            "winner_scores": winner_scores,
            "best_model": best_model,
            "best_score": best_score,
            "judge_analysis": judge_analysis,
            "approval_summary": approval_summary,
            "rounds_performed": rounds_performed,
        }

        # Set collapse flag so next render shows only the final answer + expander
        st.session_state.collapse_after_debate = True

        # Rerun so the UI refreshes and shows only the final answer
        st.rerun()

    else:
        # Non-debate modes (visible behavior)
        st.markdown(f"**Mode:** {conversation_mode}")
        cols = st.columns(len(selected_models))
        for col, model in zip(cols, selected_models):
            with col:
                with st.container(border=True):
                    st.markdown(f"<div class='model-header'>{model}</div>", unsafe_allow_html=True)
                    system_hint = None
                    if conversation_mode == "Critique baseline" and baseline_model and model != baseline_model:
                        system_hint = f"You are critiquing the baseline model {baseline_model}."
                    ph = st.empty()
                    ans_text, lat = call_model(
                        model,
                        prompt,
                        image_path=image_path,
                        system_hint=system_hint,
                        stream=False,
                        placeholder=ph,
                        messages_history=st.session_state.plain_history[-MAX_ITEMS:] if st.session_state.plain_history else None,
                    )
                    answers[model] = ans_text
                    stats[model] = {"latency_ms": lat, "tokens": approx_tokens(ans_text)}
                    st.session_state.plain_history.append({"role": "assistant", "content": ans_text})

        # After non-debate: winner scoring & judge analysis
        winner_scores = compute_winner_scores(answers)
        st.markdown("**Winner scoring (consensus-based):**")
        scoreboard = sorted(winner_scores.items(), key=lambda x: x[1], reverse=True)
        score_str = ", ".join(f"{name}: {score*100:.1f}%" for name, score in scoreboard)
        st.markdown(score_str)

        judge_model = baseline_model or (list(answers.keys())[0] if answers else None)
        judge_analysis = generate_analysis(judge_model, prompt, answers, baseline_model) if judge_model else None
        if judge_analysis:
            st.markdown("**Judge analysis (differences & contradictions):**")
            st.markdown(judge_analysis)

    # --- Build exports using current answers/stats/judge analysis/winner_scores
    if "winner_scores" not in locals():
        winner_scores = compute_winner_scores(answers)
    judge_model = baseline_model or (list(answers.keys())[0] if answers else None)
    if 'judge_analysis' not in locals() or locals().get('judge_analysis') is None:
        judge_analysis = generate_analysis(judge_model, prompt, answers, baseline_model) if judge_model else None

    md = build_markdown_export(
        prompt=prompt,
        mode=conversation_mode,
        baseline=baseline_model,
        answers=answers,
        stats=stats,
        winner_scores=winner_scores,
        judge_analysis=judge_analysis,
    )
    st.download_button("⬇️ Download comparison (Markdown)", data=md, file_name="djgpt_comparison.md", mime="text/markdown")

    if HAS_FPDF:
        try:
            pdf_bytes = build_pdf_export(md)
            if pdf_bytes:
                st.download_button("⬇️ Download debate as PDF", data=pdf_bytes, file_name="djgpt_debate.pdf", mime="application/pdf")
        except Exception as e:
            st.warning(f"PDF export failed safely: {e}")
    else:
        st.caption("Install `fpdf2` to enable PDF export: `pip install fpdf2`")

    # Save assistant turn (for UI/history)
    st.session_state.last_comparison = {
        "prompt": prompt,
        "answers": answers,
        "stats": stats,
        "baseline": baseline_model,
        "mode": conversation_mode,
        "winner_scores": winner_scores,
        "judge_analysis": judge_analysis,
    }
    st.session_state.messages.append({"role": "assistant", "content": st.session_state.last_comparison})
