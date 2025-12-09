# djGPT

djGPT is a Streamlit-based multi-model AI comparison and debate platform that lets you run the **same question** across multiple LLMs and analyze how different models respond, reason, critique, and improve answers.

It supports:
- Ollama **Cloud models**
- **Local Ollama** running on your machine
- Text and image-based questions
- Debate-style reasoning with rebuttals
- Local file (RAG) querying across your laptop
- Markdown and PDF export of results

djGPT is built for learning, evaluation, and serious model analysis.

---

## Features

### Multi-Model Comparison
- Ask one question to multiple models simultaneously
- Side-by-side responses
- Difference highlighting relative to a baseline
- Consensus-based winner scoring

### Debate Mode
- Baseline model provides an initial answer
- Other models critique it
- Baseline responds with a final rebuttal
- Useful for spotting gaps, errors, and weak reasoning

### Cloud and Local Execution
- Toggle between:
  - **Ollama Cloud** (via API key)
  - **Local Ollama** (`http://localhost:11434`)
- Same UI and features across both modes

### Image-Based Questions (Vision Models)
- Upload images and ask questions about them
- Model list automatically filters to vision-capable models

### Local Files as Knowledge (RAG)
- Index any folder on your laptop
- Queries search local `.txt` and `.md` files
- Relevant excerpts are fed into the model prompt
- Works with both cloud and local backends

### Export
- Download results as:
  - Markdown
  - PDF (safe, ASCII-normalized export)

---

## Tech Stack

- **Python**
- **Streamlit** – UI and interaction
- **Ollama Python SDK** – Cloud & local models
- **fpdf2** – PDF export
- **python-dotenv** – Environment variables

---

## Requirements

- Python 3.10+
- Ollama installed (for local mode)
- Ollama Cloud account + API key (for cloud mode)

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/dumspterdj/djGPT.git
cd djGPT
````

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
OLLAMA_API_KEY=your_ollama_cloud_api_key_here
```

> Required only for **Cloud mode**.
> Local Ollama does not require an API key.

---

## Running djGPT

```bash
streamlit run app_ollama_cloud_fixed.py
```

The app will open in your browser.

---

## Using Local Ollama

1. Install Ollama from [https://ollama.com](https://ollama.com)
2. Start Ollama normally
3. In the sidebar, select:

   ```
   Backend → Local Ollama (localhost)
   ```

Local endpoint used:

```
http://localhost:11434
```

---

## Using Local Files (RAG Mode) {In Devlopment}

1. Enable **“Use local files as context (RAG)”** in the sidebar
2. Enter a folder path, for example:

   ```
   C:\Users\dhruv\Documents
   ```
3. Click **Index folder**
4. Ask a question

djGPT will:

* Search indexed files
* Extract relevant snippets
* Inject them into the model prompts

---

## Supported Models

Examples include:

* `gpt-oss:120b-cloud`
* `gemma3:27b-cloud`
* `deepseek-v3.1:671b-cloud`
* `qwen3-vl:235b-cloud` (vision)
* `qwen3-coder:480b-cloud` (code)

Model availability depends on Ollama Cloud or your local setup.

---

## Limitations

* Local file indexing currently supports `.txt` and `.md` files
* PDF export uses ASCII-normalized text for stability
* Performance depends on network speed (cloud) or hardware (local)

---

## Use Cases

* Comparing reasoning across LLMs
* Studying answer quality and bias
* Academic and technical research
* Exam preparation and revision
* Private knowledge querying using local files

---

## License

MIT License

---

## Author

Built by **DumpsterDJ**
Created for experimentation, learning, and serious AI evaluation.

```
