# WildTableBench

**WildTableBench** is a multimodal benchmark for evaluating vision-language models on real-world table understanding. It contains **928 questions** across **402 images** spanning 5 categories and 17 subtypes.

📄 Paper: [WildTableBench: Benchmarking Multimodal Foundation Models on Table Understanding In the Wild](https://arxiv.org/abs/2605.01018)  
🤗 Dataset: [jzhuang/WildTableBench](https://huggingface.co/datasets/jzhuang/WildTableBench)
🌐 Project Page: https://hjzhe.github.io/WildTableBench/

## Dataset

| Split | Questions | Images |
|-------|-----------|--------|
| Full  | 928       | 402    |

**Categories:**

| ID | Name | Description |
|----|------|-------------|
| C1 | Cell-Level | Cell lookup, transcription, structural understanding |
| C2 | Numerical | Arithmetic, conditional, multi-step calculation |
| C3 | Verification | Fact-checking against table content |
| C4 | Hypothesis | Trend analysis, comparison, inference |
| C5 | Color | Color-based reasoning and visual attribute reading |

---

## Evaluation

### 1. Download the dataset

```bash
# Option A: Hugging Face CLI
huggingface-cli download jzhuang/WildTableBench --repo-type dataset --local-dir hf_wild

# Option B: Python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="jzhuang/WildTableBench", repo_type="dataset", local_dir="hf_wild")
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set API keys

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Gemini
export GEMINI_API_KEY=AIza...

# OpenRouter (single key, access to 200+ models)
export OPENROUTER_API_KEY=sk-or-...

# Local model (vLLM / Ollama)
export LOCAL_MODEL_BASE_URL=http://localhost:8000
export LOCAL_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
```

### 4. Run evaluation

```bash
# OpenAI GPT-4o
python evaluate.py --models gpt4o

# Gemini 2.0 Flash
python evaluate.py --models gemini_2_flash

# Multiple models
python evaluate.py --models gpt4o gemini_2_flash claude_sonnet

# Any OpenRouter model ID
python evaluate.py --models openai/gpt-4o-mini
python evaluate.py --models meta-llama/llama-3.2-90b-vision-instruct

# Local model served via vLLM or Ollama
python evaluate.py --models local
```

Results are saved to `evaluation_results.json` and a summary is printed at the end.

---

## Supported Models

The following shortcuts are available out of the box:

| Name | Model |
|------|-------|
| `gpt4o` | GPT-4o (OpenAI or OpenRouter) |
| `gpt4o_mini` | GPT-4o mini |
| `gpt_o3` | o3 |
| `gemini_2_flash` | Gemini 2.0 Flash |
| `gemini_2_5_pro` | Gemini 2.5 Pro |
| `gemini_2_5_flash` | Gemini 2.5 Flash |
| `claude_sonnet` | Claude Sonnet (via OpenRouter) |
| `claude_opus` | Claude Opus (via OpenRouter) |
| `local` | Any locally served model (vLLM, Ollama, etc.) |

You can also pass any **OpenRouter model ID** directly, e.g.:
```bash
python evaluate.py --models qwen/qwen2.5-vl-72b-instruct
```
Full list: https://openrouter.ai/models

---

## Adding a New Model

Edit `model_eva_api.py`:

```python
def run_my_model(question: str, image_data=None) -> dict:
    # call your API and return {"content": "...", "usage": None}
    ...

SUPPORTED_MODELS["my_model"] = run_my_model
```

Then run:
```bash
python evaluate.py --models my_model
```

---

## CLI Reference

```
python evaluate.py --help

--data FILE           Path to metadata.csv (default: hf_wild/metadata.csv)
--images-dir DIR      Base directory for images
--models MODEL [...]  Model names or OpenRouter IDs
--prompt-styles       cot (default) or direct
--output FILE         Output JSON path (default: evaluation_results.json)
--limit N             Evaluate only the first N questions
--only-class ID       Evaluate one category only (e.g. C2)
--judge-model MODEL   LLM used to judge answers (default: gpt4o)
--concurrent-models N Run N model calls in parallel per question
--merge FILE [...]    Merge multiple result JSONs (use with -o)
```

---

## Citation

```bibtex
@article{huang2025wildtablebench,
  title   = {WildTableBench: Benchmarking Multimodal Foundation Models on Table Understanding In the Wild},
  author  = {Junzhe Huang and Xiaoxiao Sun and Yan Yang and Yuxuan Hou and Ruotian Zhang and Sirui Li and Hehe Fan and Serena Yeung-Levy and Xin Yu},
  journal = {arXiv preprint arXiv:2605.01018},
  year    = {2025},
}
```
