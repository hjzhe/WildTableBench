"""
Evaluation script for WildTableBench.

Dataset: hf_wild/metadata.csv  (downloaded from https://huggingface.co/datasets/jzhuang/WildTableBench)

Quick start
-----------
  # OpenAI
  export OPENAI_API_KEY=sk-...
  python evaluate.py --models gpt4o

  # Gemini
  export GEMINI_API_KEY=AIza...
  python evaluate.py --models gemini_2_flash

  # OpenRouter (single key, access to 200+ models)
  export OPENROUTER_API_KEY=sk-or-...
  python evaluate.py --models openai/gpt-4o google/gemini-2.0-flash-001

  # Local vLLM / Ollama
  export LOCAL_MODEL_BASE_URL=http://localhost:8000
  export LOCAL_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
  python evaluate.py --models local

See model_eva_api.py for full backend documentation and how to add new models.
"""

import os
import json
import base64
import csv
import re
from typing import Dict, List, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

from dotenv import load_dotenv
load_dotenv()

from model_eva_api import (
    SUPPORTED_MODELS,
    run_openrouter_model,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_FILE = "hf_wild/metadata.csv"
DEFAULT_OUTPUT_FILE = "evaluation_results.json"

# Judge model: used to verify model answers against ground truth.
# Set to any key in SUPPORTED_MODELS or an OpenRouter model ID (e.g. "openai/gpt-4o").
JUDGE_MODEL = "gpt4o"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPT_COT = """You are given a question that refers to an image. Solve this question step-by-step.

Provide your final answer in the following format: Final answer: [your concise answer]

### Question
{question}
"""

PROMPT_DIRECT = """Answer the following question about the image. Start your response with "Final answer: [your answer]" directly.

### Question
{question}"""

PROMPT_STYLES = {
    "cot": PROMPT_COT,
    "direct": PROMPT_DIRECT,
}

JUDGE_PROMPT = """You are a STRICT judge evaluating if a model's answer matches the ground truth.

Question: {question}
Ground Truth Answer: {ground_truth}

**Model's Final Answer:** (what the model gave after "Final answer:")
{model_final_answer}

**Model's Full Response:** (reasoning + answer, for reference when final answer is unclear)
{model_full_response}

**Instructions:** First judge correctness using **Model's Final Answer** only. If the final answer clearly matches (or clearly does not match) the ground truth, base your decision on that. Only when the final answer is ambiguous, missing, or you cannot tell—use **Model's Full Response** (reasoning) to decide. Same value = CORRECT; ignore formatting (bold, spaces, extra words).

STRICT Evaluation Rules:
1. **Numerical/currency answers**: If the model's stated value equals the ground truth (same number), answer CORRECT. Format does not matter.
   - "$151,830" = "151830" = "The revenue was $151,830." = "**$151,830**" (CORRECT)
   - Only if the value is different (e.g. $150,000 vs $151,830) answer INCORRECT.

2. **Text answers**: Must be semantically equivalent.
   - Case differences OK: "Apple" = "apple" (CORRECT)
   - Different content is INCORRECT: "Product A" ≠ "Product B"

3. **Boolean/Yes-No answers**: Semantically equivalent forms are OK.
   - "Yes" = "True" = "Correct" (CORRECT)
   - "No" = "False" = "Incorrect" (CORRECT)

4. **Lists**: All items must be present.

Is the model's answer CORRECT or INCORRECT?

Return ONLY one word: "CORRECT" or "INCORRECT"."""

JUDGE_PROMPT_CELL_LEVEL = """You are a STRICT judge evaluating if a model's transcription matches the ground truth.

Question: {question}
Ground Truth Answer: {ground_truth}

**Model's Final Answer:** (what the model gave after "Answer:")
{model_final_answer}

**Model's Full Response:** (reasoning + answer, for reference when final answer is unclear)
{model_full_response}

**Instructions:** First judge correctness using **Model's Final Answer** only. Follow this logic:
- If the final answer clearly matches the ground truth → CORRECT.
- If the final answer contains wrong values → INCORRECT immediately (do NOT check full response).
- If the final answer appears **partial** (fewer values than expected, seems truncated, but the values present are all correct) → use **Model's Full Response** to find the complete answer and judge based on that.
- If the final answer is missing or ambiguous → use **Model's Full Response** to decide.

STRICT Rules for Cell-Level Questions:
1. **All values must be present** — missing even one value = INCORRECT.
2. **Order must match** — if the question specifies top-to-bottom or left-to-right, the order must be correct.
3. **No extra values** — values not in the ground truth = INCORRECT.
4. **Separators are flexible** — commas / pipes / newlines / spaces are interchangeable.
5. **Number formatting is flexible** — "0.10" = "0.1", "1.00" = "1" (CORRECT).
6. **Case is flexible** — "apple" = "Apple" (CORRECT).
7. **Minor punctuation differences are OK** — trailing period, extra spaces (CORRECT).

Is the model's answer CORRECT or INCORRECT?

Return ONLY one word: "CORRECT" or "INCORRECT"."""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_annotations(data_path: str, images_dir: str = None) -> List[Dict]:
    """Load questions from metadata.csv.

    images_dir defaults to the directory containing data_path.
    """
    if images_dir is None:
        images_dir = os.path.dirname(os.path.abspath(data_path))
    return _load_csv(data_path, images_dir)


def _load_csv(path: str, images_dir: str) -> List[Dict]:
    annotations = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("question") or not row.get("ground_truth"):
                continue
            img_rel = row.get("file_name", "")
            img_path = os.path.join(images_dir, img_rel) if img_rel else ""
            annotations.append({
                "uuid":         row.get("uuid", ""),
                "image_paths":  [img_path] if img_path else [],
                "question":     row["question"].strip(),
                "answer":       row["ground_truth"].strip(),
                "question_idx": int(row.get("index", 1)),
                "class_id":     row.get("category", ""),
                "class_name":   row.get("category_name", ""),
                "subtype_id":   row.get("subtype_id", ""),
                "subtype_name": row.get("subtype_name", ""),
                "image_id":     row.get("image_id", ""),
            })
    return annotations


def load_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def _call_judge(prompt: str) -> str:
    fn = SUPPORTED_MODELS.get(JUDGE_MODEL)
    if fn is not None:
        response = fn(prompt)
    elif "/" in JUDGE_MODEL:
        response = run_openrouter_model(JUDGE_MODEL, prompt)
    else:
        raise ValueError(f"Unknown judge model: {JUDGE_MODEL!r}. "
                         "Use a key from SUPPORTED_MODELS or an OpenRouter model ID.")
    if isinstance(response, dict):
        response = response.get("content", "") or ""
    return (response or "").strip()


def judge_answer(
    question: str,
    ground_truth: str,
    full_response: str,
    final_answer: str = None,
    class_id: str = None,
) -> Tuple[bool, str]:
    r = (full_response or "").strip()
    if not r or r.startswith("Error") or r.startswith("ERROR"):
        return False, "error"

    model_final = (final_answer or "").strip() or "(none)"
    model_full = (full_response or "").strip()

    template = JUDGE_PROMPT_CELL_LEVEL if (class_id or "").startswith("C1") else JUDGE_PROMPT
    prompt = template.format(
        question=question,
        ground_truth=ground_truth,
        model_final_answer=model_final,
        model_full_response=model_full,
    )
    try:
        response_upper = _call_judge(prompt).upper()
        if "INCORRECT" in response_upper:
            return False, "llm"
        return "CORRECT" in response_upper, "llm"
    except Exception as e:
        print(f"Judge error: {e}")
        return False, "llm"


def extract_final_answer(response: str, class_id: str = None) -> str:
    """Extract the final answer from a model response.

    Takes the LAST "Final answer:" occurrence to avoid prompt template placeholders
    or hits inside <think> blocks. For C1 questions, captures everything after
    "Answer:" to preserve multi-line cell answers.
    """
    if (class_id or "").startswith("C1"):
        m = re.search(r"[Aa]nswer:\s*(.+)", response, re.DOTALL)
        if m:
            return re.sub(r"<\|[^>]*\|>", "", m.group(1)).strip()
        return ""

    matches = list(re.finditer(r"[Ff]inal\s+[Aa]nswer:\s*(.+?)(?=\n\s*\n|\Z)", response, re.DOTALL))
    if not matches:
        matches = list(re.finditer(r"[Ff]inal\s+[Aa]nswer:\s*(.+)", response, re.IGNORECASE))

    if matches:
        answer = matches[-1].group(1).strip()
        if answer.lower().startswith("<insert"):
            answer = matches[-2].group(1).strip() if len(matches) >= 2 else ""
        if answer:
            answer = re.sub(r"<\|[^>]*\|>", "", answer).strip()
            if re.search(r"Row\s*\d+", answer, re.I) or "**" in answer:
                nums = re.findall(r"[\d,]+\.?\d*|\.\d+", answer)
                if nums:
                    answer = nums[-1].replace(",", "")
            return answer.rstrip(".")

    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    answer = lines[-1] if lines else response.strip()
    return re.sub(r"<\|[^>]*\|>", "", answer).strip()


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------

def run_model(
    model_name: str,
    question: str,
    image_data: List[str],
    prompt_style: str = "cot",
    class_id: str = None,
) -> Tuple[str, str, dict]:
    """Run a model on one question. Returns (full_response, final_answer, usage).

    model_name: key in SUPPORTED_MODELS, or any OpenRouter model ID (contains "/").
    """
    template = PROMPT_STYLES.get(prompt_style, PROMPT_COT)
    prompt = template.format(question=question)

    try:
        fn = SUPPORTED_MODELS.get(model_name)
        if fn is not None:
            full_response = fn(prompt, image_data)
        elif "/" in model_name:
            full_response = run_openrouter_model(model_name, prompt, image_data)
        else:
            return f"ERROR: unknown model '{model_name}'", f"ERROR: unknown model '{model_name}'", None

        usage = None
        if isinstance(full_response, dict):
            usage = full_response.get("usage")
            full_response = full_response.get("content", "")

        final_answer = extract_final_answer(full_response, class_id=class_id)
        return full_response, final_answer, usage

    except Exception as e:
        print(f"  Model {model_name} error: {e}")
        return f"ERROR: {e}", f"ERROR: {e}", None


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _normalize_model_results(mr: Dict) -> Dict:
    """Migrate legacy flat model_results[model] = {correct, ...} to nested format."""
    out = {}
    for model, data in mr.items():
        if isinstance(data, dict) and "correct" in data and "full_response" in data:
            out[model] = {"cot": data}
        elif isinstance(data, dict):
            out[model] = data
        else:
            out[model] = {}
    return out


def run_evaluation(
    annotations: List[Dict],
    models: List[str],
    resume: bool = True,
    limit: int = None,
    only_new_models: bool = True,
    prompt_styles: List[str] = None,
    output_file: str = None,
    concurrent_models: int = 1,
    only_class: str = None,
) -> Dict:
    out_path = output_file or DEFAULT_OUTPUT_FILE
    if prompt_styles is None:
        prompt_styles = ["cot"]

    results = {"evaluations": [], "summary": {}}
    if resume and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        for e in results.get("evaluations", []):
            e["model_results"] = _normalize_model_results(e.get("model_results", {}))

    eval_index = {e["key"]: i for i, e in enumerate(results.get("evaluations", []))}

    filtered = [a for a in annotations if (a.get("class_id") or "") == only_class] \
        if only_class else list(annotations)

    to_new, to_partial = [], []
    for ann in filtered:
        key = f"{ann['uuid']}_{ann['question_idx']}"
        if key not in eval_index:
            to_new.append(ann)
        elif only_new_models:
            mr = results["evaluations"][eval_index[key]].get("model_results", {})
            missing = [(m, ps) for m in models for ps in prompt_styles if mr.get(m, {}).get(ps) is None]
            if missing:
                ann["missing_pairs"] = missing
                to_partial.append(ann)

    if limit is not None:
        to_new = to_new[:limit]

    print(f"Total questions: {len(annotations)}")
    if only_class:
        print(f"Filtered to class {only_class}: {len(filtered)}")
    print(f"Already evaluated: {len(eval_index) - len(to_partial)}")
    print(f"Needs new model results: {len(to_partial)}")
    print(f"New questions to evaluate: {len(to_new)}")
    print(f"Models: {models}")
    print(f"Prompt styles: {prompt_styles}")

    if to_partial:
        print(f"\n--- Updating {len(to_partial)} existing entries ---")
        for ann in tqdm(to_partial, desc="Updating"):
            key = f"{ann['uuid']}_{ann['question_idx']}"
            idx = eval_index[key]
            image_data = [load_image_base64(ann["image_paths"][0])] if ann.get("image_paths") else []
            for model, ps in ann["missing_pairs"]:
                print(f"\n  {model} [{ps}] for {key}...")
                full, final, usage = run_model(model, ann["question"], image_data, ps, ann.get("class_id"))
                correct, method = judge_answer(ann["question"], ann["answer"], full, final, ann.get("class_id"))
                if results["evaluations"][idx]["model_results"].get(model) is None:
                    results["evaluations"][idx]["model_results"][model] = {}
                entry = {"full_response": full, "final_answer": final, "correct": correct, "judge_method": method}
                if usage:
                    entry.update({k: usage[k] for k in ("prompt_tokens", "completion_tokens", "thought_token_count") if k in usage})
                results["evaluations"][idx]["model_results"][model][ps] = entry
                time.sleep(1)
            _print_result(results["evaluations"][idx])
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    if to_new:
        print(f"\n--- Evaluating {len(to_new)} new questions ---")
        for ann in tqdm(to_new, desc="Evaluating"):
            key = f"{ann['uuid']}_{ann['question_idx']}"
            image_data = [load_image_base64(ann["image_paths"][0])] if ann.get("image_paths") else []

            eval_result = {
                "key":           key,
                "uuid":          ann["uuid"],
                "question":      ann["question"],
                "ground_truth":  ann["answer"],
                "class_id":      ann.get("class_id", ""),
                "class_name":    ann.get("class_name", ""),
                "subtype_id":    ann.get("subtype_id", ""),
                "subtype_name":  ann.get("subtype_name", ""),
                "image_id":      ann.get("image_id", ""),
                "model_results": {},
            }

            tasks = [(m, ps) for m in models for ps in prompt_styles]

            if concurrent_models <= 1 or len(tasks) <= 1:
                for model in models:
                    eval_result["model_results"][model] = {}
                    for ps in prompt_styles:
                        print(f"\n  {model} [{ps}]...")
                        full, final, usage = run_model(model, ann["question"], image_data, ps, ann.get("class_id"))
                        correct, method = judge_answer(ann["question"], ann["answer"], full, final, ann.get("class_id"))
                        entry = {"full_response": full, "final_answer": final, "correct": correct, "judge_method": method}
                        if usage:
                            entry.update({k: usage[k] for k in ("prompt_tokens", "completion_tokens", "thought_token_count") if k in usage})
                        eval_result["model_results"][model][ps] = entry
                        time.sleep(1)
            else:
                for model in models:
                    eval_result["model_results"][model] = {}

                def _run_one(m, ps):
                    full, final, usage = run_model(m, ann["question"], image_data, ps, ann.get("class_id"))
                    correct, method = judge_answer(ann["question"], ann["answer"], full, final, ann.get("class_id"))
                    data = {"full_response": full, "final_answer": final, "correct": correct, "judge_method": method}
                    if usage:
                        data.update({k: usage[k] for k in ("prompt_tokens", "completion_tokens", "thought_token_count") if k in usage})
                    return m, ps, data

                with ThreadPoolExecutor(max_workers=min(concurrent_models, len(tasks))) as ex:
                    futures = {ex.submit(_run_one, m, ps): (m, ps) for m, ps in tasks}
                    for fut in as_completed(futures):
                        m, ps, data = fut.result()
                        eval_result["model_results"][m][ps] = data
                time.sleep(1)

            _print_result(eval_result)
            results["evaluations"].append(eval_result)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def _print_result(eval_result: dict):
    print("\n" + "=" * 60)
    q = eval_result.get("question", "")
    print(f"[{eval_result.get('key', '')}] {q[:200]}{'...' if len(q) > 200 else ''}")
    print(f"Ground truth: {eval_result.get('ground_truth', '')}")
    print("-" * 60)
    for model, by_style in eval_result.get("model_results", {}).items():
        for ps, d in (by_style.items() if isinstance(by_style, dict) else []):
            if isinstance(d, dict):
                ans = (d.get("final_answer") or "").strip() or "(none)"
                ok = d.get("correct", False)
                method = d.get("judge_method", "")
                print(f"  {model} [{ps}]: {ans[:150]}{'...' if len(ans) > 150 else ''}  => {'CORRECT' if ok else 'WRONG'} ({method})")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Result analysis
# ---------------------------------------------------------------------------

def _iter_model_results(model_results: Dict):
    """Yield (model, prompt_style, data) from nested or legacy model_results."""
    for model, by_style in model_results.items():
        if isinstance(by_style, dict) and "correct" in by_style and "full_response" in by_style:
            yield model, "cot", by_style
        elif isinstance(by_style, dict):
            for ps, data in by_style.items():
                if isinstance(data, dict) and "correct" in data:
                    yield model, ps, data


def analyze_results(results: Dict) -> Dict:
    """Compute accuracy by model, by category, and by (model, prompt_style)."""
    class_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0})))
    model_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    token_agg = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "thought_token_count": 0, "count": 0})

    for ev in results.get("evaluations", []):
        class_id = ev.get("class_id", "unknown")
        for model, ps, data in _iter_model_results(ev.get("model_results", {})):
            correct = data.get("correct", False)
            class_stats[class_id][model][ps]["total"] += 1
            if correct:
                class_stats[class_id][model][ps]["correct"] += 1
            model_stats[model][ps]["total"] += 1
            if correct:
                model_stats[model][ps]["correct"] += 1
            for tok in ("prompt_tokens", "completion_tokens", "thought_token_count"):
                if data.get(tok) is not None:
                    token_agg[model][tok] += data[tok]
            if data.get("prompt_tokens") is not None:
                token_agg[model]["count"] += 1

    summary = {"by_model": {}, "by_class": {}, "by_model_prompt": {}}

    for model, by_ps in model_stats.items():
        total_c = sum(s["correct"] for s in by_ps.values())
        total = sum(s["total"] for s in by_ps.values())
        summary["by_model"][model] = {
            "accuracy": round(total_c / total * 100, 2) if total else 0,
            "correct": total_c,
            "total": total,
        }
        agg = token_agg[model]
        if agg["count"] > 0:
            n = agg["count"]
            summary["by_model"][model]["avg_prompt_tokens"] = round(agg["prompt_tokens"] / n, 1)
            summary["by_model"][model]["avg_completion_tokens"] = round(agg["completion_tokens"] / n, 1)
            if agg["thought_token_count"] > 0:
                summary["by_model"][model]["avg_thought_tokens"] = round(agg["thought_token_count"] / n, 1)

    for model, by_ps in model_stats.items():
        summary["by_model_prompt"][model] = {
            ps: {"accuracy": round(s["correct"] / s["total"] * 100, 2) if s["total"] else 0,
                 "correct": s["correct"], "total": s["total"]}
            for ps, s in by_ps.items()
        }

    class_names = {}
    for ev in results.get("evaluations", []):
        cid = ev.get("class_id", "")
        if cid and cid not in class_names:
            class_names[cid] = ev.get("class_name", "")

    for class_id, by_model in class_stats.items():
        total_c = sum(s["correct"] for bm in by_model.values() for s in bm.values())
        total = sum(s["total"] for bm in by_model.values() for s in bm.values())
        summary["by_class"][class_id] = {
            "name": class_names.get(class_id, ""),
            "avg_accuracy": round(total_c / total * 100, 2) if total else 0,
            "models": {
                model: {
                    ps: {"accuracy": round(s["correct"] / s["total"] * 100, 2) if s["total"] else 0,
                         "correct": s["correct"], "total": s["total"]}
                    for ps, s in by_ps.items()
                }
                for model, by_ps in by_model.items()
            },
        }

    return summary


def print_summary(summary: Dict):
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print("\nOverall Accuracy (all prompt styles):")
    print("-" * 50)
    for model, stats in sorted(summary["by_model"].items(), key=lambda x: -x[1]["accuracy"]):
        print(f"  {model:30s}: {stats['accuracy']:6.2f}%  ({stats['correct']}/{stats['total']})")

    token_models = [(m, s) for m, s in summary["by_model"].items() if s.get("avg_prompt_tokens")]
    if token_models:
        print("\nAverage token usage per question (input / output):")
        print("-" * 50)
        for model, stats in sorted(token_models, key=lambda x: -x[1].get("avg_prompt_tokens", 0)):
            print(f"  {model:30s}: in {stats['avg_prompt_tokens']:>8.1f}  out {stats['avg_completion_tokens']:>8.1f}")

    print("\nAccuracy by category:")
    print("-" * 70)
    for class_id in sorted(summary["by_class"].keys()):
        data = summary["by_class"][class_id]
        print(f"\n  {class_id}  {data['name']}  (avg {data['avg_accuracy']:.1f}%)")
        for model in sorted(data["models"].keys()):
            for ps, st in sorted(data["models"][model].items()):
                bar = "█" * int(st["accuracy"] / 5) + "░" * (20 - int(st["accuracy"] / 5))
                print(f"    {model:20s} [{ps}]: {bar} {st['accuracy']:5.1f}%  ({st['correct']}/{st['total']})")


def merge_evaluation_results(file_paths: List[str], out_path: str) -> str:
    """Merge multiple result JSON files by question key, combining model_results."""
    by_key = {}
    for path in file_paths:
        if not os.path.exists(path):
            print(f"  Skipping missing file: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ev in data.get("evaluations", []):
            key = ev.get("key", "")
            if not key:
                continue
            if key not in by_key:
                by_key[key] = {k: v for k, v in ev.items() if k != "model_results"}
                by_key[key]["model_results"] = {}
            for model, styles in ev.get("model_results", {}).items():
                if model not in by_key[key]["model_results"]:
                    by_key[key]["model_results"][model] = {}
                by_key[key]["model_results"][model].update(styles)
    merged = {"evaluations": [by_key[k] for k in sorted(by_key)], "summary": {}}
    merged["summary"] = analyze_results(merged)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Merged {len(file_paths)} files -> {out_path}  ({len(by_key)} questions)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    global JUDGE_MODEL
    import argparse

    parser = argparse.ArgumentParser(
        description="WildTableBench — Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Named model shortcuts
  python evaluate.py --models gpt4o
  python evaluate.py --models gemini_2_flash gpt4o_mini
  python evaluate.py --models local

  # Any OpenRouter model ID (pass directly)
  python evaluate.py --models openai/gpt-4o-mini
  python evaluate.py --models meta-llama/llama-3.2-90b-vision-instruct

  # Merge results from separate runs
  python evaluate.py --merge part0.json part1.json -o merged.json

Supported model shortcuts:
  """ + "  ".join(sorted(SUPPORTED_MODELS.keys())) + """

For local models, set:
  LOCAL_MODEL_BASE_URL  (default: http://localhost:8000)
  LOCAL_MODEL_NAME      (e.g. Qwen/Qwen2.5-VL-7B-Instruct)

Full OpenRouter model list: https://openrouter.ai/models
        """,
    )

    parser.add_argument("--data", type=str, default=DEFAULT_DATA_FILE, metavar="FILE",
                        help="Path to metadata.csv (default: hf_wild/metadata.csv)")
    parser.add_argument("--images-dir", type=str, default=None, metavar="DIR",
                        help="Base directory for image files (default: same dir as --data)")
    parser.add_argument("--models", type=str, nargs="+", default=None, metavar="MODEL",
                        help="Model names to evaluate (see examples above)")
    parser.add_argument("--prompt-styles", type=str, nargs="+", default=["cot"],
                        choices=["cot", "direct"], metavar="STYLE",
                        help="Prompt style(s): cot (default) or direct")
    parser.add_argument("--output", "-o", type=str, default=None, metavar="FILE",
                        help=f"Output JSON path (default: {DEFAULT_OUTPUT_FILE})")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Max number of new questions to evaluate")
    parser.add_argument("--only-class", type=str, default=None, metavar="ID",
                        help="Evaluate only questions of this category (e.g. C2)")
    parser.add_argument("--concurrent-models", type=int, default=1, metavar="N",
                        help="Number of (model, style) API calls to run in parallel per question")
    parser.add_argument("--judge-model", type=str, default=None, metavar="MODEL",
                        help=f"Judge model name or OpenRouter ID (default: {JUDGE_MODEL})")
    parser.add_argument("--merge", nargs="+", metavar="FILE",
                        help="Merge multiple result JSONs (requires -o)")

    args = parser.parse_args()

    if args.judge_model:
        JUDGE_MODEL = args.judge_model

    out_path = args.output or DEFAULT_OUTPUT_FILE

    if args.merge:
        if not args.output:
            parser.error("--merge requires -o OUTPUT_FILE")
        merge_evaluation_results(args.merge, args.output)
        return

    if args.models is None:
        parser.error("--models is required. Example: --models gpt4o")

    print("=" * 60)
    print("WildTableBench — Model Evaluation")
    print("=" * 60)

    print(f"\nLoading questions from: {args.data}")
    annotations = load_annotations(args.data, images_dir=args.images_dir)
    print(f"Loaded {len(annotations)} questions")

    class_counts = defaultdict(int)
    for ann in annotations:
        class_counts[ann.get("class_id") or "unknown"] += 1
    print("\nCategory distribution:")
    for cid, cnt in sorted(class_counts.items()):
        cname = next((a["class_name"] for a in annotations if a.get("class_id") == cid), "")
        print(f"  {cid:6s} {cnt:4d}  {cname}")

    print(f"\nModels:       {args.models}")
    print(f"Prompt style: {args.prompt_styles}")
    print(f"Judge model:  {JUDGE_MODEL}")
    print(f"Output:       {out_path}")

    results = run_evaluation(
        annotations,
        models=args.models,
        limit=args.limit,
        prompt_styles=args.prompt_styles,
        output_file=out_path,
        concurrent_models=args.concurrent_models,
        only_class=args.only_class,
    )

    summary = analyze_results(results)
    results["summary"] = summary
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print_summary(summary)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
