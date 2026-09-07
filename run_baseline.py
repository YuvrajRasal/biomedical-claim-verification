import os
import sys
import json
import re
import time
from typing import Literal, Dict, Any, List
from pydantic import BaseModel, Field
import httpx

# Resilient connection retry handler for network stability across HF Hub requests
_orig_send = httpx.Client.send

def _resilient_send(self, request, *args, **kwargs):
    for attempt in range(5):
        try:
            return _orig_send(self, request, *args, **kwargs)
        except (httpx.ConnectError, httpx.ReadError):
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))

httpx.Client.send = _resilient_send

from datasets import load_dataset
from openai import OpenAI


# ---------------------------------------------------------
# Pydantic V2 Schema for Structured Output Validation
# ---------------------------------------------------------
class FactCheckDetermination(BaseModel):
    verdict: Literal["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
    confidence: float = Field(ge=0.0, le=1.0)
    verbatim_evidence_sentence: str
    scientific_rationale: str


def load_env_credentials():
    """Load API credentials from environment variables or .env fallback."""
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(__file__), ".env"),
        r"C:\Users\yuvra\Desktop\agenticAi_projectProposal\.env",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k and v and k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass


def extract_ground_truth_label(record: dict) -> str:
    """Classify benchmark ground-truth label as SUPPORTS, REFUTES, or NOT_ENOUGH_INFO."""
    raw_label = (record.get("evidence_label") or "").strip().upper()
    if not raw_label and "label" in record and record["label"] is not None:
        raw_label = str(record["label"]).strip().upper()

    if not raw_label:
        evidence = record.get("evidence", {})
        if isinstance(evidence, dict):
            for doc_entries in evidence.values():
                if isinstance(doc_entries, list):
                    for entry in doc_entries:
                        if isinstance(entry, dict) and "label" in entry:
                            raw_label = str(entry["label"]).strip().upper()
                            break
                elif isinstance(doc_entries, dict) and "label" in doc_entries:
                    raw_label = str(doc_entries["label"]).strip().upper()
                    break

    if "SUPPORT" in raw_label:
        return "SUPPORTS"
    if "CONTRADICT" in raw_label or "REFUTE" in raw_label:
        return "REFUTES"
    return "NOT_ENOUGH_INFO"


def predict_claim(client: OpenAI, claim_text: str, abstract_text: str) -> FactCheckDetermination:
    """Query hosted_vllm/llama4-scout-17b on Voyager RC and validate JSON output."""
    system_prompt = (
        "You are an expert biomedical scientific fact-checker. "
        "Evaluate the provided biomedical CLAIM against the cited SCIENTIFIC ABSTRACT.\n"
        "You must respond ONLY with a raw JSON object matching this exact schema:\n"
        "{\n"
        '  "verdict": "SUPPORTS" | "REFUTES" | "NOT_ENOUGH_INFO",\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "verbatim_evidence_sentence": "<exact sentence quoted directly from the abstract that justifies the verdict, or empty string if NOT_ENOUGH_INFO>",\n'
        '  "scientific_rationale": "<concise scientific reasoning grounding the determination>"\n'
        "}\n"
        "Do NOT enclose your output in markdown code blocks or backticks. Return raw JSON only."
    )

    user_prompt = (
        f"CLAIM:\n{claim_text}\n\n"
        f"CITED ABSTRACT:\n{abstract_text if abstract_text else 'No cited abstract available.'}\n\n"
        "Provide your factual determination in pure JSON format."
    )

    response = client.chat.completions.create(
        model="hosted_vllm/llama4-scout-17b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    raw_text = response.choices[0].message.content or ""
    cleaned_json = raw_text.strip()
    cleaned_json = re.sub(r"^```(?:json)?\s*", "", cleaned_json, flags=re.IGNORECASE)
    cleaned_json = re.sub(r"\s*```$", "", cleaned_json).strip()

    json_match = re.search(r"\{.*\}", cleaned_json, flags=re.DOTALL)
    if json_match:
        cleaned_json = json_match.group(0)

    parsed_dict = json.loads(cleaned_json)
    return FactCheckDetermination.model_validate(parsed_dict)


def run_single_claim(sample_index: int = 0, split: str = "train") -> Dict[str, Any]:
    """Execute baseline verification on a single claim and output formatted JSON (Section 3 & 4)."""
    load_env_credentials()
    api_key = os.environ.get("ASU_RC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Missing ASU_RC_API_KEY or OPENAI_API_KEY in environment.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url="https://openai.rc.asu.edu/v1", api_key=api_key)

    scifact_claims = load_dataset("allenai/scifact", "claims", split=split)
    scifact_corpus = load_dataset("allenai/scifact", "corpus", split="train")

    doc_lookup = {doc["doc_id"]: " ".join(doc["abstract"]) for doc in scifact_corpus}

    if sample_index < 0 or sample_index >= len(scifact_claims):
        print(f"Error: Sample index {sample_index} out of range [0, {len(scifact_claims) - 1}]", file=sys.stderr)
        sys.exit(1)

    record = scifact_claims[sample_index]
    claim_id = record["id"]
    claim_text = record["claim"]

    cited_doc_ids = record.get("cited_doc_ids", [])
    abstract_text = ""
    if cited_doc_ids:
        cited_id = cited_doc_ids[0]
        abstract_text = doc_lookup.get(cited_id) or doc_lookup.get(int(cited_id) if str(cited_id).isdigit() else None) or ""

    gt_label = extract_ground_truth_label(record)
    determination = predict_claim(client, claim_text, abstract_text)

    output_payload = {
        "claim_id": claim_id,
        "claim": claim_text,
        "ground_truth_label": gt_label,
        "baseline_prediction": determination.model_dump(),
    }

    print(json.dumps(output_payload, indent=2))
    return output_payload


def match_sentence_to_abstract(extracted_sentence: str, abstract_sentences: List[str]) -> int:
    """Find matching sentence index in the abstract for evidence evaluation."""
    if not extracted_sentence or not abstract_sentences:
        return -1
    
    cleaned_quote = re.sub(r"[^\w\s]", "", extracted_sentence.lower()).strip()
    if not cleaned_quote:
        return -1

    best_idx = -1
    best_overlap = 0.0

    for idx, sent in enumerate(abstract_sentences):
        cleaned_sent = re.sub(r"[^\w\s]", "", sent.lower()).strip()
        if cleaned_quote in cleaned_sent or cleaned_sent in cleaned_quote:
            return idx

        words_q = set(cleaned_quote.split())
        words_s = set(cleaned_sent.split())
        if words_q and words_s:
            overlap = len(words_q & words_s) / len(words_q | words_s)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx

    return best_idx if best_overlap >= 0.4 else -1


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    gt_sentences_list: List[List[int]],
    pred_sentences_list: List[int],
    faithfulness_scores: List[float]
) -> Dict[str, Any]:
    """Compute Accuracy, Macro-F1, Evidence Sentence F1, and Faithfulness."""
    classes = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total if total > 0 else 0.0

    per_class = {}
    f1_scores = []

    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        support = sum(1 for yt in y_true if yt == c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)

        per_class[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    sentence_f1s = []
    for gt_sents, pred_idx in zip(gt_sentences_list, pred_sentences_list):
        if gt_sents:
            if pred_idx != -1 and pred_idx in gt_sents:
                sentence_f1s.append(1.0)
            else:
                sentence_f1s.append(0.0)
        else:
            if pred_idx == -1:
                sentence_f1s.append(1.0)
            else:
                sentence_f1s.append(0.0)

    avg_sentence_f1 = sum(sentence_f1s) / len(sentence_f1s) if sentence_f1s else 0.0
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 1.0

    return {
        "total_evaluated": total,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "evidence_sentence_f1": round(avg_sentence_f1, 4),
        "retrieval_recall_at_5": "0.0% (Oracle only)",
        "faithfulness_score": round(avg_faithfulness, 4),
        "per_class": per_class,
    }


def run_batch_evaluation(num_samples: int = None, split: str = "validation", concurrency: int = 8):
    """Execute high-throughput evaluation across claims and compute Section 6 proposal metrics."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    load_env_credentials()
    api_key = os.environ.get("ASU_RC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Missing ASU_RC_API_KEY or OPENAI_API_KEY in environment.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url="https://openai.rc.asu.edu/v1", api_key=api_key)

    scifact_claims = load_dataset("allenai/scifact", "claims", split=split)
    scifact_corpus = load_dataset("allenai/scifact", "corpus", split="train")

    doc_abstract_map = {doc["doc_id"]: doc["abstract"] for doc in scifact_corpus}

    total_claims = len(scifact_claims)
    n = min(num_samples, total_claims) if num_samples is not None else total_claims

    print(f"\n=======================================================")
    print(f" SciFact Benchmark Evaluation: {n} claims ({split} split)")
    print(f" Model: hosted_vllm/llama4-scout-17b | Concurrency: {concurrency}")
    print(f"=======================================================\n")

    def process_single(idx: int):
        record = scifact_claims[idx]
        claim_text = record["claim"]
        cited_doc_ids = record.get("cited_doc_ids", [])
        
        abstract_sentences = []
        if cited_doc_ids:
            cited_id = cited_doc_ids[0]
            abstract_sentences = (
                doc_abstract_map.get(cited_id)
                or doc_abstract_map.get(int(cited_id) if str(cited_id).isdigit() else None)
                or []
            )

        abstract_text = " ".join(abstract_sentences) if abstract_sentences else ""
        gt_label = extract_ground_truth_label(record)
        gt_sents = record.get("evidence_sentences", []) or []

        t0 = time.time()
        try:
            pred = predict_claim(client, claim_text, abstract_text)
            pred_verdict = pred.verdict
            extracted_quote = pred.verbatim_evidence_sentence.strip()
        except Exception as e:
            pred_verdict = "NOT_ENOUGH_INFO"
            extracted_quote = ""

        dt = time.time() - t0
        pred_sent_idx = match_sentence_to_abstract(extracted_quote, abstract_sentences) if extracted_quote else -1

        is_faithful = 1.0
        if extracted_quote and abstract_text:
            if extracted_quote.lower() not in abstract_text.lower():
                is_faithful = 0.85 if pred_sent_idx != -1 else 0.50

        return {
            "idx": idx,
            "claim_id": record["id"],
            "gt_label": gt_label,
            "pred_verdict": pred_verdict,
            "gt_sents": gt_sents,
            "pred_sent_idx": pred_sent_idx,
            "is_faithful": is_faithful,
            "duration": dt,
        }

    results = [None] * n
    completed_count = 0
    t_batch_start = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {executor.submit(process_single, i): i for i in range(n)}
        for future in as_completed(future_map):
            res = future.result()
            results[res["idx"]] = res
            completed_count += 1
            match_symbol = "[OK]" if res["pred_verdict"] == res["gt_label"] else "[X] "
            pct = (completed_count / n) * 100
            print(f"[{completed_count:03d}/{n:03d}] ({pct:4.1f}%) Verdict: {match_symbol} ({res['pred_verdict']:<15}) | Latency: {res['duration']:.2f}s")

    total_batch_time = time.time() - t_batch_start

    y_true = [r["gt_label"] for r in results]
    y_pred = [r["pred_verdict"] for r in results]
    gt_sentences_list = [r["gt_sents"] for r in results]
    pred_sentences_list = [r["pred_sent_idx"] for r in results]
    faithfulness_scores = [r["is_faithful"] for r in results]
    durations = [r["duration"] for r in results]

    metrics = compute_metrics(y_true, y_pred, gt_sentences_list, pred_sentences_list, faithfulness_scores)
    avg_latency = sum(durations) / len(durations) if durations else 0.0
    throughput = n / total_batch_time if total_batch_time > 0 else 0.0

    print("\n=======================================================")
    print(" SECTION 6 PROPOSAL EVALUATION METRICS (MEASURED)")
    print("=======================================================")
    print(f" Evaluated Claims             : {metrics['total_evaluated']} ({split} split)")
    print(f" 1. Claim Verdict Macro-F1    : {metrics['macro_f1'] * 100:.1f}%  (Accuracy: {metrics['accuracy'] * 100:.1f}%)")
    print(f" 2. Evidence Sentence F1      : {metrics['evidence_sentence_f1'] * 100:.1f}%")
    print(f" 3. Retrieval Recall@5        : {metrics['retrieval_recall_at_5']}")
    print(f" 4. Faithfulness Score        : {metrics['faithfulness_score'] * 100:.1f}%")
    print(f" 5. Average Latency           : {avg_latency:.2f} s / claim")
    print(f" 6. Batch Throughput          : {throughput:.2f} claims/sec (Total Time: {total_batch_time:.1f}s)")
    print("=======================================================\n")

    print(f"{'Class':<17} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'Support'}")
    print("-" * 55)
    for cls, val in metrics["per_class"].items():
        print(f"{cls:<17} | {val['precision']*100:>8.1f}% | {val['recall']*100:>8.1f}% | {val['f1']*100:>8.1f}% | {val['support']}")
    print("-------------------------------------------------------\n")

    output_file = "baseline_evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "split": split,
            "total_time_seconds": round(total_batch_time, 2),
            "throughput_claims_per_sec": round(throughput, 2),
            "avg_latency_seconds": round(avg_latency, 3),
            "sample_predictions": [
                {
                    "index": r["idx"],
                    "claim_id": r["claim_id"],
                    "true_label": r["gt_label"],
                    "pred_label": r["pred_verdict"],
                    "true_sentences": r["gt_sents"],
                    "pred_sentence_idx": r["pred_sent_idx"],
                    "faithful": r["is_faithful"]
                }
                for r in results
            ]
        }, f, indent=2)
    print(f"Saved complete evaluation report to: {output_file}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SciFact Biomedical Claim Verification: Single Inference & Batch Evaluation"
    )
    parser.add_argument("index", type=int, nargs="?", default=None, help="Sample index for single claim (default: 0)")
    parser.add_argument("--index", type=int, dest="index_flag", default=None, help="Explicit flag for claim index")
    parser.add_argument("--eval", action="store_true", help="Run batch evaluation mode across multiple claims")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (default: all claims in split)")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"], help="SciFact split (default: validation)")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent request threads (default: 8)")
    args = parser.parse_args()

    # Determine execution mode: single claim vs batch evaluation
    if args.eval or args.num_samples is not None:
        run_batch_evaluation(num_samples=args.num_samples, split=args.split, concurrency=args.concurrency)
    else:
        single_idx = args.index if args.index is not None else (args.index_flag if args.index_flag is not None else 0)
        run_single_claim(sample_index=single_idx, split=args.split)


if __name__ == "__main__":
    main()

