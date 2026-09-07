# Biomedical Claim Verification System: Capstone Baseline

A minimal, reproducible baseline system for automated biomedical claim verification against peer-reviewed scientific literature, developed as part of an Applied Agentic AI Capstone Project.

---

## 1. Problem Statement & Overview

Biomedical misinformation and unverified scientific assertions pose severe risks to public health and clinical translation. Manually verifying emerging medical claims against voluminous, rapidly expanding peer-reviewed biomedical literature is cognitively demanding and time-prohibitive.

This project implements an end-to-end evidence-grounded verification pipeline that evaluates biomedical claims against peer-reviewed scientific abstracts, producing:
1. **Categorical Verdict**: `SUPPORTS`, `REFUTES`, or `NOT_ENOUGH_INFO`
2. **Calibrated Confidence**: Scaled score $[0.0, 1.0]$
3. **Verbatim Evidence Sentence**: Direct sentence-level quotation extracted from the scientific corpus
4. **Scientific Rationale**: Formal domain justification grounding the verdict

---

## 2. Benchmark Dataset Notice

> **Dataset Integrity Notice**: This baseline does **not** rely on mock, synthetic, or manually authored data files. Benchmark samples are ingested directly and dynamically from the official Hugging Face **[`allenai/scifact`](https://huggingface.co/datasets/allenai/scifact)** benchmark (splits: `claims` and `corpus`).

- **Corpus**: 5,183 peer-reviewed scientific abstracts indexed by PubMed ID with sentence-level boundaries.
- **Claims**: 1,261 expert-annotated biomedical statements mapped to cited rationale documents with ground-truth verification labels.

---

## 3. System Architecture & Model Configuration

- **High-Performance Inference Endpoint**: ASU Research Computing Voyager Cluster (`https://openai.rc.asu.edu/v1`)
- **Underlying LLM**: `hosted_vllm/llama4-scout-17b`
- **Output Validation**: Strict JSON schema enforcement via Pydantic V2 (`FactCheckDetermination`) with regex-based markdown fence sanitization for vLLM compatibility.

---

## 4. Prerequisites

- **Python**: Version 3.10 or higher (tested on Python 3.10, 3.11, 3.12, 3.14)
- **ASU Voyager RC API Credentials**: Active API key granting access to `https://openai.rc.asu.edu/v1`

---

## 5. Step-by-Step Setup & Installation

### Step 1: Clone or Navigate to the Repository
```bash
git clone https://github.com/YuvrajRasal/biomedical-claim-verification.git
cd biomedical-claim-verification
```

### Step 2: Initialize Virtual Environment
On Linux / macOS:
```bash
python -m venv .venv
# or: python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
You can install all dependencies via the requirements file:
```bash
pip install -r requirements.txt
```
> **Notice on `datasets`**: Ingesting the Hugging Face `allenai/scifact` benchmark dynamically requires the `datasets` package. Specifying `datasets>=2.18.0,<3.0.0` (pinned in `requirements.txt`) ensures full compatibility with SciFact's dataset loader script.


---

## 6. ASU Research Computing Voyager Setup & Authentication

Follow these steps to set up ASU Research Computing Voyager access for this project:

### Step 1: Connect to the ASU Cisco VPN
1. Open **Cisco Secure Client** (AnyConnect).
2. Connect to: `sslvpn.asu.edu/2fa`
3. Sign in with your **ASURITE** credentials and complete **Duo 2FA**.
4. Confirm that the VPN shows as **Connected**.

### Step 2: Configure Voyager API Access
1. Make sure you have ASU Voyager LLM API access.
2. Follow the official ASU documentation: [https://docs.rc.asu.edu/ai/api/](https://docs.rc.asu.edu/ai/api/)
3. Obtain your personal Voyager API key.

### Step 3: Set the API Key Environment Variable

**Windows (PowerShell):**
```powershell
$env:ASU_RC_API_KEY="your_voyager_api_key"
```

**Windows (Command Prompt):**
```cmd
set ASU_RC_API_KEY=your_voyager_api_key
```

**Linux / macOS:**
```bash
export ASU_RC_API_KEY="your_voyager_api_key"
```

*(Optional fallback: The runner also automatically checks a local `.env` file or `OPENAI_API_KEY`).*

> [!CAUTION]
> **Security Notice**: Never commit, publish, or share your Voyager API key.

### Step 4: Endpoint & Model Specifications
The project utilizes:
- **API Endpoint**: `https://openai.rc.asu.edu/v1`
- **Model**: `hosted_vllm/llama4-scout-17b`
- **API Key Variable**: `ASU_RC_API_KEY`

---

## 7. Execution Commands

### Mode A: Single Claim Verification (Baseline Output for Section 4)
Run verification on sample index `0` (or any valid integer from `0` to `1260`):
```bash
python run_baseline.py 0
```

To run against claim index `1` (which tests supporting evidence with rationale extraction):
```bash
python run_baseline.py 1
```

### Mode B: Full 450-Claim Benchmark Evaluation (Section 6)
Run the automated benchmark evaluation across all 450 canonical SciFact validation claims using multi-threaded vLLM batching:
```bash
python run_baseline.py --eval --split validation --concurrency 8
```
*(Completes in ~2 minutes; results are summarized in a terminal metrics table and saved to `baseline_evaluation_results.json`).*

*(Optional: Pass `--num_samples 20` for a quick 20-claim smoke test: `python run_baseline.py --eval --num_samples 20`).*

---



## 8. Expected Terminal Output

When executed against sample index `0` of the SciFact benchmark:

```bash
$ python run_baseline.py 0
```

```json
{
  "claim_id": 1,
  "claim": "0-dimensional biomaterials show inductive properties.",
  "ground_truth_label": "NOT_ENOUGH_INFO",
  "baseline_prediction": {
    "verdict": "NOT_ENOUGH_INFO",
    "confidence": 0.0,
    "verbatim_evidence_sentence": "",
    "scientific_rationale": "The abstract discusses various nanotechnologies and their applications in stem cell research, but it does not specifically mention 0-dimensional biomaterials or their inductive properties."
  }
}
```

When executed against sample index `1` (which contains supporting evidence):

```bash
$ python run_baseline.py 1
```

```json
{
  "claim_id": 3,
  "claim": "1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.",
  "ground_truth_label": "SUPPORTS",
  "baseline_prediction": {
    "verdict": "SUPPORTS",
    "confidence": 0.9,
    "verbatim_evidence_sentence": "In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.",
    "scientific_rationale": "The cited abstract discusses how rare genetic variants can create 'synthetic associations' that are mistakenly attributed to common variants in genome-wide association studies (GWAS). This supports the claim that the 1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants, as it suggests that rare variants can have significant effects that are not captured by common variant analysis."
  }
}
```

> **Baseline Evaluation Summary**: Both test cases aligned with ground truth (`NOT_ENOUGH_INFO` and `SUPPORTS`); exact evidence quotation succeeded on the support case; confidence was calibrated at 0.0 for indeterminate evidence vs. 0.90 for unequivocal literature support; Pydantic validation produced zero parsing errors. Weaknesses: oracle document dependency, no cross-study synthesis, and no study-quality assessment.

### Full Benchmark Evaluation Output (SciFact 450 Validation Claims)

When evaluated across all 450 validation claims via multithreaded vLLM batching:

```bash
$ python run_baseline.py --eval --split validation --concurrency 8
```

```text
=======================================================
 SECTION 6 PROPOSAL EVALUATION METRICS (MEASURED)
=======================================================
 Evaluated Claims             : 450 (validation split)
 1. Claim Verdict Macro-F1    : 79.3%  (Accuracy: 81.6%)
 2. Evidence Sentence F1      : 50.7%
 3. Retrieval Recall@5        : 0.0% (Oracle only)
 4. Faithfulness Score        : 99.2%
 5. Average Latency           : 2.14 s / claim
 6. Batch Throughput          : 3.73 claims/sec (Total Time: 120.6s)
=======================================================

Class             | Precision  | Recall     | F1         | Support
-------------------------------------------------------
SUPPORTS          |     80.3% |     94.4% |     86.8% | 216
REFUTES           |     82.1% |     75.4% |     78.6% | 122
NOT_ENOUGH_INFO   |     84.5% |     63.4% |     72.5% | 112
-------------------------------------------------------
```


---

## 9. Capstone Semester Roadmap: Transition to Multi-Agent System

While this baseline executes single-turn verification against a pre-identified abstract, the full semester project scales into an autonomous multi-agent consensus architecture:

1. **Automated PubMed Search Agent**:
   - Query formulation via domain NER and MeSH (Medical Subject Headings) ontology mapping.
   - Dynamic retrieval of live peer-reviewed biomedical literature via NCBI E-Utilities API.

2. **Methodology & Bias Scorer Agent**:
   - Critiques study design (e.g., in vitro vs. randomized controlled trial [RCT], sample power, p-hacking risks, conflict of interest disclosures).
   - Downweights lower-evidence studies (case reports, non-peer-reviewed preprints).

3. **Adversarial Consensus & Synthesis Agent**:
   - Orchestrates multi-agent debate between proponent and skeptic critic agents.
   - Computes weighted Bayesian consensus across conflicting medical studies.
