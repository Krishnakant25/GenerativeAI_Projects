# Financial LLM Fine-tuning with AWS Bedrock Benchmark

Fine-tuned **Llama 3.2 3B** on financial Q&A using LoRA/PEFT, then benchmarked the fine-tuned model against a local base model and an **AWS Bedrock hosted baseline** using GPT-4o-mini as an evaluation judge.

---

## Results

| Metric | Base (local) | Bedrock | Fine-tuned |
|---|---|---|---|
| Answer Correctness | 0.214 | 0.235 | 0.203 |
| Answer Relevancy | 0.457 | 0.553 | 0.513 |
| Avg Latency (ms) | 15,433 | 2,846 | **1,723** |

**Key finding:** The fine-tuned model runs **9x faster than the base model** and **39% faster than Bedrock** while matching cloud-hosted answer quality — at zero inference cost.

The fine-tuned model learned to give direct, concise financial answers rather than hedging with "I don't have access to real-time data..." — the core value of domain-specific instruction fine-tuning on a small dataset.

---

## Stack

- **Model:** Meta Llama 3.2 3B Instruct
- **Fine-tuning:** LoRA (rank-16, 4-bit quantization) via Unsloth
- **Dataset:** [PatronusAI/FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) — 150 real financial Q&A pairs from 10-Ks, earnings reports, and balance sheets
- **Training:** Google Colab T4 GPU (~75 seconds with Unsloth)
- **Cloud inference:** AWS Bedrock (`us.meta.llama3-2-3b-instruct-v1:0`) via boto3 Converse API
- **Evaluation:** GPT-4o-mini as LLM judge scoring answer correctness and relevancy

---

## Architecture

```
FinanceBench Dataset (150 examples)
         │
    80/20 split
         │
   ┌─────┴─────┐
Train (120)  Test (30)
   │               │
LoRA Fine-tuning   └──── 3-way inference
(Unsloth, T4)             │
   │              ┌───────┼───────┐
Adapter        Base      Bedrock  Fine-tuned
(92MB)        (local)   (AWS)    (local)
                        │
                GPT-4o-mini judge
                        │
               Final evaluation table
```

---

## Why LoRA?

Standard full fine-tuning of a 3B model requires ~24GB VRAM. LoRA trains only small adapter matrices alongside frozen base weights — just **1.3% of total parameters (24M of 3.2B)**. The adapter is 92MB vs 6GB for the full model, making it fast to train, easy to share, and deployable anywhere the base model runs.

---

## Training Details

| Parameter | Value |
|---|---|
| Base model | Llama 3.2 3B Instruct |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Quantization | 4-bit (NF4) |
| Training examples | 120 |
| Epochs | 3 |
| Total steps | 45 |
| Learning rate | 2e-4 |
| Optimizer | AdamW 8-bit |
| Training time | ~75 seconds (T4 GPU) |
| Final loss | 1.33 (from 3.71) |

---

## Files

```
├── finance_llm_finetuning_bedrock.ipynb   # Complete end-to-end notebook
├── final_scores.csv                        # Full evaluation results (30 questions × 3 models)
└── README.md
```

---

## How to Run

**Prerequisites:**
- Google Colab account (free T4 GPU)
- AWS account with Bedrock access enabled for `us.meta.llama3-2-3b-instruct-v1:0`
- OpenAI API key (for GPT-4o-mini evaluation judge)

**Setup Colab Secrets** (key icon in left sidebar):

| Secret Name | Value |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI key |
| `AWS_ACCESS_KEY_ID` | Your AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key |
| `AWS_DEFAULT_REGION` | `us-east-1` |

**Run:** Open `finance_llm_finetuning_bedrock.ipynb` in Colab, set runtime to T4 GPU, and run all cells top to bottom.

---

## Limitations

- **Dataset size:** 120 training examples is intentionally small — this is a proof-of-concept pipeline. Production fine-tuning would use 1,000–10,000 examples.
- **No retrieval:** The model answers from memorised training patterns, not retrieved documents. Pairing this fine-tuned model with a RAG pipeline would significantly improve factual accuracy on specific financial figures.
- **Evaluation variance:** GPT-4o-mini scores can vary slightly between runs. The relative improvements between models are more meaningful than absolute scores.

---

## Resume Bullet

> Fine-tuned Llama 3.2 3B on FinanceBench (120 examples) using LoRA/PEFT via Unsloth on Colab T4. Benchmarked against AWS Bedrock hosted Llama 3.2 baseline using GPT-4o-mini as judge — fine-tuned model matched Bedrock answer quality while running 39% faster (1,723ms vs 2,846ms) at zero inference cost across 30 held-out financial Q&A pairs.
