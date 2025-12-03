# 🚀 Mini LLM Post-Training Pipeline (SFT + DPO + Eval + Experiment Logs)

*A minimal, production-inspired post-training workflow for aligning open-source LLMs using Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), structured experiment logging, and offline A/B-style evaluation.*

This project implements a **fully reproducible, end-to-end post-training pipeline** similar to what modern alignment teams use in industry (Inflection AI, Anthropic, OpenAI). It provides:

- 🔧 **Config-driven SFT and DPO training** (PyTorch + HuggingFace + TRL + LoRA)  
- 📚 **Dataset curation** for instruction-tuning & preference pairs  
- 📊 **Batch evaluation + A/B comparison** across Base / SFT / DPO models  
- 🧪 **Keyword-based scoring + per-prompt win-rate metrics**  
- 📝 **Structured experiment logging** for reproducibility  
- ⚙️ **Modular design** to extend with RLHF, reward models, ORPO, or larger architectures  
- 🧵 **TinyLlama 1.1B** as default lightweight backbone for fast local experiments  

The goal is to provide a **clean, understandable, and realistic** implementation of a modern LLM post-training workflow—small enough to run locally, yet architected like a real industrial system.

---

## 🎯 Why This Project Exists

Most open-source LLM tutorials show only a single training script or a small notebook.  
But real alignment work requires **pipelines, evaluation loops, experiment logs, and reproducibility**.

This project aims to bridge that gap:

> **“A minimal pipeline that mirrors real-world LLM post-training systems, but small enough for one person to run and understand fully.”**

It is especially suited for:

- Candidates preparing for **Applied ML / LLM Infra / Model Optimization** interviews  
- Researchers wanting a clean baseline for SFT → DPO → Eval  
- Students learning how RLHF-era post-training systems are structured  
- Engineers building their first alignment pipeline  

---

## 🧩 Features at a Glance

| Component | Description |
|----------|-------------|
| **SFT Training** | LoRA-based instruction tuning with config-driven training |
| **DPO Training** | Preference optimization with TRL (policy + reference model) |
| **Dataset Curation** | JSONL instruction and preference pair format |
| **Batch Evaluation** | Compare Base / SFT / DPO outputs at scale |
| **Metrics** | Keyword hit-rate & win-rate analysis |
| **A/B Testing** | Side-by-side answer comparison per prompt |
| **Experiment Logging** | Auto-save JSON logs for every run |
| **Reproducibility** | Seed control + config files + full run history |

---

## 🛠️ Tech Stack

- **PyTorch** — Core training execution  
- **HuggingFace Transformers** — Model & tokenizer  
- **TRL (Transformer Reinforcement Learning)** — DPO implementation  
- **PEFT / LoRA** — Parameter-efficient tuning  
- **YAML configs** — Full pipeline configurability  
- **JSONL datasets** — Easy dataset curation & extension  

The entire project is intentionally dependency-light and easy to run locally.

---

## 📦 Project Structure

```text
mini-llm-alignment-pipeline/
│
├── configs/
│   ├── sft_config.yaml
│   └── dpo_config.yaml
│
├── data/
│   ├── sft_examples.jsonl
│   ├── dpo_pairs.jsonl
│   └── eval_prompts.jsonl
│
├── aligner/
│   ├── data.py
│   ├── models.py
│   ├── eval.py
│   └── utils.py
│
├── scripts/
│   ├── train_sft.py
│   ├── train_dpo.py
│   └── eval_batch.py
│
├── experiments/        ← auto-generated logs
└── outputs/            ← trained model checkpoints
```

## ⚡ Quickstart

### 1. Install Requirements

```bash
# Option 1
pip3 install -r requirements.txt

# Option 2
pip install -r requirements.txt
```
### 2. Run SFT Training
```bash
python3 -m scripts.train_sft --config configs/sft_config.yaml
```
### 3. Run DPO Training
```bash
python3 -m scripts.train_dpo --config configs/dpo_config.yaml
```
### 4. Evaluate all models
```bash
python3 -m scripts.eval_batch \
  --base TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --sft outputs/sft \
  --dpo outputs/dpo
```
### 5. Inspect Experiments logs
```bash
ls experiments/
cat experiments/sft-*.json
```
## 📊 Evaluation & A/B Testing

This project includes a lightweight but practical offline evaluation framework for comparing the base, SFT, and DPO models across domain-specific prompts in probability, Markov chains, and time-series analysis.

The evaluation pipeline consists of: 
- Batch generation for all models on the same prompt set
- Keyword-based scoring (checks if key statistical ideas appear in the answer)
- Per-prompt win-rate analysis
- tructured logs saved to experiments/eval_batch_results.jsonl