# scripts/eval_batch.py
import argparse
import json
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(path: str):
    """Load a causal LM and tokenizer from a path or model name."""
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype="auto",
        device_map="auto",
    )
    return model, tokenizer


def generate(model, tokenizer, prompt: str) -> str:
    full_prompt = f"Instruction:\n{prompt}\n\nResponse:\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(full_prompt):].strip()


def load_eval_prompts(path: str) -> List[Dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            rows.append(obj)
    return rows


def score_answer(answer: str, keywords: List[str]) -> float:
    """
    Very simple metric: fraction of keywords that appear
    (case-insensitive, substring-based).
    """
    text = answer.lower()
    hit = 0
    for kw in keywords:
        if kw.lower() in text:
            hit += 1
    if not keywords:
        return 0.0
    return hit / len(keywords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, default="data/eval_prompts.jsonl")
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--sft", type=str, required=True)
    parser.add_argument("--dpo", type=str, required=True)
    args = parser.parse_args()

    print("Loading models...")
    base_model, base_tok = load_model(args.base)
    sft_model, sft_tok = load_model(args.sft)
    dpo_model, dpo_tok = load_model(args.dpo)

    prompts = load_eval_prompts(args.eval_file)
    print(f"Loaded {len(prompts)} eval prompts.\n")

    results = []  # list of dicts with scores and winners

    base_scores = []
    sft_scores = []
    dpo_scores = []

    for i, row in enumerate(prompts):
        prompt = row["prompt"]
        keywords = row.get("keywords", [])

        print("=" * 80)
        print(f"[{i}] PROMPT: {prompt}\n")
        print(f"Keywords: {keywords}\n")

        base_ans = generate(base_model, base_tok, prompt)
        sft_ans = generate(sft_model, sft_tok, prompt)
        dpo_ans = generate(dpo_model, dpo_tok, prompt)

        base_score = score_answer(base_ans, keywords)
        sft_score = score_answer(sft_ans, keywords)
        dpo_score = score_answer(dpo_ans, keywords)

        base_scores.append(base_score)
        sft_scores.append(sft_score)
        dpo_scores.append(dpo_score)

        # Decide winner(s) for this prompt
        scores = {
            "base": base_score,
            "sft": sft_score,
            "dpo": dpo_score,
        }
        best_model = max(scores, key=scores.get)

        results.append(
            {
                "prompt": prompt,
                "keywords": keywords,
                "base_answer": base_ans,
                "sft_answer": sft_ans,
                "dpo_answer": dpo_ans,
                "base_score": base_score,
                "sft_score": sft_score,
                "dpo_score": dpo_score,
                "winner": best_model,
            }
        )

        print("[BASE] score={:.2f}\n{}".format(base_score, base_ans))
        print("\n[SFT ] score={:.2f}\n{}".format(sft_score, sft_ans))
        print("\n[DPO ] score={:.2f}\n{}".format(dpo_score, dpo_ans))
        print(f"\n>>> Winner for this prompt: {best_model.upper()}\n")

    # Summary stats
    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    avg_base = avg(base_scores)
    avg_sft = avg(sft_scores)
    avg_dpo = avg(dpo_scores)

    winners = [r["winner"] for r in results]
    base_wins = winners.count("base")
    sft_wins = winners.count("sft")
    dpo_wins = winners.count("dpo")

    print("=" * 80)
    print("SUMMARY\n")
    print(f"Average keyword hit rate:")
    print(f"  BASE: {avg_base:.3f}")
    print(f"  SFT : {avg_sft:.3f}")
    print(f"  DPO : {avg_dpo:.3f}\n")

    print("Win counts over prompts:")
    print(f"  BASE wins: {base_wins}")
    print(f"  SFT  wins: {sft_wins}")
    print(f"  DPO  wins: {dpo_wins}")
    print("=" * 80)

    # Optionally, save detailed results to a JSONL
    out_path = "experiments/eval_batch_results.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nDetailed eval results saved to {out_path}")


if __name__ == "__main__":
    main()
