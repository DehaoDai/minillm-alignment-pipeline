# aligner/eval.py
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype="auto",
        device_map="auto",
    )
    return model, tokenizer


def generate(model, tokenizer, prompt):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--dpo", required=True)
    args = parser.parse_args()

    print("Loading models...")
    base_model, base_tok = load_model(args.base)
    sft_model, sft_tok = load_model(args.sft)
    dpo_model, dpo_tok = load_model(args.dpo)

    print("\nModels loaded. Enter a prompt (or 'exit' to quit)\n")

    while True:
        prompt = input(">>> ")
        if prompt.lower() in ["exit", "quit"]:
            break

        print("\n===== BASE MODEL =====")
        print(generate(base_model, base_tok, prompt))

        print("\n===== SFT MODEL =====")
        print(generate(sft_model, sft_tok, prompt))

        print("\n===== DPO MODEL =====")
        print(generate(dpo_model, dpo_tok, prompt))

        print("\n---------------------------------------------\n")


if __name__ == "__main__":
    main()
