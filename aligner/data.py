# aligner/data.py
import json
from dataclasses import dataclass
from typing import List, Dict

from datasets import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class SFTExample:
    instruction: str
    output: str


def load_sft_jsonl(path: str) -> List[SFTExample]:
    """Load SFT examples from a JSONL file."""
    examples: List[SFTExample] = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            examples.append(
                SFTExample(
                    instruction=obj["instruction"],
                    output=obj["output"],
                )
            )
    return examples


def make_sft_dataset(
    path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    """
    Build a HuggingFace Dataset for SFT.
    Each example is turned into a single text string:
    'Instruction: ... Response: ...'
    """
    examples = load_sft_jsonl(path)
    texts = [
        f"Instruction:\n{ex.instruction}\n\nResponse:\n{ex.output}"
        for ex in examples
    ]
    raw = {"text": texts}
    ds = Dataset.from_dict(raw)

    def tokenize_fn(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    ds = ds.map(tokenize_fn, batched=True)
    return ds

# ----- DPO dataset loader -----


def load_dpo_jsonl(path: str) -> Dataset:
    """
    Load preference pairs for DPO from a JSONL file.
    Each line should be a JSON object with
    {
        "prompt": "...",
        "chosen": "...",
        "rejected": "..."
    }
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            rows.append(
                {
                    "prompt": obj["prompt"],
                    "chosen": obj["chosen"],
                    "rejected": obj["rejected"],
                }
            )
    return Dataset.from_list(rows)
