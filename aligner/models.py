# aligner/models.py
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def load_model_and_tokenizer(
    model_name: str,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a causal LM and tokenizer. Optionally wrap the model with a LoRA adapter.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    if use_lora:
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    return model, tokenizer
