# scripts/train_sft.py
import argparse
from pathlib import Path

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from aligner.utils import load_yaml, set_seed, SimpleLogger, log_experiment
from aligner.models import load_model_and_tokenizer
from aligner.data import make_sft_dataset


def main(config_path: str):
    # 1. Load config and set seed
    cfg = load_yaml(config_path)
    set_seed(cfg["seed"])

    logger = SimpleLogger()

    # 2. Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(
        cfg["model_name"],
        use_lora=cfg.get("use_lora", False),
        lora_r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
    )

    # 3. Build dataset
    dataset = make_sft_dataset(
        cfg["sft_file"],
        tokenizer,
        cfg["max_seq_length"],
    )

    # 4. Data collator for language modeling
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 5. TrainingArguments
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        max_steps=cfg["max_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        bf16=True,
        report_to="none",
    )

    def compute_metrics(_):
        # For simplicity, we don't compute metrics here.
        return {}

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 7. Train
    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # 简单拿最后一个 loss（如果有的话）
    final_loss = None
    if hasattr(train_result, "training_loss"):
        final_loss = float(train_result.training_loss)

    logger.log(
        {
            "status": "finished_sft",
            "output_dir": str(output_dir),
            "final_loss": final_loss,
        }
    )

    # 记录 experiment 到 JSON
    log_experiment(
        kind="sft",
        config=cfg,
        output_dir=str(output_dir),
        extra={"final_loss": final_loss},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
