# scripts/train_dpo.py
import argparse
from pathlib import Path

from trl import DPOTrainer, DPOConfig

from aligner.utils import load_yaml, set_seed, SimpleLogger, log_experiment
from aligner.models import load_model_and_tokenizer
from aligner.data import load_dpo_jsonl


def main(config_path: str):
    # 1. Load config and seed
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    logger = SimpleLogger()

    # 2. Load policy model (to be trained) and tokenizer
    # 注意：policy_model 已经是 SFT 过的 LoRA 模型，所以这里不要再用 LoRA 包一层
    policy_model, policy_tokenizer = load_model_and_tokenizer(
        cfg["policy_model"],
        use_lora=False,  # very important: avoid double-PEFT wrapping
    )

    # 3. Load reference model (fixed base model)
    ref_model, _ = load_model_and_tokenizer(
        cfg["reference_model"],
        use_lora=False,
    )

    # 4. Load DPO dataset (prompt, chosen, rejected)
    dataset = load_dpo_jsonl(cfg["dpo_file"])

    # 5. DPO training config
    training_args = DPOConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=int(cfg["batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        max_steps=int(cfg["max_steps"]),
        logging_steps=int(cfg["logging_steps"]),
        save_steps=int(cfg["save_steps"]),
        bf16=True,
        report_to=[],  # 不用 wandb 等
        beta=float(cfg["beta"]),  # DPO 的 beta 参数
    )

    # 6. Construct DPO trainer
    # 新版本 API 变化：使用 processing_class 而不是 tokenizer
    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=policy_tokenizer,  # ✅ 改用 processing_class
    )

    # 7. Train
    train_result = trainer.train()

    # 8. Save model & tokenizer
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    policy_tokenizer.save_pretrained(output_dir)

    final_loss = None
    if hasattr(train_result, "training_loss"):
        final_loss = float(train_result.training_loss)

    logger.log(
        {
            "status": "finished_dpo",
            "output_dir": str(output_dir),
            "final_loss": final_loss,
        }
    )

    # 记录 experiment
    log_experiment(
        kind="dpo",
        config=cfg,
        output_dir=str(output_dir),
        extra={"final_loss": final_loss},
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)