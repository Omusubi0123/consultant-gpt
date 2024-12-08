import os
from datetime import date

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

import wandb
from gemma_ft.gemma_format import format_qa_to_prompt
from gemma_ft.jsonl_dataset import JSONLDataset
from scraping.g_category_info import category_dict


def load_dataset(path: str = "./data/goo_q_n_a/{category}.jsonl") -> Dataset:
    category_datasets = []
    for category in category_dict.keys():
        dataset = JSONLDataset(path.format(category=category))
        dataset = dataset.map(format_qa_to_prompt)
        category_datasets.append(dataset)

    dataset = JSONLDataset.combine(*category_datasets)
    return dataset


def load_model(repo_id: str = "google/gemma-2-2b-jpn-it") -> tuple:
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=repo_id,
        # device_map={"": "cuda"},
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
        token=os.environ["HF_TOKEN"],
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=repo_id,
        attn_implementation="eager",
        add_eos_token=True,
        token=os.environ["HF_TOKEN"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def set_lora_config():
    Lora_config = LoraConfig(
        lora_alpha=8,  # LoRAによる学習の影響力を調整（スケーリング)
        lora_dropout=0.1,  # ドロップアウト率
        r=4,  # 低ランク行列の次元数
        bias="none",  # バイアスのパラメータ更新
        task_type="CAUSAL_LM",  # タスクの種別
    )
    return Lora_config


def fine_tuning(
    repo_id: str = "google/gemma-2-2b-jpn-it",
    dataset_path: str = "./data/goo_q_n_a/{category}.jsonl",
    save_dir: str = "./model/gemma-ft-{date}",
    output_dir: str = "./model/gemma-ft-log-{date}",
    train_epoch: int = 4,
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 8,
    max_grad_norm: float = 0.3,
    warmup_step_rate: float = 0.03,
    learning_rate: float = 5e-5,
    max_seq_length: int = 8192,
    wandb_project: str = "gemma-fine-tuning",
):
    print(f"device: {torch.cuda.is_available()}")

    print("Load dataset")
    dataset = load_dataset(dataset_path)

    print("Load model")
    model, tokenizer = load_model(repo_id)
    lora_config = set_lora_config()

    today = date.today().strftime("%Y%m%d")

    print("Set wandb project: ", wandb_project)
    wandb.init(project=wandb_project)

    print("Start fine-tuning")
    training_arguments = TrainingArguments(
        output_dir=output_dir.format(date=today),
        fp16=True,  # fp16を使用
        logging_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=train_epoch,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="paged_adamw_32bit",  # 最適化アルゴリズム
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",  # WarmupCosineLR
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_step_rate,
        weight_decay=0.001,  # 重み減衰率
        group_by_length=True,  # シーケンスの長さが近いものをまとめてバッチ化
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        peft_config=lora_config,
        args=training_arguments,
        max_seq_length=max_seq_length,
    )

    trainer.train()
    trainer.model.save_pretrained(save_dir.format(date=today))

    wandb.finish()


if __name__ == "__main__":
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
    fine_tuning()
