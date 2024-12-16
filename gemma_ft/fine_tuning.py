import os
from datetime import date

import bitsandbytes as bnb
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

import wandb
from gemma_ft.gemma_format import format_qa_to_prompt
from gemma_ft.jsonl_dataset import JSONLDataset
from scraping.g_category_info import category_dict


def load_dataset(path: str = "./data/goo_q_n_a/{category}.jsonl") -> Dataset:
    # データセットをロード
    category_datasets = []
    for category in category_dict.keys():
        dataset = JSONLDataset(path.format(category=category))
        dataset = dataset.map(format_qa_to_prompt)
        category_datasets.append(dataset)

    dataset = JSONLDataset.combine(*category_datasets)
    dataset = Dataset.from_list([{"text": data["text"]} for data in dataset])
    return dataset


def set_quantization_config():
    # モデル量子化の設定
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4ビット量子化を使用
        bnb_4bit_quant_type="nf4",  # 4ビット量子化の種類にnf4（NormalFloat4）を使用
        bnb_4bit_use_double_quant=True,  # 二重量子化を使用
        bnb_4bit_compute_dtype=torch.float16,  # 量子化のデータ型をfloat16に設定
    )
    return quantization_config


def set_lora_config(target_modules):
    # LoRAの設定
    Lora_config = LoraConfig(
        lora_alpha=8,  # LoRAによる学習の影響力を調整（スケーリング)
        lora_dropout=0.1,  # ドロップアウト率
        r=4,  # 低ランク行列の次元数
        bias="none",  # バイアスのパラメータ更新
        task_type="CAUSAL_LM",  # タスクの種別
        target_modules=target_modules,  # 量子化対象のモジュール
    )
    return Lora_config


def find_all_linear_names(model):
    # モデルから4ビット量子化された線形層の名前を取得
    target_class = bnb.nn.Linear4bit
    linear_layer_names = set()
    for name_list, module in model.named_modules():
        if isinstance(module, target_class):
            names = name_list.split(".")
            layer_name = names[-1] if len(names) > 1 else names[0]
            linear_layer_names.add(layer_name)
    if "lm_head" in linear_layer_names:
        linear_layer_names.remove("lm_head")
    return list(linear_layer_names)


def load_model(repo_id: str = "google/gemma-2-2b-jpn-it") -> tuple:
    # モデルとトークナイザーをロード
    quantization_config = set_quantization_config()

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=repo_id,
        device_map={"": "cuda"},
        torch_dtype=torch.float16,
        attn_implementation="eager",
        quantization_config=quantization_config,
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


def set_normlayer_float32(trainer):
    # 正規化層をfloat32に変換(学習を安定させるため)
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)
    return trainer


def print_train_statusy(args: TrainingArguments):
    """deepspeedの"auto"設定の結果、batch_sizeなどがいくつで実行されたか確認"""
    print("------------------------------")
    print(f"Train Batch Size: {args.train_batch_size}")
    print(f"Per Device Train Batch Size: {args.per_device_train_batch_size}")
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    else:
        print("Running on CPU")
    print("------------------------------")


def fine_tuning(
    repo_id: str = "google/gemma-2-2b-jpn-it",
    dataset_path: str = "./data/goo_q_n_a/{category}.jsonl",
    save_dir: str = "./model/gemma-ft-{date}",
    output_dir: str = "./model/gemma-ft-log-{date}-{max_seq_length}",
    train_epoch: int = 100,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 64,
    max_grad_norm: float = 0.3,
    warmup_step_rate: float = 0.03,
    learning_rate: float = 5e-5,
    max_seq_length: int = 2048,
    wandb_project: str = "gemma-fine-tuning",
):
    print(f"device: {torch.cuda.is_available()}")

    dataset = load_dataset(dataset_path)
    train_val_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_val_dataset["train"]
    val_dataset = train_val_dataset["test"]

    # debug
    print("Dataset size: ", len(dataset))
    print("Train dataset size: ", len(train_dataset))
    print("Validation dataset size: ", len(val_dataset))

    model, tokenizer = load_model(repo_id)

    lora_target_modules = find_all_linear_names(model)
    lora_config = set_lora_config(lora_target_modules)

    today = date.today().strftime("%Y%m%d")

    substantial_batch_size = (
        per_device_train_batch_size
        * torch.cuda.device_count()
        * gradient_accumulation_steps
    )
    wandb.init(
        project=wandb_project,
        name=f"gemma-ft-{today}-{max_seq_length}-{substantial_batch_size}-{train_epoch}-{learning_rate}",
    )

    print("Start fine-tuning")
    training_arguments = TrainingArguments(
        output_dir=output_dir.format(date=today, max_seq_length=max_seq_length),
        fp16=True,  # fp16を使用
        eval_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=100,
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
        eval_dataset=val_dataset,
        dataset_text_field="text",
        peft_config=lora_config,
        args=training_arguments,
        max_seq_length=max_seq_length,
    )

    trainer = set_normlayer_float32(trainer)

    print_train_statusy(training_arguments)

    trainer.train()
    trainer.model.save_pretrained(save_dir.format(date=today))

    wandb.finish()


if __name__ == "__main__":
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

    torch.cuda.empty_cache()
    os.environ[
        "PYTORCH_CUDA_ALLOC_CONF"
    ] = "garbage_collection_threshold:0.8,max_split_size_mb:64"

    fine_tuning()
