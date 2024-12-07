import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# Hugging Faceトークンの設定
os.environ["HF_TOKEN"] = "your_huggingface_token_here"

# モデルとトークナイザーの設定
model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])
tokenizer.pad_token = tokenizer.eos_token

# モデルのロード
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", token=os.environ["HF_TOKEN"]
)


# カスタムデータセットの読み込み
def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)


dataset = load_dataset("path_to_your_dataset.jsonl")

# LoRA設定
lora_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

# トレーニング引数の設定
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,
    save_total_limit=2,
    logging_steps=100,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="none",
    load_best_model_at_end=True,
    optim="adamw_torch",
)

# トレーナーの設定
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    args=training_args,
    tokenizer=tokenizer,
    max_seq_length=512,
)

# トレーニングの実行
trainer.train()

# モデルの保存
trainer.model.save_pretrained("./gemma_2b_jpn_it_finetuned")
