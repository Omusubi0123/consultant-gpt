import os

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from fine_tuning.gemma_format import format_qa_to_prompt
from fine_tuning.jsonl_dataset import JSONLDataset
from scraping.g_category_info import category_dict

# huggingfaceトークンの設定（gemma2を使用するのに必要なため）
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


def load_dataset(path: str = "./data/goo_q_n_a/{category}.jsonl") -> Dataset:
    category_datasets = []
    for category in category_dict.keys():
        dataset = JSONLDataset(path.format(category=category))
        dataset = dataset.map(format_qa_to_prompt)
        category_datasets.append(dataset)

    dataset = JSONLDataset.combine(*category_datasets)
    return dataset


def load_model(repo_id: str = "google/gemma-2-2b-it") -> tuple:
    # Decoder Model
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=repo_id,
        device_map={"": "cuda"},
        attn_implementation="eager",
    )
    # キャッシュを無効化（メモリ使用量を削減）
    model.config.use_cache = False
    # テンソル並列ランクを１に設定（テンソル並列化を使用しない）
    model.config.pretraining_tp = 1

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=repo_id,
        attn_implementation="eager",
        add_eos_token=True,
    )
    # パディングトークンが設定されていない場合、EOSトークンを設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # パディングを右側に設定(fp16を使う際のオーバーフロー対策)
    tokenizer.padding_side = "right"

    return model, tokenizer


def set_lora_config():
    # LoRAのConfigを設定
    Lora_config = LoraConfig(
        lora_alpha=8,  # LoRAによる学習の影響力を調整（スケーリング)
        lora_dropout=0.1,  # ドロップアウト率
        r=4,  # 低ランク行列の次元数
        bias="none",  # バイアスのパラメータ更新
        task_type="CAUSAL_LM",  # タスクの種別
    )
    return Lora_config


def fine_tuning():
    dataset = load_dataset()
    model, tokenizer = load_model()
    lora_config = set_lora_config()

    # 学習パラメータを設定
    training_arguments = TrainingArguments(
        output_dir="./train_logs",  # ログの出力ディレクトリ
        fp16=True,  # fp16を使用
        logging_strategy="epoch",  # 各エポックごとにログを保存（デフォルトは"steps"）
        save_strategy="epoch",  # 各エポックごとにチェックポイントを保存（デフォルトは"steps"）
        num_train_epochs=3,  # 学習するエポック数
        per_device_train_batch_size=1,  # （GPUごと）一度に処理するバッチサイズ
        gradient_accumulation_steps=4,  # 勾配を蓄積するステップ数
        optim="paged_adamw_32bit",  # 最適化アルゴリズム
        learning_rate=5e-4,  # 初期学習率
        lr_scheduler_type="cosine",  # 学習率スケジューラの種別
        max_grad_norm=0.3,  # 勾配の最大ノルムを制限（クリッピング）
        warmup_ratio=0.03,  # 学習を増加させるウォームアップ期間の比率
        weight_decay=0.001,  # 重み減衰率
        group_by_length=True,  # シーケンスの長さが近いものをまとめてバッチ化
        report_to="tensorboard",  # TensorBoard使用してログを生成（"./train_logs"に保存）
    )

    # SFTパラメータの設定
    trainer = SFTTrainer(
        model=model,  # モデルをセット
        tokenizer=tokenizer,  # トークナイザーをセット
        train_dataset=dataset,  # データセットをセット
        dataset_text_field="text",  # 学習に使用するデータセットのフィールド
        peft_config=lora_config,  # LoRAのConfigをセット
        args=training_arguments,  # 学習パラメータをセット
        max_seq_length=8192,  # 入力シーケンスの最大長を設定
    )

    # 正規化層をfloat32に変換(学習を安定させるため)
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    # モデルの学習
    trainer.train()

    # 学習したアダプターを保存
    trainer.model.save_pretrained("./ずんだもん_Adapter")
