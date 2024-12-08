import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# huggingfaceトークンの設定（gemma2を使用するのに必要なため）
os.environ["HF_TOKEN"] = ""

# モデルのリポジトリIDを設定
repo_id = "google/gemma-2-2b-it"

# データセットのパス
dataset_path = "./zmn.jsonl"

# jsonlファイルを読み込む
json_data = []
with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        json_data.append(json.loads(line))

# DatasetオブジェクトにJSONデータを変換
dataset = Dataset.from_list(json_data)

# プロンプトフォーマット
PROMPT_FORMAT = """<start_of_turn>user
{system}

{instruction}
<end_of_turn>
<start_of_turn>model
{output}
<end_of_turn>
"""


# データセットの内容をプロンプトにセット → textフィールドとして作成する関数
def generate_text_field(data):
    messages = data["messages"]
    system = ""
    instruction = ""
    output = ""
    for message in messages:
        if message["role"] == "system":
            system = message["content"]
        elif message["role"] == "user":
            instruction = message["content"]
        elif message["role"] == "assistant":
            output = message["content"]
    full_prompt = PROMPT_FORMAT.format(
        system=system, instruction=instruction, output=output
    )
    return {"text": full_prompt}


# データセットに（generate_text_fieldの処理を用いて）textフィールドを追加
train_dataset = dataset.map(generate_text_field)

# messagesフィールドを削除
train_dataset = train_dataset.remove_columns(["messages"])

# モデルの読み込み
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=repo_id,  # モデルのリポジトリIDをセット
    device_map={"": "cuda"},  # 使用デバイスを設定
    attn_implementation="eager",  # 注意機構に"eager"を設定（Gemma2モデルの学習で推奨されているため）
)

# キャッシュを無効化（メモリ使用量を削減）
model.config.use_cache = False

# テンソル並列ランクを１に設定（テンソル並列化を使用しない）
model.config.pretraining_tp = 1

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=repo_id,  # モデルのリポジトリIDをセット
    attn_implementation="eager",  # 注意機構に"eager"を設定（Gemma2モデルの学習で推奨されているため）
    add_eos_token=True,  # EOSトークンの追加を設定
)

# パディングトークンが設定されていない場合、EOSトークンを設定
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# パディングを右側に設定(fp16を使う際のオーバーフロー対策)
tokenizer.padding_side = "right"

# LoRAのConfigを設定
Lora_config = LoraConfig(
    lora_alpha=8,  # LoRAによる学習の影響力を調整（スケーリング)
    lora_dropout=0.1,  # ドロップアウト率
    r=4,  # 低ランク行列の次元数
    bias="none",  # バイアスのパラメータ更新
    task_type="CAUSAL_LM",  # タスクの種別
)

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
    train_dataset=train_dataset,  # データセットをセット
    dataset_text_field="text",  # 学習に使用するデータセットのフィールド
    peft_config=Lora_config,  # LoRAのConfigをセット
    args=training_arguments,  # 学習パラメータをセット
    max_seq_length=512,  # 入力シーケンスの最大長を設定
)

# 正規化層をfloat32に変換(学習を安定させるため)
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

# モデルの学習
trainer.train()

# 学習したアダプターを保存
trainer.model.save_pretrained("./ずんだもん_Adapter")
