import os

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def set_quantization_config():
    # モデル量子化の設定
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4ビット量子化を使用
        bnb_4bit_quant_type="nf4",  # 4ビット量子化の種類にnf4（NormalFloat4）を使用
        bnb_4bit_use_double_quant=True,  # 二重量子化を使用
        bnb_4bit_compute_dtype=torch.float16,  # 量子化のデータ型をfloat16に設定
    )
    return quantization_config


def load_model(repo_id: str = "google/gemma-2-2b-jpn-it") -> tuple:
    # モデルとトークナイザーをロード
    quantization_config = set_quantization_config()

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=repo_id,
        device_map="auto",
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


def chkp_to_hf(chkp_path: str, model_name: str, save_name: str):
    base_model, tokenizer = load_model(model_name)
    peft_model = PeftModel.from_pretrained(base_model, chkp_path)
    merged_model = peft_model.merge_and_unload()  # これをしないとconfig.jsonが出力されない
    merged_model.save_pretrained(save_name)
    tokenizer.save_pretrained(save_name)


if __name__ == "__main__":
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    chkp_to_hf(
        "model/gemma-ft-log-20241218-2048/checkpoint-1200",
        "google/gemma-2-2b-jpn-it",
        "model/gemma-2-2b-jpn-it-ft-20241218",
    )
