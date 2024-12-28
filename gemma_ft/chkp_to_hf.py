import os

from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def chkp_to_hf(chkp_path: str, model_name: str, tokenizer_name: str):
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    peft_model = PeftModel.from_pretrained(base_model, chkp_path)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(tokenizer_name)


if __name__ == "__main__":
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    chkp_to_hf(
        "model/checkpoint-1200",
        "google/gemma-2-2b-jpn-it",
        "google/gemma-2-2b-jpn-it-20241218",
    )
