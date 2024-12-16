import os

import torch
import torch.backends
from dotenv import load_dotenv
from fire import Fire
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline


def load_model(
    repo_id: str = "google/gemma-2-2b-it",
    lora_model_path: str = "./model/gemma-ft-{date}",
) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoPeftModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=lora_model_path,
        device_map="auto",
        # torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=repo_id,
        add_eos_token=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def llm_response(pipe, question) -> str:
    print("A: ", end="", flush=True)

    response = ""
    for chunk in pipe(
        question,
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
    ):
        print(chunk["generated_text"], end="", flush=True)
        response += chunk["generated_text"]

    print()
    return response


def inference_gemma(
    lora_model_path: str = "./model/gemma-ft-{date}",
    model_ft_date: str = "mmdd",
):
    model, tokenizer = load_model(
        lora_model_path=lora_model_path.format(date=model_ft_date)
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map={"": model.device},
        stream=True,
    )

    while True:
        question = input("Q: ")
        response = llm_response(pipe, question)
        print(f"A: {response}")


if __name__ == "__main__":
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

    Fire(inference_gemma)
