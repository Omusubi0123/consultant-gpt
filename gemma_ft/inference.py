import os
import sys
import threading

import torch
import torch.backends
from dotenv import load_dotenv
from fire import Fire
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, TextIteratorStreamer


def load_model(
    repo_id: str = "google/gemma-2-2b-it",
    lora_model_path: str = "./model/gemma-ft-{date}",
) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoPeftModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=lora_model_path,
        device_map="cpu",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=repo_id,
        add_eos_token=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def llm_response(model, tokenizer, question) -> str:
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    inputs = tokenizer(question, return_tensors="pt").to(model.device)

    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
    )

    print("A: ", end="", flush=True)
    generation_thread = threading.Thread(
        target=model.generate, kwargs=generation_kwargs
    )
    generation_thread.start()

    generated_text = ""
    try:
        for new_token in streamer:
            print(new_token, end="", flush=True)
            generated_text += new_token
    except Exception as e:
        print(f"\nError during streaming: {e}", file=sys.stderr)

    generation_thread.join()

    print()
    return generated_text


def inference_gemma(
    lora_model_path: str = "./model/gemma-ft-{date}",
    model_ft_date: str = "mmdd",
):
    model, tokenizer = load_model(
        lora_model_path=lora_model_path.format(date=model_ft_date)
    )

    while True:
        try:
            question = input("Q: ")
            if question.lower() in ["exit", "quit", "q"]:
                break

            response = llm_response(model, tokenizer, question)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Type 'exit' to quit.")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

    Fire(inference_gemma)
