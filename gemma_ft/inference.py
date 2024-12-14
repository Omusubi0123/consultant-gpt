import os

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


def load_model(
    repo_id: str = "google/gemma-2-2b-it",
    lora_model_path: str = "./model/gemma-ft-{date}",
) -> tuple:
    model = AutoPeftModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=lora_model_path,
        device_map={"": "cuda"},
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=repo_id,
        add_eos_token=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def llm_response(model, tokenizer, messages) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=1000,
    )

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def inference_gemma():
    model, tokenizer = load_model()

    while True:
        question = input("Q: ")

        messages = [{"role": "user", "content": question}]

        response = llm_response(model, tokenizer, messages)

        print(f"A: {response}")


if __name__ == "__main__":
    inference_gemma()
