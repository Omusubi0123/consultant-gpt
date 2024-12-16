TRAIN_PROMPT_FORMAT = """<start_of_turn>user
{system}

{instruction}
<end_of_turn>
<start_of_turn>model
{output}
<end_of_turn>"""


INFERENCE_PROMPT_FORMAT = """<start_of_turn>user
{system}

{instruction}
<end_of_turn>
<start_of_turn>model
"""


def format_qa_to_prompt(data):
    system = "あなたはインターネットの相談掲示板回答者です。質問に対して適切なアドバイスをしてください。"
    instruction = f"## Question: {data['q_title']}\n\n{data['q_text']}"
    output = "## Answer: \n\n" + "\n\n".join(
        [f"ans{i}: {a_text}" for i, a_text in enumerate(data["a_texts"], start=1)]
    )
    full_prompt = TRAIN_PROMPT_FORMAT.format(
        system=system, instruction=instruction, output=output
    )
    return {"text": full_prompt}


def format_inference_prompt(question: str):
    system = "あなたはインターネットの相談掲示板回答者です。質問に対して適切なアドバイスをしてください。"
    instruction = f"## Question: \n\n{question}"
    instruction += "\n\n## Answer: "
    prompt = INFERENCE_PROMPT_FORMAT.format(system=system, instruction=instruction)
    return prompt
