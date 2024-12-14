# %%
import json

from torch.utils.data import Dataset


class JSONLDataset(Dataset):
    def __init__(self, json_file):
        super(JSONLDataset, self).__init__()
        with open(json_file, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def map(self, function):
        self.data = [function(data) for data in self.data]
        return self


# %%

# %%
import json

from torch.utils.data import Dataset


class JSONLDataset(Dataset):
    def __init__(self, json_file):
        super(JSONLDataset, self).__init__()
        with open(json_file, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def map(self, function):
        self.data = [function(data) for data in self.data]
        return self


# %%

PROMPT_FORMAT = """<start_of_turn>user
{system}

{instruction}
<end_of_turn>
<start_of_turn>model
{output}
<end_of_turn>"""
# %%


def format_qa_to_prompt(data):
    system = "あなたはインターネットの相談掲示板回答者です。質問に対して適切なアドバイスをしてください。"
    instruction = f"## Question: {data['q_title']}\n\n{data['q_text']}"
    output = "## Answer: \n\n" + "\n\n".join(
        [f"ans{i}: {a_text}" for i, a_text in enumerate(data["a_texts"], start=1)]
    )
    full_prompt = PROMPT_FORMAT.format(
        system=system, instruction=instruction, output=output
    )
    return {"text": full_prompt}


# %%
# %%
import json

from torch.utils.data import Dataset


class JSONLDataset(Dataset):
    def __init__(self, json_file):
        super(JSONLDataset, self).__init__()
        with open(json_file, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def map(self, function):
        self.data = [function(data) for data in self.data]
        return self


# %%

PROMPT_FORMAT = """<start_of_turn>user
{system}

{instruction}
<end_of_turn>
<start_of_turn>model
{output}
<end_of_turn>"""
# %%


def format_qa_to_prompt(data):
    system = "あなたはインターネットの相談掲示板回答者です。質問に対して適切なアドバイスをしてください。"
    instruction = f"## Question: {data['q_title']}\n\n{data['q_text']}"
    output = "## Answer: \n\n" + "\n\n".join(
        [f"ans{i}: {a_text}" for i, a_text in enumerate(data["a_texts"], start=1)]
    )
    full_prompt = PROMPT_FORMAT.format(
        system=system, instruction=instruction, output=output
    )
    return {"text": full_prompt}


# %%
dataset = JSONLDataset("../data/goo_q_n_a/カップル・彼氏・彼女.jsonl")
train_dataset = dataset.map(format_qa_to_prompt)
train_dataset[0]


# %%
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


# データセットに（generate_text_fieldの処理を用いて）textフィールドを追加
train_dataset = dataset.map(generate_text_field)


# %%
dataset = JSONLDataset("../data/goo_q_n_a/カップル・彼氏・彼女.jsonl")
train_dataset = dataset.map(format_qa_to_prompt)
train_dataset[0]


# %%
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
