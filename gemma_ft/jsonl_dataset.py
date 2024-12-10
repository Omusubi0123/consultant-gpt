import json

from torch.utils.data import Dataset


class JSONLDataset(Dataset):
    def __init__(self, json_file, batch_size: int):
        super(JSONLDataset, self).__init__()
        self.batch_size = batch_size
        with open(json_file, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def map(self, function, batched=False):
        if batched:
            batched_data = [
                self.data[i : i + self.batch_size]
                for i in range(0, len(self.data), self.batch_size)
            ]
            self.data = []
            for batch in batched_data:
                result = function(batch)
                self.data.extend(result)
        else:
            self.data = [function(data) for data in self.data]
        return self

    @classmethod
    def combine(cls, *datasets, batch_size: int):
        combined = cls.__new__(cls)
        combined.batch_size = batch_size
        combined.data = []
        for dataset in datasets:
            combined.data.extend(dataset.data)
        return combined
