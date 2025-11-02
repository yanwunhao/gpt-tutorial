import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import tiktoken
from torch.utils.data import Dataset, DataLoader

# setting the parameters

num_proc = 14

local_data_path = "./sample_data"

blocksize = 128
batchsize = 32

device_type = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

device = torch.device(device_type)

enc = tiktoken.get_encoding("gpt2")


def process(example):
    ids = enc.encode_ordinary(example["text"])
    ids.append(enc.eot_token)
    return {"ids": ids, "len": len(ids)}


example = {"text": "Gonsalves AI Lab"}
print(process(example))


class StreamingParquetDataset(torch.utils.IterableDataset):
    def __init__(self, data_files, split, block_size, num_proc=14):
        self.data_file = data_files
        self.split = split
        self.block_size = block_size
        self.dataset = load_dataset(
            "arrow", data_files={split: data_files}, streaming=True
        )
        self.tokenized = self.dataset[split].map(process, remove_columns=["text"])

    def __iter__(self):
        for example in self.tokenized:
            ids = example["ids"]
            for i in range(0, len(ids) - self.block_size, self.block_size):
                x = torch.tensor(ids[i : i + self.block_size], dtype=torch.int64)
                y = torch.tensor(
                    ids[i + 1 : i + 1 + self.block_size], dtype=torch.int64
                )
                yield x, y
