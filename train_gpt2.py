import os
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from tqdm import tqdm
import tiktoken


# setting the parameters

num_proc = 14

local_data_path = "./sample_data"

blocksize = 1024
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


example = {"text": "Make America Great Again"}
print(process(example))


class StreamingParquetDataset(torch.utils.data.IterableDataset):
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


arrow_files = [
    os.path.join(local_data_path, f)
    for f in os.listdir(local_data_path)
    if f.endswith(".arrow")
]


train_dataset = StreamingParquetDataset(arrow_files, "train", blocksize, num_proc)
val_dataset = StreamingParquetDataset(arrow_files[-1], "val", blocksize, num_proc)

train_dataloader = DataLoader(
    train_dataset, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=True
)
val_dataloader = DataLoader(
    val_dataset, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=True
)


def get_batch(dataloader, device):
    for x, y in dataloader:
        if device_type == "cuda":
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

        yield x, y


class config:
    block_size = 1024
    vocab_size = 50304
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0
    bias = False


wte = nn.Embedding(config.vocab_size, config.n_embd).to(device)
wpe = nn.Embedding(config.block_size, config.n_embd).to(device)
drop = nn.Dropout(config.dropout).to(device)
ln_f = nn.LayerNorm(config.n_embd, bias=config.bias).to(device)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_procj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_procj(x)
        x = self.dropout(x)
        return x


mlp = MLP(config).to(device)


for x, y in get_batch(train_dataloader, device):
    print(x.shape, y.shape)
    x_embd = wte(x)
    print(x_embd.shape)
    x_embd_ln = ln_f(x_embd)
    print(x_embd_ln.shape)
    x_mlp = mlp(x_embd_ln)
    print(x_mlp.shape)

    break


mask = torch.tril(torch.ones(4, 4)).view(1, 1, 4, 4)
print("mask matric:")
print(mask)

q = k = v = torch.rand(1, 4, 10)
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
print("att score")
print(att)

mask_att = att.masked_fill(mask[:, :, :4, :4] == 0, float("-inf"))
print("masked att")
print(mask_att)
