"""Train a tiny causal language model on textual tool-call transcripts."""
import json
from pathlib import Path
from time import perf_counter_ns

import torch
from torch import nn
from torch.nn import functional as F

OUT = Path(__file__).resolve().parents[1] / "evidence/two-call-controller"
WORDS = ["<pad>", "available", "alpha", "beta", "does", "offset", "boost", ".",
         "please", "kindly", "first", "then", "times", "assistant", "call", "result",
         "answer", "unavailable", "invalid", "exhausted", "stop", "once"]
VOCAB = WORDS + [str(i) for i in range(32)]
IDS = {word: index for index, word in enumerate(VOCAB)}
LENGTH = 64


def encode(text):
    return [IDS[word] for word in text.split()]


def reserved(seed, repeat):
    return (3 * seed + repeat) % 7 == 0


def prompt(polite, offset, boost, seed, repeat):
    return (f"available alpha does {'offset' if offset == 'alpha' else 'boost'} . "
            f"beta does {'offset' if offset == 'beta' else 'boost'} . "
            f"{polite} first offset {seed} once then boost {repeat} times . assistant")


class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(VOCAB), 48)
        self.positions = nn.Embedding(LENGTH, 48)
        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            48, 4, 96, dropout=0, batch_first=True, norm_first=True), 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(48)
        self.output = nn.Linear(48, len(VOCAB))

    def forward(self, ids: torch.Tensor):
        length = ids.size(1)
        pos = torch.arange(length, device=ids.device)
        mask = pos.unsqueeze(0) > pos.unsqueeze(1)
        hidden = self.embedding(ids) + self.positions(pos)
        return self.output(self.norm(self.blocks(hidden, mask=mask)))


def training_examples():
    examples = []
    rng = torch.Generator().manual_seed(9182)
    for seed in range(24):
        for repeat in range(6):
            if reserved(seed, repeat):
                continue
            for offset, boost in (("alpha", "beta"), ("beta", "alpha")):
                for polite in ("please", "kindly"):
                    prefix = prompt(polite, offset, boost, seed, repeat)
                    for case in range(8):
                        first, second = torch.randint(32, (2,), generator=rng).tolist()
                        begin = f" call {offset} {seed} 1 ."
                        if case == 0:
                            suffix = " unavailable assistant stop ."
                        else:
                            suffix = f" result {first} assistant call {boost} {first} {repeat} ."
                            if repeat == 0:
                                suffix += " invalid assistant stop ."
                            elif case == 1:
                                suffix += " exhausted assistant stop ."
                            else:
                                suffix += f" result {second} assistant answer {second} ."
                        examples.append((prefix, begin+suffix))
    size = max(len(encode(a+b)) for a, b in examples)
    inputs = torch.zeros((len(examples), size-1), dtype=torch.long)
    labels = torch.full_like(inputs, -100)
    for index, (prefix, suffix) in enumerate(examples):
        tokens = encode(prefix+suffix)
        start = len(encode(prefix))-1
        inputs[index, :len(tokens)-1] = torch.tensor(tokens[:-1])
        # Supervise generated responses only. The tool-result span is an input.
        for i in range(start, len(tokens)-1):
            preceding = VOCAB[tokens[i]]
            target = VOCAB[tokens[i+1]]
            if preceding == "assistant" or preceding in ("call", "answer", "stop", "alpha", "beta"):
                labels[index, i] = tokens[i+1]
            elif target == "." or (preceding.isdecimal() and target.isdecimal()):
                labels[index, i] = tokens[i+1]
    return inputs, labels


def train():
    torch.manual_seed(41)
    torch.set_num_threads(1)
    inputs, labels = training_examples()
    model = Controller()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    start = perf_counter_ns()
    for _ in range(1000):
        selection = torch.randint(len(inputs), (32,))
        loss = F.cross_entropy(model(inputs[selection]).flatten(0, 1), labels[selection].flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    elapsed = perf_counter_ns()-start
    model.eval()
    scripted = torch.jit.trace(model, (inputs[:1],), check_trace=False)
    torch.jit.freeze(scripted).save(str(OUT / "controller.pt"))
    (OUT / "training.json").write_text(json.dumps(dict(seed=41, examples=len(inputs), updates=1000,
             train_ns=elapsed, final_batch_loss=float(loss.detach()), vocabulary=VOCAB,
             torch=str(torch.__version__)), indent=2)+"\n")


if __name__ == "__main__":
    train()
