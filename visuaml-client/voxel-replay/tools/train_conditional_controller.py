"""Train a separate controlled-English controller to choose a follow-up tool."""
import json

import torch
from torch.nn import functional as F

import two_call_controller as base
from composed_inference import OUT, build

EXTRA = ["gamma", "below", "otherwise", "choose", "if"]


def reserved(seed, count):
    return (3 * seed + count) % 7 == 0


def corpus():
    rng = torch.Generator().manual_seed(2904)
    rows = []
    for seed in range(24):
        for count in range(6):
            if reserved(seed, count):
                continue
            for route in range(2):
                for case in range(24):
                    first = case
                    second = int(torch.randint(32, (), generator=rng))
                    low, high = ("beta", "gamma") if route == 0 else ("gamma", "beta")
                    prompt = (f"please first call alpha {seed} once . then choose {low} "
                              f"if below 12 otherwise {high} . {count} times . assistant")
                    initial = f"call alpha {seed} 1 ."
                    branch = low if first < 12 else high
                    if case == 0:
                        feedback, follow, result = "unavailable", "stop .", ""
                    elif case == 1:
                        feedback, follow, result = "exhausted", "stop .", ""
                    else:
                        feedback = f"result {first}"
                        follow = f"call {branch} {first} {count} ."
                        result = (" invalid assistant stop ." if count == 0 else
                                  f" result {second} assistant answer {second} .")
                    parts = [(prompt, False), (initial, True), (feedback + " assistant", False),
                             (follow, True)]
                    if result:
                        before, after = result.split("assistant")
                        parts.extend([(before + " assistant", False), (after, True)])
                    ids, labels = [], []
                    for text, train in parts:
                        tokens = base.encode(text)
                        ids.extend(tokens)
                        labels.extend(tokens if train else [-100] * len(tokens))
                    rows.append((ids, labels))
    size = max(map(lambda row: len(row[0]), rows))
    inputs = torch.zeros((len(rows), size - 1), dtype=torch.long)
    targets = torch.full_like(inputs, -100)
    for index, (ids, labels) in enumerate(rows):
        inputs[index, :len(ids)-1] = torch.tensor(ids[:-1])
        targets[index, :len(ids)-1] = torch.tensor(labels[1:])
    return inputs, targets


def main():
    torch.set_num_threads(1)
    torch.manual_seed(84)
    base.VOCAB.extend(EXTRA)
    base.IDS.update({word: index for index, word in enumerate(base.VOCAB)})
    inputs, labels = corpus()
    model = base.Controller()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    for _ in range(1500):
        selection = torch.randint(len(inputs), (48,))
        optimizer.zero_grad()
        loss = F.cross_entropy(model(inputs[selection]).flatten(0, 1), labels[selection].flatten())
        loss.backward()
        optimizer.step()
    controller = OUT / "conditional-controller.pt"
    torch.jit.freeze(torch.jit.trace(model.eval(), (inputs[:1],), check_trace=False)).save(str(controller))
    build(controller, OUT / "conditional.pt")
    (OUT / "training.json").write_text(json.dumps(dict(vocabulary=base.VOCAB,
        examples=len(inputs), updates=1500, seed=84, final_loss=float(loss.detach())), indent=2) + "\n")


if __name__ == "__main__":
    main()
