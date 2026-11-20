"""A bounded learned registry router composed with the unchanged synthesized VM."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from composed_inference import walk
from synthesize_attention_vm import SynthesizedVM

OUT = Path(__file__).resolve().parents[1] / "evidence/paired-routing"
VOCAB = ["<pad>", "offset", "shift", "seed", "start", "boost", "raise", "increase",
         "trim", "lower", "reduce", "the", "value", "input", "repeat", "a", "each", "by"]
DESCRIPTIONS = [
    ["offset the value", "shift input", "start by offset", "seed by shift input"],
    ["boost the value", "raise each repeat", "increase value", "repeat a boost"],
    ["trim the value", "lower each repeat", "reduce value", "repeat a trim"],
]


def description_ids(text):
    words = text.split()
    if not 1 <= len(words) <= 8:
        raise ValueError("Description has to fit the eight-token ABI")
    return [VOCAB.index(word) for word in words] + [0] * (8 - len(words))


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.words = nn.Embedding(len(VOCAB), 16, padding_idx=0)
        self.condition = nn.Sequential(nn.Linear(3, 24), nn.Tanh(), nn.Linear(24, 1))
        self.first_query = nn.Parameter(torch.randn(16))

    def encode(self, words: torch.Tensor):
        attributes = self.words(words) * (words != 0).unsqueeze(-1)
        return attributes.sum(dim=-2) / (words != 0).sum(dim=-1, keepdim=True).clamp_min(1)

    def first_scores(self, registry: torch.Tensor):
        return self.encode(registry) @ self.first_query

    def second_scores(self, registry: torch.Tensor, choices: torch.Tensor,
                      feedback: torch.Tensor, threshold: torch.Tensor, relation: torch.Tensor):
        # No exact comparison decides the outcome. The learned gate receives typed
        # numeric inputs and the relation selector and produces a soft routing score.
        inputs = torch.stack((feedback / 31 - 0.5, threshold / 31 - 0.5,
                              relation.to(torch.float32) - 0.5), dim=-1)
        preference = torch.sigmoid(self.condition(inputs))
        candidates = self.encode(choices)
        query = preference * candidates[..., 0, :] + (1 - preference) * candidates[..., 1, :]
        return (self.encode(registry) * query.unsqueeze(-2)).sum(dim=-1)


class Inference(nn.Module):
    def __init__(self, router):
        super().__init__()
        self.router = router
        self.executor = SynthesizedVM()

    def forward(self, descriptions: torch.Tensor, handles: torch.Tensor, schemas: torch.Tensor,
                programs: torch.Tensor, fuel: torch.Tensor, choices: torch.Tensor,
                seed: int, repeat: int, threshold: int, relation: int, override: int = -1,
                forced_second_handle: int = -1):
        # rows: handle, value, count, status (0 success), computed, reported, steps
        calls = torch.full((2, 7), -1, dtype=torch.long)
        traces = torch.zeros((2, 48, 8), dtype=torch.long)
        scores = torch.full((2, handles.numel()), float("nan"))
        value = seed
        for turn in range(2):
            if turn == 0:
                logits = self.router.first_scores(descriptions)
            else:
                logits = self.router.second_scores(descriptions, choices, torch.tensor(value),
                                                   torch.tensor(threshold), torch.tensor(relation))
            scores[turn] = logits
            # Stable tie break by the unique handle, independent of physical row order.
            ties = handles[logits == logits.max()]
            handle = int(ties.min())
            if turn == 1 and forced_second_handle >= 0:
                handle = forced_second_handle
            rows = (handles == handle).nonzero()
            count = 1 if turn == 0 else repeat
            status, computed, reported, steps = 1, -1, -1, 0
            if rows.numel() == 1:
                row = int(rows[0, 0])
                bounds = schemas[row]
                if int(bounds[0]) <= value <= int(bounds[1]) and int(bounds[2]) <= count <= int(bounds[3]):
                    budget = int(fuel[row])
                    if budget < 0:
                        status = 2
                    elif budget == 0:
                        status = 3
                    else:
                        number, trace = self.executor(programs[row], torch.tensor([value, count]), min(48, budget))
                        halt = (trace[:, 1] == 0).nonzero()
                        steps = trace.size(0)
                        status = 3
                        if halt.numel() != 0:
                            steps = int(halt[0, 0]) + 1
                            computed = int(number)
                            reported = override if turn == 0 and override >= 0 else computed
                            status = 0 if 0 <= reported <= 31 else 1
                        traces[turn, :steps] = trace[:steps]
            calls[turn] = torch.tensor([handle, value, count, status, computed, reported, steps])
            if status != 0:
                break
            value = reported
        return calls, traces, scores


def training_data():
    rng = torch.Generator().manual_seed(789)
    phrases = torch.tensor([[description_ids(text) for text in family] for family in DESCRIPTIONS])
    pairs_count = 7200
    threshold = torch.randint(1, 31, (pairs_count,), generator=rng)
    lower = (torch.rand(pairs_count, generator=rng) * threshold).long()
    higher = threshold + (torch.rand(pairs_count, generator=rng) * (32 - threshold)).long()
    adjacent = torch.rand(pairs_count, generator=rng) < 0.5
    lower = torch.where(adjacent, threshold - 1, lower)
    higher = torch.where(adjacent, threshold, higher)
    feedback = torch.cat((lower, higher))
    threshold = threshold.repeat(2)
    relation = torch.randint(2, (pairs_count,), generator=rng).repeat(2)
    order = torch.rand((pairs_count, 3), generator=rng).argsort(dim=1).repeat(2, 1)
    registry = phrases[order, torch.randint(4, (pairs_count, 3), generator=rng).repeat(2, 1)]
    query_order = torch.randint(2, (pairs_count,), generator=rng)
    pairs = torch.stack((query_order + 1, 2 - query_order), dim=1).repeat(2, 1)
    choices = phrases[pairs, torch.randint(4, (pairs_count, 2), generator=rng).repeat(2, 1)]
    first = (order == 0).long().argmax(dim=1)
    use_first = ((feedback < threshold) & (relation == 0)) | ((feedback >= threshold) & (relation == 1))
    correct_family = torch.where(use_first, pairs[:, 0], pairs[:, 1])
    second = (order == correct_family.unsqueeze(1)).long().argmax(dim=1)
    return registry, choices, feedback, threshold, relation, first, second


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(514)
    registry, choices, values, limits, relations, first, second = training_data()
    router = Router()
    optimizer = torch.optim.AdamW(router.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(1800):
        batch = torch.randint(len(values), (256,))
        loss = loss_fn(router.first_scores(registry[batch]), first[batch])
        loss += loss_fn(router.second_scores(registry[batch], choices[batch], values[batch],
                                            limits[batch], relations[batch]), second[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    router = router.eval()
    with torch.inference_mode():
        whole = router.second_scores(registry, choices, values, limits, relations).argmax(dim=-1)
        train_accuracy = float((whole == second).to(torch.float32).mean())
    args.output.mkdir(parents=True, exist_ok=True)
    module = torch.jit.freeze(torch.jit.script(Inference(router).eval()))
    kinds = sorted({node.kind() for node in walk(module.inlined_graph)})
    assert not {"prim::PythonOp", "prim::CallMethod"}.intersection(kinds)
    module.save(str(args.output / "inference.pt"))
    (args.output / "inference.ir").write_text(str(module.inlined_graph) + "\n")
    stats = dict(seed=514, row_count=len(values), paired_contexts=len(values) // 2, updates=1800, vocabulary=VOCAB,
                 descriptions=DESCRIPTIONS, training_accuracy=train_accuracy, operators=kinds)
    (args.output / "training.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(stats["training_accuracy"])


if __name__ == "__main__":
    main()
