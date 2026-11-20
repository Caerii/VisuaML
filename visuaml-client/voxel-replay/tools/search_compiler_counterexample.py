"""Search ordinary input prefixes for a rejected local-read preview."""
import argparse
import json
import struct
from pathlib import Path
from unittest.mock import patch

import terrain_verify as terrain
from fork_divergence import first_divergence
from frozen_call_checkpoint import canonical, digest, encode, identities, restore
from replay_compiler_fork import observe, PLAN, EXECUTOR, LOCAL_GET, propose, prove, torch, CALLS
from replay_frozen_recurrent import PACKED, wasmtime_run
from replay_voxel_call_checkpoint import OUT as PASSAGE
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image

OUT = PASSAGE.parent / "counterexample-search"
CASES = {"terrain": (terrain.GUEST, 100, 1, terrain.BASE),
         "passage": (PASSAGE / "passage.wasm", 0, 1, 65512)}
FUEL = 5000


class Experiment:
    def __init__(self, name):
        self.name = name
        path, self.site, wrong, self.argument = CASES[name]
        self.raw = path.read_bytes()
        self.image = load_call_image(self.raw)
        if name == "passage":
            self.site = self.image.entry
        graph = CALLS / "machine.pt"
        self.identity = identities(graph.read_bytes(), self.image)
        self.one = torch.jit.load(str(graph))
        rows = self.image.rows[self.site:self.site+2]
        self.slots = tuple(rows[:, 1].tolist())
        assert rows[:, 3:5].tolist() == [[self.site+1]*2, [self.site+2]*2]
        assert rows[:, 0].tolist() == [LOCAL_GET, LOCAL_GET]
        assert not rows[:, 5:8].any() and not rows[:, 10].any()
        planner = (PLAN / "planner.wasm").read_bytes()
        self.good = propose(torch.jit.load(str(EXECUTOR / "machine.pt")), planner,
                            [("local.get", s) for s in self.slots])["plan_words"]
        self.bad = (*self.good[:3], wrong, *self.good[4:])
        self.planner_hash = digest(planner)
        self.prepared = torch.full((len(self.image.rows), 2), -1, dtype=torch.int64)
        self.prepared[self.site] = torch.tensor(self.good[1:3])
        self.prepared = torch.cat((self.image.rows, self.prepared), dim=1)
        self.region = torch.jit.load(str(PLAN / "machine.pt"))

    def initial(self):
        if self.name == "terrain":
            page = terrain.initial_page(terrain.scale_session(8, False, 1)["initial"])
        else:
            page = bytearray(65536)
            struct.pack_into("<6i", page, self.argument, 7, 0, 0, 0, -1, 0)
        return encode(initialize(self.image, self.argument, page, FUEL), self.identity)

    def tick(self, checkpoint, action, rejected):
        state = restore(checkpoint, self.identity, self.image)
        memory = state.vm.page.numpy()
        if self.name == "terrain":
            struct.pack_into("<i", memory, self.argument+320, action*4)
        else:
            struct.pack_into("<i", memory, self.argument+4, action)
        initial = memory.tobytes()
        first = None
        visits = 0
        while int(state.vm.status) == 0:
            pc = int(state.vm.pc)
            if pc != self.site:
                state = self.one(self.image.rows, state).state
                continue
            visits += 1
            root = encode(state, self.identity)
            other = restore(root, self.identity, self.image)
            fused = self.region(self.prepared, other, 2)
            assert fused.fused_pairs == 1
            one = self.one(self.image.rows, state).state
            state = self.one(self.image.rows, one).state
            vm = other.vm
            plan = self.bad if rejected else self.good
            stack = vm.stack.clone()
            stack[int(vm.height)] = vm.locals[plan[3]]
            candidate = other._replace(vm=vm._replace(stack=stack, height=vm.height+1,
                                                      pc=vm.pc+1, fuel=vm.fuel-1))
            reference = [observe(one, self.identity), observe(state, self.identity)]
            alternative = [observe(candidate, self.identity), observe(fused.state, self.identity)]
            assert reference[1] == alternative[1]
            delta = first_divergence(reference, alternative)
            if delta is not None and first is None:
                first = dict(pc=pc+delta, source_step=FUEL-int(vm.fuel)+delta+1,
                             reference=reference[delta], candidate=alternative[delta],
                             region_exit_sha256=reference[1]["state_sha256"])
        final = state.vm.page.numpy().tobytes()
        result = int(state.vm.stack[0])
        assert int(state.vm.status) == 1
        assert wasmtime_run(self.raw, self.argument, initial) == (result, final)
        next_root = encode(initialize(self.image, self.argument, final, FUEL), self.identity)
        return next_root, dict(visited=visits, first_difference=first, page_sha256=digest(final))

    def search(self, root, rejected):
        results = []
        def visit(prefix, checkpoint):
            for bit in (0, 1):
                actions = (*prefix, bit)
                next_root, detail = self.tick(checkpoint, bit, rejected)
                results.append(dict(actions=list(actions), **detail))
                if len(actions) < 2 and detail["first_difference"] is None:
                    visit(actions, next_root)
        visit((), root)
        failures = [r for r in results if r["first_difference"] is not None]
        winner = min(failures, key=lambda r: (sum(r["actions"]), len(r["actions"]), r["actions"]), default=None)
        return dict(executed_prefixes=results, winner=winner)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--session", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        if args.session:
            session = json.loads(args.session.read_text())
            run = Experiment(session["guest"])
            root = session["checkpoint"]
            assert session["plan_sha256"] == digest(canonical(run.bad))
            winner = session["bad"]["winner"]
            for action in winner["actions"]:
                root, detail = run.tick(root, action, True)
            assert detail == {k: v for k, v in winner.items() if k != "actions"}
            print(json.dumps(detail["first_difference"]))
            return
        for name in CASES:
            run = Experiment(name)
            root = run.initial()
            bad, good = run.search(root, True), run.search(root, False)
            assert bad["winner"] and good["winner"] is None
            assert [r["result"] for r in prove(run.slots, run.good)] == ["unsat"]*2
            assert [r["result"] for r in prove(run.slots, run.bad)] == ["sat", "unsat"]
            session = dict(version=1, guest=name, checkpoint=root, plan=list(run.bad),
                           plan_sha256=digest(canonical(run.bad)), planner_sha256=run.planner_hash,
                           search=dict(alphabet=[0, 1], maximum_ticks=2, baseline=0),
                           bad=bad, corrected=good)
            if name == "terrain":
                assert bad["winner"]["actions"] == [1]
                assert any(r["actions"] == [0, 0] and r["first_difference"] is None
                           for r in bad["executed_prefixes"])
            path = OUT / f"{name}-session.json"
            if args.write:
                path.write_text(json.dumps(session, indent=2)+"\n")
            assert json.loads(path.read_text()) == session
            print(name, bad["winner"], len(good["executed_prefixes"]), "corrected prefixes")


if __name__ == "__main__":
    main()
