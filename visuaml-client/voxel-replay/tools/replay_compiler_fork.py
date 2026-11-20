"""Fork a real terrain continuation at a verified local.get pair."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

import terrain_verify as terrain
from fork_divergence import first_divergence
from frozen_call_checkpoint import canonical, digest, encode, identities, restore
from replay_frozen_multistep import CALLS
from replay_frozen_recurrent import PACKED
from replay_guest_region_plan import OUT as PLAN, LOCAL_GET, prove, propose
from replay_guest_rewrite import EXECUTOR
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image

OUT = PLAN.parent / "shared-fork-search"


def observe(state, identity):
    vm = state.vm
    return dict(next_pc=int(vm.pc), stack=vm.stack[:int(vm.height)].tolist(), fuel=int(vm.fuel),
                state_sha256=encode(state, identity)["checkpoint_sha256"])


def capture(one, image, identity):
    scene = terrain.scale_session(8, False, 1)
    memory = terrain.initial_page(scene["initial"])
    terrain.write(memory, 320, [4 * move for move in scene["inputs"][0]["moves"]])
    state = initialize(image, terrain.BASE, memory, 5000)
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        while int(state.vm.status) == 0:
            pc = int(state.vm.pc)
            function = max(i for i, start in enumerate(image.starts) if start <= pc)
            offset = pc - image.starts[function]
            if image.functions[function][offset:offset+2] == ("local.get", "local.get"):
                slots = tuple(image.rows[pc:pc+2, 1].tolist())
                if state.vm.locals[slots[0]] != state.vm.locals[slots[1]]:
                    break
            state = one(image.rows, state).state
        else:
            raise AssertionError("Expected a live pair with distinct values")
        return encode(state, identity)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=OUT / "compiler-checkpoint.json")
    parser.add_argument("--session", type=Path, default=OUT / "compiler-session.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    graph = CALLS / "machine.pt"
    one = torch.jit.load(str(graph))
    image = load_call_image(terrain.GUEST.read_bytes())
    identity = identities(graph.read_bytes(), image)
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        if args.write:
            args.checkpoint.write_bytes(canonical(capture(one, image, identity))+b"\n")
        checkpoint = json.loads(args.checkpoint.read_text())
        entry = restore(checkpoint, identity, image)
        pc = int(entry.vm.pc)
        function = max(i for i, start in enumerate(image.starts) if start <= pc)
        offset = pc - image.starts[function]
        assert image.functions[function][offset:offset+2] == ("local.get", "local.get")
        slots = tuple(image.rows[pc:pc+2, 1].tolist())

        planner = (PLAN / "planner.wasm").read_bytes()
        baseline = torch.jit.load(str(EXECUTOR / "machine.pt"))
        source = [("local.get", slot) for slot in slots]
        accepted, rejected = (propose(baseline, planner, source, challenge=flag)["plan_words"]
                              for flag in (False, True))
        expected = [["unsat", "unsat"], ["sat", "unsat"]]
        for plan, verdict in zip((accepted, rejected), expected):
            assert [item["result"] for item in prove(slots, plan)] == verdict

        rows = image.rows[pc:pc+2]
        assert rows[:, 0].tolist() == [LOCAL_GET, LOCAL_GET]
        assert rows[:, 3:5].tolist() == [[pc+1, pc+1], [pc+2, pc+2]]
        assert not rows[:, 5:8].any() and not rows[:, 10].any()
        metadata = torch.full((len(image.rows), 2), -1, dtype=torch.int64)
        metadata[pc] = torch.tensor(accepted[1:3])
        region = torch.jit.load(str(PLAN / "machine.pt"))
        good = region(torch.cat((image.rows, metadata), dim=1),
                      restore(checkpoint, identity, image), 2)
        assert good.fused_pairs == 1

        normal, candidate = (restore(checkpoint, identity, image) for _ in range(2))
        assert normal.vm.page.data_ptr() != candidate.vm.page.data_ptr()
        reference = []
        for _ in slots:
            normal = one(image.rows, normal).state
            reference.append(observe(normal, identity))

        # Execute the rejected plan's observation contract. Its preview
        # reads the wrong slot; its region exit writes the planned final pair.
        assert observe(good.state, identity) == reference[-1]
        alternative = []
        vm = candidate.vm
        at = int(vm.height)
        for index, selected in enumerate(((rejected[3],), rejected[1:3]), 1):
            stack = vm.stack.clone()
            for i, slot in enumerate(selected):
                stack[at+i] = vm.locals[slot]
            preview = vm._replace(stack=stack, height=vm.height+index, pc=vm.pc+index,
                                  fuel=vm.fuel-rejected[3+index])
            alternative.append(observe(candidate._replace(vm=preview), identity))
        difference = first_divergence(reference, alternative)
        assert difference == 0 and reference[1] == alternative[1]
        assert first_divergence(reference, reference[:1]) == 1
        assert encode(candidate, identity) == checkpoint

    session = dict(version=1, checkpoint_sha256=checkpoint["checkpoint_sha256"],
                   identities=identity, planner_sha256=digest(planner),
                   source_slots=slots, accepted_plan=accepted, rejected_plan=rejected,
                   proof_results=expected, first_source_pc=pc+difference,
                   source_boundary=difference+1, reference=reference, candidate=alternative)
    if args.write:
        args.session.write_text(json.dumps(session, indent=2)+"\n")
    assert canonical(json.loads(args.session.read_text())) == canonical(session)
    print(f"First divergence after original PC {pc}: {reference[0]['stack']} vs {alternative[0]['stack']}; region exit matches.")


if __name__ == "__main__":
    main()
