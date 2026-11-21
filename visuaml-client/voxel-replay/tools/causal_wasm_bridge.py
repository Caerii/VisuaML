"""Static lowering of a bounded single-function WASM profile to the frozen sparse decoder."""
from visuaml.compiler_lab.wasm_control_flow_v51 import decode_control

HALT, LOAD, STORE, ADD, BRNZ = range(5)
ZERO = 15
SLOTS = 16


def lower(raw):
    program = decode_control(raw, scaled=True)
    if program.local_count:
        raise ValueError("Locals outside the causal bridge profile")
    ops = program.operations
    names = [op.opcode for op in ops]
    labels, destinations = [], {}
    for pc, op in enumerate(ops):
        if op.opcode in ("block", "loop"):
            labels.append((op.opcode, pc))
        elif op.opcode == "end" and labels:
            destinations[labels.pop()] = pc + 1

    code, source, branches, entries = [], [], [], {}
    labels = []
    pc = 0
    # Each logical group consumes its stack inputs locally. The return load
    # is separately constrained to fall straight into the function's end.
    while pc < len(ops):
        entries[pc] = len(code)
        name = names[pc]
        if name in ("block", "loop"):
            labels.append((name, pc))
            pc += 1
            continue
        if name == "end" and labels:
            labels.pop()
            pc += 1
            continue
        start = pc
        emitted = []
        if names[pc:pc+3] == ["i32.const", "i32.const", "i32.store"]:
            address = word_address(ops[pc], ops[pc+2])
            emitted = [(LOAD, ZERO)]
            if ops[pc+1].immediate:
                emitted.append((ADD, ops[pc+1].immediate))
            emitted.append((STORE, address))
            pc += 3
        elif names[pc:pc+6] == ["i32.const", "i32.const", "i32.load",
                                "i32.const", "i32.add", "i32.store"]:
            emitted = [(LOAD, word_address(ops[pc+1], ops[pc+2])),
                       (ADD, ops[pc+3].immediate),
                       (STORE, word_address(ops[pc], ops[pc+5]))]
            pc += 6
        elif names[pc:pc+2] == ["i32.const", "i32.load"]:
            emitted = [(LOAD, word_address(ops[pc], ops[pc+1]))]
            pc += 2
            invert = names[pc:pc+2] == ["i32.eqz", "br_if"]
            if invert:
                pc += 1
            if names[pc:pc+1] == ["br_if"]:
                target = branch_target(labels, destinations, ops[pc].immediate)
                if invert:
                    # On a nonzero input skip the unconditional jump. On
                    # zero, add one to the known-zero accumulator and jump.
                    emitted.extend([(BRNZ, len(code)+4), (ADD, 1)])
                branches.append((len(code)+len(emitted), target))
                emitted.append((BRNZ, 0))
                pc += 1
            elif labels or pc != len(ops)-1:
                raise ValueError("Unconsumed WASM load")
        elif name == "br":
            target = branch_target(labels, destinations, ops[pc].immediate)
            branches.append((len(code)+2, target))
            emitted = [(LOAD, ZERO), (ADD, 1), (BRNZ, 0)]
            pc += 1
        elif name == "end" and pc == len(ops)-1 and names[pc-2:pc] == ["i32.const", "i32.load"]:
            emitted = [(HALT, 0)]
            pc += 1
        else:
            raise ValueError(f"Unsupported source group at {pc}: {name}")
        source.extend([[start, pc]] * len(emitted))
        code.extend([[opcode, 0, arg] for opcode, arg in emitted])
    for instruction, target in branches:
        if target not in entries or target > len(ops)-3:
            raise ValueError("Branch can bypass the required return load")
        code[instruction][2] = entries[target]
    if len(code) > SLOTS or not code or code[-1][0] != HALT:
        raise ValueError("Code outside the frozen 16-row image")
    return code, source, names


def word_address(base, access):
    address = base.immediate + access.immediate.offset
    if base.immediate < 0 or address % 4 or not 0 <= address < 4*ZERO:
        raise ValueError("Address outside the aligned fixed-word profile")
    return address // 4


def branch_target(labels, destinations, depth):
    if not 0 <= depth < len(labels):
        raise ValueError("Branch outside the closed source profile")
    label = labels[-1-depth]
    return label[1]+1 if label[0] == "loop" else destinations[label]
