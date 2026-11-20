"""Assemble fixed finite control for the Pocket binary tape interpreter."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "evidence/pocket-universal/table.json"
SYMBOLS = ["_", "#", ";", "@", "-", "*", "!", "u", "v", "|",
           "0", "1", "a", "b", "L", "R", "M", "E", "J", "G", "H"]
OPS = "LRMEJGH"


def assemble():
    rows = {}

    def edge(state, alphabet, target, write=None, move="S"):
        for symbol in alphabet:
            key = state, symbol
            assert key not in rows, key
            rows[key] = target, symbol if write is None else write, move

    def scan(state, stop, target, move, write=None, across=None, stop_move=None):
        edge(state, stop, target, write, stop_move or move)
        edge(state, (s for s in (SYMBOLS if across is None else across) if s not in stop),
             state, move=move)

    # At a boundary the physical head sits on #, with one program counter
    # marker (*) and one simulated head marker (a or b).
    edge("boundary", "#", "fetch", move="R")
    scan("fetch", "*", "opcode", "R", across=";-LRMEJGHu")
    edge("opcode", "H", "halt")
    for op in "LRMEJ":
        edge("opcode", op, "data_" + op, move="R")
        scan("data_" + op, "ab", "execute_" + op, "R", stop_move="S")
    edge("opcode", "G", "jump_origin", move="L")
    for op, value in (("M", "b"), ("E", "a")):
        edge("execute_" + op, "ab", "advance", value, "L")
    for op in "LR":
        for marked, plain in (("a", "0"), ("b", "1")):
            edge("execute_" + op, marked, "mark_head", plain, op)
    edge("mark_head", "0_", "advance", "a", "L")
    edge("mark_head", "1", "advance", "b", "L")
    edge("mark_head", "|", "restore_edge", move="R")
    edge("restore_edge", "0", "left_edge", "a")
    edge("restore_edge", "1", "left_edge", "b")
    edge("execute_J", "a", "advance", move="L")
    edge("execute_J", "b", "jump_origin", move="L")
    scan("advance", "*", "next_record", "L", "-", stop_move="R")
    scan("next_record", ";", "activate", "R", across=OPS + "u")
    edge("activate", "-", "home", "*", "L")
    scan("home", "#", "boundary", "L", stop_move="S")

    # Jump counts the unary argument using u -> v. The destination cursor
    # @ walks from record zero and is separate from the jump origin !.
    scan("jump_origin", "*", "jump_home_init", "L", "!")
    scan("jump_home_init", "#", "first", "L", stop_move="R")
    edge("first", ";", "find_origin", "@", "R")
    scan("find_origin", "!", "skip_opcode", "R")
    edge("skip_opcode", "JG", "consume", move="R")
    edge("consume", "v", "consume", move="R")
    edge("consume", "u", "cursor_home", "v", "L")
    edge("consume", ";|", "restore", move="L")
    scan("cursor_home", "#", "find_cursor", "L", stop_move="R")
    scan("find_cursor", "@", "cursor_next", "R", ";")
    scan("cursor_next", ";", "origin_home", "R", "@", across="-!"+OPS+"uv", stop_move="L")
    scan("origin_home", "#", "find_origin", "L", stop_move="R")
    edge("restore", "v", "restore", "u", "L")
    edge("restore", "JG", "clear_origin", move="L")
    edge("clear_origin", "!", "final_home", "-", "L")
    scan("final_home", "#", "final_cursor", "L", stop_move="R")
    scan("final_cursor", "@", "activate", "R", ";")

    states = list(dict.fromkeys(["boundary", *(key[0] for key in rows),
                                *(value[0] for value in rows.values())]))
    rules = [" ".join(map(str, (states.index(state), SYMBOLS.index(symbol), states.index(target),
                                SYMBOLS.index(write), move)))
             for (state, symbol), (target, write, move) in rows.items()]
    return dict(states=states, symbols=SYMBOLS, rules="\n".join(rules))


if __name__ == "__main__":
    OUT.write_text(json.dumps(assemble(), indent=2) + "\n")
