"""Complete numeric workloads for the bounded packed WASM scale profile."""
from __future__ import annotations

import struct
from collections import deque
from random import Random
from functools import lru_cache

from scripts.verify_memory_recovery_v58 import initial_page

SUDOKU_PEERS = 512
SUDOKU_BLANKS = 7168
SUDOKU_FORBIDDEN = 8192
SUDOKU_DECISIONS = 12000
SUDOKU_CHOICES = 12700
GRAPH_SIZE = 64
GRAPH_QUEUE = 0
GRAPH_DISTANCE = 256
GRAPH_EDGES = 512
MATRIX_LIMIT = 32
MATRIX_B = 4096
MATRIX_C = 8192

PUZZLES = {
    "classic": "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
    "norvig_grid": "400000805030000000000700000020000060000080400000010000000603070500200000104000000",
    "escargot": "100007090030020008009600500005300900010080002600004000300000010040000007007000300",
    "euler_grid": "003020600900305001001806400008102900700000008006708200002609500800203009005010300",
}


def module(body):
    return f'''
(module
  (func (export "run") (param i32) (result i32)
    (local i32 i32 i32 i32)
    {body}
  )
  (memory (export "memory") 1 1))
'''


def sudoku_wat():
    # local 1 = decision depth, 2 = cell address, 3 = proposed digit,
    # 4 = cursor used during the blank or peer scan.
    decision = f"local.get 1 i32.const 8 i32.mul i32.const {SUDOKU_DECISIONS} i32.add"
    peer_start = f"local.get 2 i32.const 20 i32.mul i32.const {SUDOKU_PEERS} i32.add"
    peer_limit = f"local.get 2 i32.const 20 i32.mul i32.const {SUDOKU_PEERS + 80} i32.add"
    peer_entry = "local.get 4 i32.load i32.const 10 i32.mul local.get 3 i32.const 4 i32.mul i32.add"

    def update_peers(delta):
        return f"""
        {peer_start} local.set 4
        loop
          {peer_entry} {peer_entry} i32.load offset={SUDOKU_FORBIDDEN}
          i32.const {delta} i32.add i32.store offset={SUDOKU_FORBIDDEN}
          local.get 4 i32.const 4 i32.add local.set 4
          local.get 4 {peer_limit} i32.lt_s br_if 0
        end
        """

    choices = ""  # Build a left-associated sum on the WASM stack.
    for digit in range(1, 10):
        choices += (
            f" local.get 4 i32.load i32.const 10 i32.mul "
            f"i32.load offset={SUDOKU_FORBIDDEN + digit * 4} i32.eqz"
        )
        if digit > 1:
            choices += " i32.add"

    return module(f"""
    loop
      local.get 1 local.get 0 i32.sub i32.eqz
      if i32.const 1 return end
      local.get 1 i32.const 0 i32.lt_s
      if i32.const 0 return end
      {decision} i32.load local.set 2
      local.get 2 i32.const 0 i32.lt_s
      if
        i32.const 10 local.set 3
        i32.const {SUDOKU_BLANKS} local.set 4
        block
          loop
            local.get 4 local.get 0 i32.const 4 i32.mul
            i32.const {SUDOKU_BLANKS} i32.add i32.lt_s i32.eqz br_if 1
            local.get 4 i32.load i32.load i32.eqz
            if
              i32.const {SUDOKU_CHOICES} {choices} i32.store
              i32.const {SUDOKU_CHOICES} i32.load local.get 3 i32.lt_s
              if
                local.get 4 i32.load local.set 2
                i32.const {SUDOKU_CHOICES} i32.load local.set 3
              end
              local.get 3 i32.const 2 i32.lt_s br_if 2
            end
            local.get 4 i32.const 4 i32.add local.set 4
            br 0
          end
        end
        {decision} local.get 2 i32.store
        i32.const 1 local.set 3
      else
        {decision} i32.load offset=4 local.set 3
        {update_peers(-1)}
        local.get 2 i32.const 0 i32.store
        local.get 3 i32.const 1 i32.add local.set 3
      end
      block
        loop
          local.get 3 i32.const 9 i32.gt_s br_if 1
          local.get 2 i32.const 10 i32.mul local.get 3 i32.const 4 i32.mul i32.add
          i32.load offset={SUDOKU_FORBIDDEN} i32.eqz
          if
            local.get 2 local.get 3 i32.store
            {decision} local.get 3 i32.store offset=4
            {update_peers(1)}
            local.get 1 i32.const 1 i32.add local.set 1
            br 3
          end
          local.get 3 i32.const 1 i32.add local.set 3
          br 0
        end
      end
      {decision} i32.const -1 i32.store
      local.get 1 i32.const 1 i32.sub local.set 1
      br 0
    end
    i32.const 0
    """)


def bfs_wat():
    # local 1 = queue head, 2 = tail, 3 = current vertex, 4 = neighbor.
    return module(f"""
    i32.const {GRAPH_QUEUE} local.get 0 i32.store
    local.get 0 i32.const 4 i32.mul i32.const 0 i32.store offset={GRAPH_DISTANCE}
    i32.const 1 local.set 2
    block
      loop
        local.get 1 local.get 2 i32.lt_s i32.eqz br_if 1
        local.get 1 i32.const 4 i32.mul i32.load offset={GRAPH_QUEUE} local.set 3
        i32.const 0 local.set 4
        block
          loop
            local.get 4 i32.const {GRAPH_SIZE} i32.lt_s i32.eqz br_if 1
            local.get 3 i32.const {GRAPH_SIZE} i32.mul local.get 4 i32.add
            i32.const 4 i32.mul i32.load offset={GRAPH_EDGES}
            if
              local.get 4 i32.const 4 i32.mul i32.load offset={GRAPH_DISTANCE} i32.const 0 i32.lt_s
              if
                local.get 4 i32.const 4 i32.mul
                local.get 3 i32.const 4 i32.mul i32.load offset={GRAPH_DISTANCE}
                i32.const 1 i32.add i32.store offset={GRAPH_DISTANCE}
                local.get 2 i32.const 4 i32.mul local.get 4 i32.store offset={GRAPH_QUEUE}
                local.get 2 i32.const 1 i32.add local.set 2
              end
            end
            local.get 4 i32.const 1 i32.add local.set 4
            br 0
          end
        end
        local.get 1 i32.const 1 i32.add local.set 1
        br 0
      end
    end
    local.get 2
    """)


def matrix_wat(unroll=1):
    # Input argument is square dimension. Locals hold i, j, k, and sum.
    def address(row, column):
        return f"local.get {row} local.get 0 i32.mul local.get {column} i32.add i32.const 4 i32.mul"

    inner = f"""
      local.get 4
      {address(1, 3)} i32.load
      {address(3, 2)} i32.load offset={MATRIX_B}
      i32.mul i32.add local.set 4
      local.get 3 i32.const 1 i32.add local.set 3
    """
    return module(f"""
    block
      loop
        local.get 1 local.get 0 i32.lt_s i32.eqz br_if 1
        i32.const 0 local.set 2
        block
          loop
            local.get 2 local.get 0 i32.lt_s i32.eqz br_if 1
            i32.const 0 local.set 3
            i32.const 0 local.set 4
            block
              loop
                local.get 3 local.get 0 i32.lt_s i32.eqz br_if 1
                {inner * unroll}
                br 0
              end
            end
            {address(1, 2)} local.get 4 i32.store offset={MATRIX_C}
            local.get 2 i32.const 1 i32.add local.set 2
            br 0
          end
        end
        local.get 1 i32.const 1 i32.add local.set 1
        br 0
      end
    end
    local.get 0
    """)


@lru_cache(maxsize=1)
def peers():
    return tuple(tuple(other for other in range(81) if cell != other and (
        cell // 9 == other // 9 or cell % 9 == other % 9
        or (cell // 27, cell % 9 // 3) == (other // 27, other % 9 // 3)
    )) for cell in range(81))


def sudoku_page(board):
    if (len(board) != 81 or any(type(value) is not int or not 0 <= value <= 9 for value in board)
            or any(value and any(board[j] == value for j in peers()[i]) for i, value in enumerate(board))):
        raise ValueError("Expected 81 Sudoku cells with consistent givens")
    # Initialize topology and counts implied by givens. WASM maintains
    # these counts, selects the next cell, and makes every search decision.
    page = initial_page(board)
    for index, neighbors in enumerate(peers()):
        struct.pack_into("<20i", page, SUDOKU_PEERS + index * 80, *(4 * item for item in neighbors))
        counts = [0] * 10
        for neighbor in neighbors:
            if board[neighbor]:
                counts[board[neighbor]] += 1
        struct.pack_into("<10i", page, SUDOKU_FORBIDDEN + index * 40, *counts)
        struct.pack_into("<ii", page, SUDOKU_DECISIONS + index * 8, -1, 0)
    order = sorted((i for i, value in enumerate(board) if value == 0),
                   key=lambda i: (-len({board[j] for j in peers()[i]} - {0}), i))
    for index, cell in enumerate(order):
        struct.pack_into("<i", page, SUDOKU_BLANKS + index * 4, 4 * cell)
    return page, len(order)


def sudoku_reference(board):
    """Independent dynamic minimum-candidate backtracking, capped at two solutions."""
    work = list(board)
    solutions = []

    def visit():
        best = None
        for index, value in enumerate(work):
            if value:
                continue
            choices = set(range(1, 10)) - {work[i] for i in peers()[index]}
            if not choices:
                return
            if best is None or len(choices) < len(best[1]):
                best = index, choices
        if best is None:
            solutions.append(tuple(work))
            return
        index, choices = best
        for value in sorted(choices):
            work[index] = value
            visit()
            if len(solutions) == 2:
                break
        work[index] = 0

    visit()
    return tuple(solutions)

def sudoku_variant(board, seed):
    rng = Random(seed)
    digits = [0, *rng.sample(range(1, 10), 9)]
    groups = rng.sample(range(3), 3)
    rows = [group * 3 + row for group in groups for row in rng.sample(range(3), 3)]
    groups = rng.sample(range(3), 3)
    columns = [group * 3 + col for group in groups for col in rng.sample(range(3), 3)]
    if seed % 2:
        return tuple(digits[board[col * 9 + row]] for row in rows for col in columns)
    return tuple(digits[board[row * 9 + col]] for row in rows for col in columns)


def contradictory_sudoku(board):
    for index, value in enumerate(board):
        if value:
            continue
        used = {board[peer] for peer in peers()[index]}
        for digit in sorted(set(range(1, 10)) - used):
            candidate = (*board[:index], digit, *board[index + 1:])
            if not sudoku_reference(candidate):
                return candidate
    raise AssertionError("Expected a locally consistent unsatisfiable board")


def graph_case(seed):
    rng = Random(seed)
    size = GRAPH_SIZE
    split = (size // 2) if seed % 3 == 0 else size
    edges = [
        [int(a != b and (a < split) == (b < split) and (rng.randrange(16) < 2 or b == a + 1))
         for b in range(size)] for a in range(size)
    ]
    page = initial_page(())
    struct.pack_into(f"<{size}i", page, GRAPH_DISTANCE, *([-1] * size))
    struct.pack_into(f"<{size * size}i", page, GRAPH_EDGES, *(edge for row in edges for edge in row))
    start = seed % size
    distance = [-1] * size
    distance[start] = 0
    queue = deque((start,))
    order = [start]
    while queue:
        vertex = queue.popleft()
        for neighbor, edge in enumerate(edges[vertex]):
            if edge and distance[neighbor] == -1:
                distance[neighbor] = distance[vertex] + 1
                queue.append(neighbor)
                order.append(neighbor)
    expected = bytearray(page)
    struct.pack_into(f"<{len(order)}i", expected, GRAPH_QUEUE, *order)
    struct.pack_into(f"<{size}i", expected, GRAPH_DISTANCE, *distance)
    return start, page, len(order), expected


def matrix_case(size, seed):
    rng = Random(seed)
    left, right = ([rng.randrange(-9, 10) for _ in range(size * size)] for _ in range(2))
    fmt = f"<{size * size}i"
    page = initial_page(())
    struct.pack_into(fmt, page, 0, *left)
    struct.pack_into(fmt, page, MATRIX_B, *right)
    result = [sum(left[i * size + k] * right[k * size + j] for k in range(size))
              for i in range(size) for j in range(size)]
    expected = bytearray(page)
    struct.pack_into(fmt, expected, MATRIX_C, *result)
    return size, page, size, expected
