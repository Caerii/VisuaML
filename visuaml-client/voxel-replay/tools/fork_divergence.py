"""Compare ordered observable boundaries, including reconvergent histories."""
from itertools import zip_longest


def first_divergence(reference, candidate):
    missing = object()
    for index, (left, right) in enumerate(zip_longest(reference, candidate, fillvalue=missing)):
        if left != right:
            return index
    return None
