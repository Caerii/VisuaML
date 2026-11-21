"""Shared canonical JSON and digest helpers for replay formats."""
from hashlib import sha256
import json


def digest(raw):
    return sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
