"""Assemble browser captures into readable, deliberately slowed GIFs."""
import argparse
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for stem, delays, name in (
        ("live", (1100, 950, 1600), "playing"),
        ("replay", (750, 1000, 1350, 1050, 1550), "edit-vs-omitted"),
    ):
        images = []
        for index in range(len(delays)):
            with Image.open(args.captures / f"{stem}-{index}.png") as source:
                canvas = Image.new("RGB", (1120, 890), "#12191c")
                canvas.paste(source.convert("RGB"), (0, 0))
            images.append(canvas.resize((900, 715), Image.Resampling.LANCZOS))
        images[0].save(args.out / f"{name}.gif", save_all=True, append_images=images[1:],
                       duration=delays, loop=0, optimize=True)


if __name__ == "__main__":
    main()
