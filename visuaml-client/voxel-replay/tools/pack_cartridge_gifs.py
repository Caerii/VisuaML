"""Package the real-browser playthrough captures."""
import argparse

from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for capture, name, delays in (
        ("rail", "rail-crate", [950, 750, 1100, 1400]),
        ("monitor", "monitor", [900, 500, 900, 500, 1100, 500, 1050, 500, 1600]),
    ):
        frames = []
        for index in range(len(delays)):
            with Image.open(args.source / f"{capture}-{index}.png") as image:
                frames.append(image.convert("RGB").resize((860, 645), Image.Resampling.LANCZOS))
        frames[0].save(args.output / f"{name}.gif", save_all=True, append_images=frames[1:],
                       duration=delays, loop=0, optimize=True)


if __name__ == "__main__":
    main()
