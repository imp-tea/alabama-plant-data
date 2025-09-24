#!/usr/bin/env python3
"""Batch apply fix_pixel_art to every plants/<name>/sprite.png."""
from pathlib import Path
from typing import Iterable

from PIL import Image

from fix_pixel_art import fix_pixel_art


DEFAULT_PLANTS_DIR = Path(__file__).resolve().parent / "plants"


def iter_sprite_paths(plants_dir: Path) -> Iterable[Path]:
    for child in sorted(plants_dir.iterdir()):
        if not child.is_dir():
            continue
        sprite_path = child / "sprite.png"
        if sprite_path.is_file():
            yield sprite_path
        else:
            print(f"Skipping {child}: no sprite.png found")


def process_directory(plants_dir: Path) -> None:
    if not plants_dir.is_dir():
        raise FileNotFoundError(f"Plants directory not found: {plants_dir}")

    for sprite_path in iter_sprite_paths(plants_dir):
        output_path = sprite_path.with_name("sprite_fixed.png")
        print(f"Processing {sprite_path} -> {output_path}")
        with Image.open(sprite_path) as img:
            _, _, fixed = fix_pixel_art(img)
        fixed.save(output_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply fix_pixel_art.py to every sprite.png under plants/ subdirectories."
    )
    parser.add_argument(
        "--plants-dir",
        type=Path,
        default=DEFAULT_PLANTS_DIR,
        help="Folder containing subdirectories with sprite.png files (default: ./plants)",
    )
    args = parser.parse_args()

    process_directory(args.plants_dir)


if __name__ == "__main__":
    main()
