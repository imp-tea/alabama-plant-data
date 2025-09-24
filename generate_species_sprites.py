"""Batch sprite generation for Alabama plant dataset."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Iterable

from openai import OpenAI

BASE_PROMPT = (
    "Using the provided reference image and plant information, generate a 2D image "
    "of the plant in a low resolution pixel art style with a transparent background, "
    "to be used as a sprite in a video game. The generated image should be styled "
    "as a 32x32 sprite with a small palette size, and each pixel should be a clearly "
    "defined square. Rely on the image and biological data as inspiration, but "
    "exaggerate the plant's features and give it a game-y visual pop. Here is the "
    "plant species information:\n"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def iter_species_dirs(plants_root: Path) -> Iterable[Path]:
    for path in sorted(plants_root.iterdir()):
        if path.is_dir():
            yield path


def find_text_file(species_dir: Path) -> Path:
    txt_candidates = list(species_dir.glob("*.txt"))
    if not txt_candidates:
        raise FileNotFoundError(f"No .txt file found in {species_dir}")
    # Prefer text file matching the directory name if present.
    for candidate in txt_candidates:
        if candidate.stem.lower() == species_dir.name.lower():
            return candidate
    return txt_candidates[0]


def find_image_file(species_dir: Path) -> Path:
    image_candidates = [
        path for path in species_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_candidates:
        raise FileNotFoundError(f"No image file found in {species_dir}")
    if len(image_candidates) > 1:
        # Choose the first alphabetically to keep selection deterministic.
        image_candidates.sort()
    return image_candidates[0]


def main() -> None:
    plants_root = Path(__file__).resolve().parent / "plants"
    if not plants_root.is_dir():
        raise FileNotFoundError(f"Plants directory not found at {plants_root}")

    client = OpenAI()

    for species_dir in iter_species_dirs(plants_root):
        try:
            text_path = find_text_file(species_dir)
            species_info = text_path.read_text(encoding="utf-8").strip()
            image_path = find_image_file(species_dir)
        except Exception as exc:  # noqa: BLE001 - keep context for logging
            print(f"Skipping {species_dir.name}: {exc}")
            continue

        prompt = f"{BASE_PROMPT}{species_info}\n"
        sprite_path = species_dir / "sprite.png"

        print(f"Generating sprite for {species_dir.name} using {image_path.name}")

        with image_path.open("rb") as image_file:
            result = client.images.edit(
                model="gpt-image-1",
                image=image_file,
                prompt=prompt,
                #input_fidelity="high",
                background="transparent",
            )

        image_base64 = result.data[0].b64_json
        sprite_bytes = base64.b64decode(image_base64)
        sprite_path.write_bytes(sprite_bytes)

        print(f"Saved {sprite_path.relative_to(species_dir)}")


if __name__ == "__main__":
    main()
