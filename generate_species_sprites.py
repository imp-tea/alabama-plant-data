"""Batch sprite generation for Alabama plant dataset."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Iterable

from openai import OpenAI

BASE_PROMPT = ("""Using the provided reference image and plant information, generate a prompt for an image generation model to create a 2D image of the plant in a low resolution pixel art style with a flat black background, to be used as a sprite in a RPG video game with 3/4 perspective. The generated image should be styled as a pixelized sprite with a small palette size, and each pixel should be a clearly defined square. DO NOT instruct the model to convert or recreate the reference image. We are generating an entirely new image, not editing an existing one. Rely on the image and biological data as inspiration, but the final image should exaggerate the plant's features (e.g., make the distinguishing characteristics larger for more visual clarity; bigger leaves on a tree, bigger berries, bigger flowers, etc.) and give it a game-y visual pop. The plants should be featured in the height of their growth cycle, when their distinguishing features are most visible - flowers in full bloom, fruits or seeds visible, healthy green foliage, etc. For smaller plants - shrubs, flowers, vines, etc. - target a 1:1 image aspect ratio and a sprite size of 32x32. For a larger plants like trees or bamboos, target a 1:2 aspect ratio and a sprite size of 32x64. For vining plants, the plant should be depicted growing on a trellis or pole. ***Respond with the prompt only; do not preface your response or ask any follow-up questions.*** Here is the plant species information:\n""")

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
        sprite_path = species_dir / "sprite.png"
        if sprite_path.exists():
            print(f"Skipping {species_dir.name}: sprite.png already exists")
            continue

        try:
            text_path = find_text_file(species_dir)
            species_info = text_path.read_text(encoding="utf-8").strip()
            image_path = find_image_file(species_dir)
        except Exception as exc:  # noqa: BLE001 - keep context for logging
            print(f"Skipping {species_dir.name}: {exc}")
            continue

        prompt = f"{BASE_PROMPT}{species_info}\n"

        print(f"Generating prompt for {species_dir.name} using {image_path.name}")

        with image_path.open("rb") as image_file:
            response = client.responses.create(
                model="gpt-5",
                input=prompt,
            )
            IMAGE_PROMPT = response.output_text
            print(f"Image prompt:\n{IMAGE_PROMPT}\n")
            print(f"Generating image for {species_dir.name} using {image_path.name}")
            result = client.images.edit(
                model="gpt-image-1",
                image=image_file,
                prompt=IMAGE_PROMPT,
                #input_fidelity="high",
                #background="transparent",
            )

        image_base64 = result.data[0].b64_json
        sprite_bytes = base64.b64decode(image_base64)
        sprite_path.write_bytes(sprite_bytes)

        print(f"Saved {sprite_path.relative_to(species_dir)}")


if __name__ == "__main__":
    main()
