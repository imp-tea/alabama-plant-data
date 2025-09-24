from __future__ import annotations

import base64
import sys
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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".tiff": "image/tiff",
}


def iter_species_directories(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_dir():
            yield path


def find_matching_file(directory: Path, extensions: set[str]) -> Path | None:
    candidates = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    if not candidates:
        return None

    for candidate in candidates:
        if candidate.stem.lower() == directory.name.lower():
            return candidate
    return candidates[0]


def load_species_text(species_dir: Path) -> str | None:
    preferred_name = species_dir.name
    preferred_txt = species_dir / f"{preferred_name}.txt"
    if preferred_txt.exists():
        return preferred_txt.read_text(encoding="utf-8").strip()

    camel_case_name = preferred_name.replace("-", " ").title().replace(" ", "-")
    camel_case_txt = species_dir / f"{camel_case_name}.txt"
    if camel_case_txt.exists():
        return camel_case_txt.read_text(encoding="utf-8").strip()

    txt_file = find_matching_file(species_dir, {".txt"})
    if txt_file:
        return txt_file.read_text(encoding="utf-8").strip()

    return None


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def generate_sprite_for_species(client: OpenAI, species_dir: Path, *, overwrite: bool = True) -> None:
    text = load_species_text(species_dir)
    if not text:
        print(f"[skip] No species text found in {species_dir}")
        return

    image_path = find_matching_file(species_dir, IMAGE_EXTENSIONS)
    if not image_path:
        print(f"[skip] No reference image found in {species_dir}")
        return

    sprite_path = species_dir / "sprite.png"
    if sprite_path.exists() and not overwrite:
        print(f"[skip] Sprite already exists for {species_dir.name}")
        return

    prompt = f"{BASE_PROMPT}{text}\n"
    image_data_base64 = encode_image(image_path)
    mime_type = MIME_TYPES.get(image_path.suffix.lower(), "image/jpeg")

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": f"data:{mime_type};base64,{image_data_base64}",
                    },
                ],
            }
        ],
        tools=[{"type": "image_generation"}],
    )

    image_generation_calls = [
        output
        for output in response.output
        if getattr(output, "type", None) == "image_generation_call"
    ]

    if not image_generation_calls:
        print(f"[error] Image generation failed for {species_dir.name}: {response.output}")
        return

    image_base64 = image_generation_calls[0].result

    try:
        sprite_path.write_bytes(base64.b64decode(image_base64))
        print(f"[done] Saved sprite for {species_dir.name} -> {sprite_path}")
    except Exception as exc:
        print(f"[error] Failed to write sprite for {species_dir.name}: {exc}")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "plants"
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Plants directory not found: {root}")

    client = OpenAI()

    for species_dir in iter_species_directories(root):
        try:
            generate_sprite_for_species(client, species_dir)
        except Exception as exc:
            print(f"[error] Unexpected failure for {species_dir.name}: {exc}")


if __name__ == "__main__":
    main()
