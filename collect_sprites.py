from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
plants_dir = ROOT / "plants"
target_dir = ROOT / "plant_sprites"

target_dir.mkdir(exist_ok=True)

for species_dir in sorted(plants_dir.iterdir()):
    if not species_dir.is_dir():
        continue

    sprite_path = species_dir / "sprite.png"
    fixed_path = species_dir / "sprite_fixed.png"

    if sprite_path.exists():
        destination_path = target_dir / f"{species_dir.name}.png"
        shutil.copy2(sprite_path, destination_path)
        print(f"Copied {sprite_path} -> {destination_path}")
    else:
        print(f"Missing sprite.png in {species_dir}")

    if fixed_path.exists():
        fixed_path.unlink()
        print(f"Deleted {fixed_path}")
