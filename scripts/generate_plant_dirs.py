#!/usr/bin/env python3
import csv, json, re, unicodedata, sys
from pathlib import Path

def slugify(text):
    if text is None:
        return ''
    text = str(text)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = text.strip('-')
    return text

def unique_dir(base_dir: Path, base_slug: str) -> Path:
    d = base_dir / base_slug
    if not d.exists():
        return d
    i = 2
    while True:
        candidate = base_dir / f"{base_slug}-{i}"
        if not candidate.exists():
            return candidate
        i += 1

def main():
    cwd = Path.cwd()
    csv_path = cwd / "central_alabama_plants.csv"
    if not csv_path.exists():
        print(f"CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)
    out_root = cwd / "plants"
    out_root.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", encoding="utf-8-sig", newline='') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            sci = row.get("ScientificName") or ""
            com = row.get("CommonName") or ""
            sci_slug = slugify(sci)
            com_slug = slugify(com)
            dir_slug = sci_slug or com_slug or f"plant-{count+1}"
            plant_dir = unique_dir(out_root, dir_slug)
            plant_dir.mkdir(parents=True, exist_ok=True)

            plant_data = {
                "CommonName": row.get("CommonName", ""),
                "ScientificName": row.get("ScientificName", ""),
                "GrowthForm": row.get("GrowthForm", ""),
                "NativeStatus": row.get("NativeStatus", ""),
                "GeneralDescription": row.get("GeneralDescription", ""),
                "NotableCharacteristics": row.get("NotableCharacteristics", ""),
                "PreferredConditions": row.get("PreferredConditions", ""),
                "ReproductionTiming": row.get("ReproductionTiming", ""),
                "Lifespan": row.get("Lifespan", ""),
                "KeyRelationships": row.get("KeyRelationships", "")
            }
            plant_data_path = plant_dir / "plant_data"
            with plant_data_path.open("w", encoding="utf-8") as pf:
                json.dump(plant_data, pf, indent=2, ensure_ascii=False)
                pf.write("\n")

            game_data_path = plant_dir / "game_data"
            if not game_data_path.exists():
                with game_data_path.open("w", encoding="utf-8") as gf:
                    gf.write("{}\n")

            count += 1

    print(f"Created {count} plant subdirectories under {out_root}")

if __name__ == "__main__":
    main()