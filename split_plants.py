#!/usr/bin/env python3
"""
Clean and split plants.txt into one file per plant.

- Strips citations like ":contentReference[oaicite:0]{index=0}" (including repeats).
- Splits on "Plant Name:" boundaries.
- Uses the scientific name from parentheses on the Plant Name line
  to create the output filename in slug case, e.g., "Pinus-palustris.txt".
- Writes files to the current working directory.

Usage:
    python split_plants.py [path/to/plants.txt]

If no path is given, defaults to "./plants.txt".
"""

import re
import sys
from pathlib import Path

# Pattern for the inline citations, e.g. :contentReference[oaicite:0]{index=0}
CITATION_RE = re.compile(r':contentReference\[oaicite:\d+\]\{index=\d+\}')

# Start of each entry
ENTRY_SPLIT_RE = re.compile(r'(?=^Plant Name:\s*)', flags=re.M)

# Scientific name on the Plant Name line, inside parentheses
SCIENTIFIC_NAME_RE = re.compile(r'^Plant Name:\s*.*\(([^)]+)\)', flags=re.M)

def slug_from_scientific(scientific: str) -> str:
    """
    Convert a scientific name to a slug, keeping capitalization and replacing spaces with hyphens.
    Non-alphanumeric characters (except hyphens) are stripped/replaced with a single hyphen.
    """
    # Replace whitespace with single hyphen
    slug = re.sub(r'\s+', '-', scientific.strip())
    # Remove/normalize anything that isn't A–Z, a–z, 0–9, or hyphen
    slug = re.sub(r'[^A-Za-z0-9\-]+', '-', slug)
    # Collapse multiple hyphens
    slug = re.sub(r'-{2,}', '-', slug).strip('-')
    return slug or "unknown"

def clean_citations(text: str) -> str:
    """Remove all :contentReference[...] citations from the text and tidy excess spaces around them."""
    cleaned = CITATION_RE.sub('', text)
    # Remove accidental leftover double spaces before punctuation/line breaks caused by deletion
    cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)
    cleaned = re.sub(r' +([,;:.])', r'\1', cleaned)
    # Collapse multiple blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

def main():
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("plants.txt")
    if not in_path.exists():
        sys.stderr.write(f"Input file not found: {in_path}\n")
        sys.exit(1)

    raw = in_path.read_text(encoding="utf-8")

    # Split into entries while keeping the "Plant Name:" header in each chunk
    chunks = [c.strip() for c in ENTRY_SPLIT_RE.split(raw) if c.strip()]
    if not chunks:
        sys.stderr.write("No plant entries found. Ensure entries start with 'Plant Name:'.\n")
        sys.exit(2)

    out_dir = Path(".")  # current directory; change if you want a subfolder
    written = 0
    seen_filenames = set()

    for chunk in chunks:
        # Extract scientific name
        m = SCIENTIFIC_NAME_RE.search(chunk)
        if not m:
            # Skip chunks that don't have a recognizable scientific name
            continue
        scientific = m.group(1)
        slug = slug_from_scientific(scientific)
        filename = f"{slug}.txt"

        # Ensure uniqueness if a duplicate scientific name somehow appears
        base_slug = slug
        n = 2
        while filename in seen_filenames or (out_dir / filename).exists():
            slug = f"{base_slug}-{n}"
            filename = f"{slug}.txt"
            n += 1
        seen_filenames.add(filename)

        # Clean out citations
        cleaned = clean_citations(chunk)

        # Write file
        (out_dir / filename).write_text(cleaned + "\n", encoding="utf-8")
        written += 1
        print(f"Wrote {filename}")

    if written == 0:
        sys.stderr.write("No files were written. Check the input format.\n")
        sys.exit(3)

if __name__ == "__main__":
    main()
