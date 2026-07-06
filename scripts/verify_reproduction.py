from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


GRAPHIC_SUFFIXES = {".svg", ".eps"}
IGNORED_NAMES = {".DS_Store"}


def _normalize_svg(text: str) -> str:
    text = re.sub(r"<dc:date>.*?</dc:date>", "<dc:date>DATE</dc:date>", text)
    for prefix in ("p", "m"):
        text = re.sub(rf"url\(#{prefix}[0-9a-f]+\)", f"url(#{prefix}HASH)", text)
        text = re.sub(rf'id="{prefix}[0-9a-f]+"', f'id="{prefix}HASH"', text)
        text = re.sub(rf'xlink:href="#{prefix}[0-9a-f]+"', f'xlink:href="#{prefix}HASH"', text)
    text = re.sub(r'id="C([0-9A-Za-z]+)_\d+_[0-9a-f]+"', r'id="C\1_HASH"', text)
    text = re.sub(r'xlink:href="#C([0-9A-Za-z]+)_\d+_[0-9a-f]+"', r'xlink:href="#C\1_HASH"', text)
    return text


def _normalize_eps(text: str) -> str:
    return re.sub(r"%%CreationDate:.*", "%%CreationDate: DATE", text)


def _read_for_compare(path: Path) -> bytes | str:
    if path.suffix.lower() == ".svg":
        return _normalize_svg(path.read_text(errors="replace"))
    if path.suffix.lower() == ".eps":
        return _normalize_eps(path.read_text(errors="replace"))
    return path.read_bytes()


def _iter_files(root: Path) -> set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and not any(part in IGNORED_NAMES for part in path.relative_to(root).parts)
    }


def compare(reference: Path, candidate: Path) -> list[str]:
    reference_files = _iter_files(reference)
    candidate_files = _iter_files(candidate)
    messages: list[str] = []

    for missing in sorted(reference_files - candidate_files):
        messages.append(f"missing from candidate: {missing}")
    for extra in sorted(candidate_files - reference_files):
        messages.append(f"extra in candidate: {extra}")

    for relative_path in sorted(reference_files & candidate_files):
        ref_path = reference / relative_path
        candidate_path = candidate / relative_path
        if _read_for_compare(ref_path) != _read_for_compare(candidate_path):
            comparison = "normalized graphics differ" if ref_path.suffix.lower() in GRAPHIC_SUFFIXES else "bytes differ"
            messages.append(f"{comparison}: {relative_path}")

    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a fresh model-scatter run with the embedded reference output.")
    parser.add_argument("--reference", type=Path, default=Path("done-final-fix"))
    parser.add_argument("--candidate", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    messages = compare(args.reference, args.candidate)
    if messages:
        for message in messages:
            print(message)
        return 1

    print(f"{args.candidate} reproduces {args.reference}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
