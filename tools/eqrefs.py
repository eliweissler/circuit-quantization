"""Check that every notes label cited in the code exists in the .tex source.

Usage: python tools/eqrefs.py path/to/full_matrix_quantization.tex [--numbers path/to/.aux]
With --numbers, also print the printed equation number of each label.
"""

import argparse
import pathlib
import re
import sys

LABEL = re.compile(r"\b((?:eq|tab|fig|apx|sec):[A-Za-z0-9_]+)")


def cited_labels(root):
    cited = {}
    for path in sorted(root.rglob("*.py")):
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            for label in LABEL.findall(line):
                cited.setdefault(label, []).append(f"{path}:{line_no}")
    return cited


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tex", type=pathlib.Path)
    parser.add_argument("--numbers", type=pathlib.Path, help="the .aux file")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    defined = set(re.findall(r"\\label\{([^}]*)\}", args.tex.read_text()))
    numbers = {}
    if args.numbers:
        for label, number in re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", args.numbers.read_text()):
            numbers[label] = number
    cited = {}
    for folder in ("src", "tests"):
        cited.update(cited_labels(root / folder))
    missing = sorted(label for label in cited if label not in defined)
    for label in sorted(cited):
        mark = "MISSING" if label in missing else numbers.get(label, "")
        print(f"{label:40s} {mark}")
    if missing:
        print(f"\n{len(missing)} cited label(s) not in {args.tex}", file=sys.stderr)
        for label in missing:
            print(f"  {label}: {', '.join(cited[label])}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
