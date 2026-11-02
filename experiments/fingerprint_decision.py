"""Phase 6 gate: does v1's "failure fingerprint" claim hold across datasets, or is it cut?

Reads the comparison files for NYU test, DIODE and iBims-1 and the external label checks, then applies
the rule in docs/MODEL_COMPARISON_PROTOCOL.md.
"""

import argparse
import json
from pathlib import Path

from veo.model_comparison import fingerprint_verdict
from veo.reports import write_result


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    paths = {name: args.results / name / "model_comparison.json" for name in ("nyu_test", "diode", "ibims")}
    hypersim = load(args.results / "hypersim" / "hypersim_reflection.json")["external_label_check"]["categories"]
    ibims = load(args.results / "ibims" / "label_check.json")["categories"]
    label_auroc = {
        label: {"hypersim": hypersim[label]["label_model"]["auroc"], "ibims": ibims[label]["label_model"]["auroc"]}
        for label in ibims
    }
    report = fingerprint_verdict({name: load(path) for name, path in paths.items()}, label_auroc)
    output = args.results / "fingerprint_decision.json"
    write_result(output, {"contrasts": report, "label_auroc": label_auroc}, inputs=paths)
    for contrast, entry in report.items():
        print(f"{contrast}: {entry['verdict']}; passing labels: {[label for label, row in entry['labels'].items() if row['passes']] or 'none'}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
