"""Analyse a multi-rater human study (docs/HUMAN_STUDY_PROTOCOL.md).

    python experiments/human_study.py --manifest study_manifest.json --ratings ratings/*.json \
        --calibration results/nyu_test/<model>/silver_calibration.json --output results/nyu_test/<model>/human_study.json
"""

import argparse
import json
from pathlib import Path

from veo.human_study import analyze, load_ratings
from veo.reports import write_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration", type=Path, help="silver_calibration.json with Phase 3 models (optional)")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    raters = load_ratings(args.ratings, manifest["study_id"])
    calibration = json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else None
    result = analyze(manifest, raters, calibration, args.replicates)
    write_result(args.output, result, inputs={"manifest": args.manifest, **{f"ratings_{i}": p for i, p in enumerate(args.ratings)}})
    agreement = result["veo_vs_consensus"]
    print(f"Raters included: {sum(r['included'] for r in result['raters'].values())}/{len(result['raters'])}; "
          f"Krippendorff alpha {result['inter_rater']['krippendorff_alpha']:.3f}")
    print(f"VEO vs consensus on {agreement['n']} items: accuracy {agreement['accuracy']:.3f}, kappa {agreement['kappa']:.3f}")
    print(f"Pre-registered checks: {result['preregistered_checks']}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
