"""Feature sensitivity ablation: how many region labels change when one feature is set to a neutral value."""

import argparse
import json
from pathlib import Path

from veo.label_model import load_label_model
from veo.reports import write_result
from veo.validation import NEUTRAL_VALUES, ablation_change, model_neutral_values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=Path, default=Path("outputs/nyu_final/test/model_results.json"))
    parser.add_argument("--output", type=Path, default=Path("results/nyu_test/signal_ablation.json"))
    args = parser.parse_args()

    payload = json.loads(args.results_path.read_text(encoding="utf-8"))
    records = payload["records"]
    settings = payload.get("config", {}).get("experiment", {})
    label_model = load_label_model(settings.get("label_model"))
    neutral_values = model_neutral_values(label_model) if label_model else NEUTRAL_VALUES
    rows = []
    for signal, neutral in neutral_values.items():
        counts = ablation_change(records, signal, include_reflection=True, label_model=label_model, settings=settings)
        total = counts["total_regions"]
        rows.append({
            "signal": signal,
            "neutral_value": neutral,
            **counts,
            "changed_fraction": counts["changed_regions"] / total if total else 0.0,
        })
        print(f"{signal:<22} -> {neutral:<8.3f} changed {counts['changed_regions']:>5}/{total} ({rows[-1]['changed_fraction']:.2%})")
    write_result(args.output, {"rows": rows}, inputs={"veo_results": args.results_path})
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
