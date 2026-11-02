"""The v1 record's numbers must match the committed result files."""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
V1 = REPO / "results" / "v1"
V1_RECORD = (REPO / "docs" / "V1_RESULTS.md").read_text(encoding="utf-8")


def _json(name: str) -> dict:
    return json.loads((V1 / name).read_text(encoding="utf-8"))


def _expected() -> list[str]:
    human = _json("human_agreement.json")
    rater1 = human["veo_vs_rater1"]
    selective = rater1["selective"]
    per_class = rater1["per_class"]
    da = _json("nyu_final_summary.json")
    dpt = _json("dpt_hybrid_summary.json")

    def share(summary: dict, label: str) -> str:
        return f"{100 * summary['cause_counts'][label] / summary['num_regions']:.1f}%"

    return [
        f"$\\kappa$ = {human['inter_rater_kappa']:.3f}",
        f"$\\kappa$ = {rater1['kappa']:.3f}",
        f"$\\kappa$ = {human['veo_vs_rater2_kappa']:.3f}",
        f"accuracy {100 * rater1['accuracy']:.0f}% ({round(rater1['accuracy'] * rater1['n'])}/{rater1['n']})",
        f"macro-F1 = {rater1['macro_f1']:.3f}",
        f"{rater1['abstained']} / {rater1['n']}",
        f"{100 * selective['accuracy']:.1f}%",
        f"$\\kappa$ = {selective['kappa']:.3f}",
        f"{100 * selective['majority_baseline_accuracy']:.1f}%",
        *(f"`{label}` {per_class[label]['correct']}/{per_class[label]['predicted']}" for label in ("textureless", "thin_object", "boundary_mixing", "reflection")),
        f"{da['num_regions']:,} regions",
        f"{dpt['num_regions']:,} regions",
        f"AbsRel = {da['mean_metrics']['abs_rel']:.4f}",
        f"$\\delta_1$ = {da['mean_metrics']['delta1']:.4f}",
        f"{share(da, 'textureless')} `textureless`",
        f"{share(da, 'boundary_mixing')} `boundary_mixing`",
        f"{share(da, 'ambiguous')} `ambiguous`",
        f"{share(dpt, 'reflection')} `reflection` vs {share(da, 'reflection')}",
    ]


@pytest.mark.parametrize("text", _expected())
def test_v1_record_states_result_file_numbers(text):
    assert text in V1_RECORD
