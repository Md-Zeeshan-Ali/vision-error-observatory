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


README = (REPO / "README.md").read_text(encoding="utf-8")
RESULTS = REPO / "results"
MODELS = {
    "Depth Anything V2 Large": "depth-anything-v2-large-hf", "Depth Pro": "depthpro-hf", "Depth Anything V2 Base": "depth-anything-v2-base-hf",
    "ZoeDepth NYU+KITTI*": "zoedepth-nyu-kitti", "Depth Anything V2 Small": "depth-anything-v2-small-hf", "DPT-Hybrid": "dpt-hybrid-midas", "DPT-Large": "dpt-large",
}


def _results(*parts: str) -> dict:
    return json.loads(RESULTS.joinpath(*parts).read_text(encoding="utf-8"))


def _readme_rows() -> list[str]:
    nyu = _results("nyu_test", "depth-anything-v2-small-hf", "silver_calibration.json")["test"]["evaluation"]["categories"]
    hypersim = _results("hypersim", "hypersim_reflection.json")
    rows = [f"| `{label}` | {nyu[label]['label_model']['auroc']:.2f} | {hypersim['external_label_check']['categories'][label]['label_model']['auroc']:.2f} |"
            for label in ("boundary_mixing", "thin_object", "reflection", "textureless")]
    for name, slug in MODELS.items():
        summary, diode = _results("nyu_test", slug, "summary.json"), _results("diode", slug, "summary.json")
        rows.append(f"| {name} | {summary['mean_metrics']['abs_rel']:.3f} | {summary['mean_metrics']['delta1']:.3f} | "
                    f"{100 * summary['mean_failure_pixel_fraction']:.1f}% | {diode['mean_metrics']['abs_rel']:.3f} |")
    tests = _results("nyu_test", "depth-anything-v2-small-hf", "interventions.json")["tests"]
    for name, title in (("resolution_boundary", "Higher resolution, `boundary_mixing`"), ("resolution_thin", "Higher resolution, `thin_object`"),
                        ("exposure", "Brighten, `poor_lighting`"), ("texture", "Add texture, `textureless`")):
        d = tests[name]["difference"]
        signed = lambda value, plus=True: f"{value:+.3f}".replace("-", "\u2212") if plus else f"{value:.3f}".replace("-", "\u2212")  # noqa: E731
        p_value = f"{d['p_holm']:.3f}" if d["p_holm"] < 0.01 else f"{d['p_holm']:.2f}"
        rows.append(f"| {title} | {signed(d['mean'])} [{signed(d['lower'])}, {signed(d['upper'])}] | {p_value} |")
    dose = hypersim["dose_response_spearman"]
    rows.append(f"\u03c1 = {dose['estimate']:.2f}".replace("-", "\u2212"))
    rows.append(f"[{dose['lower']:.2f}, {dose['upper']:.2f}]".replace("-", "\u2212"))
    return rows


@pytest.mark.parametrize("row", _readme_rows())
def test_readme_states_current_results(row):
    assert row in README


def test_study_size_in_readme_matches_the_protocol():
    assert "A blinded study of 300 regions" in README
