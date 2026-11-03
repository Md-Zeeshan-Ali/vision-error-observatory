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
    "ZoeDepth NYU+KITTI*": "zoedepth-nyu-kitti", "Depth Anything V2 Small": "depth-anything-v2-small-hf", "Marigold\u2020": "marigold-depth-v1-1",
    "DPT-Hybrid": "dpt-hybrid-midas", "DPT-Large": "dpt-large",
}


MIN_POSITIVES = 10  # below this an AUROC isn't reported (docs/MODEL_COMPARISON_PROTOCOL.md)


def _results(*parts: str) -> dict:
    return json.loads(RESULTS.joinpath(*parts).read_text(encoding="utf-8"))


def _readme_rows() -> list[str]:
    nyu = _results("nyu_test", "depth-anything-v2-small-hf", "silver_calibration.json")["test"]["evaluation"]["categories"]
    hypersim = _results("hypersim", "hypersim_reflection.json")
    ibims = _results("ibims", "label_check.json")

    def ibims_cell(label: str) -> str:
        entry = ibims["categories"].get(label)
        return f"{entry['label_model']['auroc']:.2f}" if entry and entry["positives"] >= MIN_POSITIVES else "\u2014"

    rows = [f"| `{label}` | {nyu[label]['label_model']['auroc']:.2f} | {hypersim['external_label_check']['categories'][label]['label_model']['auroc']:.2f} | {ibims_cell(label)} |"
            for label in ("boundary_mixing", "thin_object", "reflection", "textureless")]
    rows.append(f"which alone score {ibims['reflection_sensor_holes_only']['auroc']:.2f} on iBims-1")
    rows.append(f"only {ibims['categories']['textureless']['positives']} regions fall on its wall masks")
    decision = _results("fingerprint_decision.json")["contrasts"]
    assert decision["backbone_hybrid_vs_large"]["verdict"] == "cut"
    family = decision["family_dav2_small_vs_dpt_hybrid"]["labels"]
    assert [label for label, entry in family.items() if entry["passes"]] == ["textureless"] and not family["textureless"]["failure_type_difference"]
    shift = {name: f"{family['textureless']['datasets'][name]['mean']:.2f}".replace("-", "\u2212") for name in ("nyu_test", "diode", "ibims")}
    rows.append(f"({shift['nyu_test']} on NYU, {shift['diode']} on DIODE, {shift['ibims']} on iBims-1)")
    for name, slug in MODELS.items():
        summary, diode, ibims_summary = (_results(name, slug, "summary.json") for name in ("nyu_test", "diode", "ibims"))
        rows.append(f"| {name} | {summary['mean_metrics']['abs_rel']:.3f} | {summary['mean_metrics']['delta1']:.3f} | "
                    f"{100 * summary['mean_failure_pixel_fraction']:.1f}% | {diode['mean_metrics']['abs_rel']:.3f} | {ibims_summary['mean_metrics']['abs_rel']:.3f} |")
    tests = _results("nyu_test", "depth-anything-v2-small-hf", "interventions.json")["tests"]
    for name, title in (("resolution_boundary", "Higher resolution, `boundary_mixing`"), ("resolution_thin", "Higher resolution, `thin_object`"),
                        ("exposure", "Brighten, `poor_lighting`"), ("texture", "Add texture, `textureless`")):
        d = tests[name]["difference"]
        signed = lambda value, plus=True: f"{value:+.3f}".replace("-", "\u2212") if plus else f"{value:.3f}".replace("-", "\u2212")  # noqa: E731
        p_value = f"{d['p_holm']:.3f}" if d["p_holm"] < 0.01 else f"{d['p_holm']:.2f}"
        rows.append(f"| {title} | {signed(d['mean'])} [{signed(d['lower'])}, {signed(d['upper'])}] | {p_value} |")
    finetune = _results("finetune", "guided_finetune.json")
    means = finetune["condition_means"]
    for name, outcome, veo, control in (("boundary", "abs_rel_boundary_mixing", "veo_boundary", "random_boundary"),
                                               ("textureless", "abs_rel_textureless", "veo_textureless", "random_textureless")):
        row = finetune["primary"][name]
        assert not row["supported"]
        rows.append(f"| {means[veo][outcome]:.3f} | {means[control][outcome]:.3f} | {row['mean']:+.3f} [{row['lower']:+.3f}, {row['upper']:+.3f}] | No |")
    rows.append(f"improved AbsRel from {means['pretrained']['abs_rel']:.3f} to {means['uniform']['abs_rel']:.3f}")
    check = _results("finetune", "training_pixel_check.json")["models"]
    rows.append(f"from {check['uniform_seed0']['boundary_mixing']['abs_rel']:.2f} to {check['veo_boundary_seed0']['boundary_mixing']['abs_rel']:.2f}")
    sweep = _results("finetune", "weight_sweep.json")
    minus = lambda value: f"{value:+.4f}".replace("-", "−")  # noqa: E731
    for family, key in (("sweep", "boundary_share05"), ("sweep", "boundary_share10"), ("sweep", "textureless_share05"), ("sweep", "textureless_share10"),
                        ("hard_example", "boundary_vs_failure_weighted"), ("hard_example", "textureless_vs_failure_weighted")):
        row = sweep[family][key]
        assert not row["supported"]
        p_value = f"{row['p_holm']:.4f}" if row["p_holm"] < 0.01 else f"{row['p_holm']:.2f}"
        rows.append(f"| {minus(row['mean'])} [{minus(row['lower'])}, {minus(row['upper'])}] | {p_value} | No |")
    rows.append(f"({minus(sweep['descriptive']['failure_weighted_vs_uniform_overall']['mean'])} overall AbsRel)")
    review = _results("review_checks.json")
    cues, small = review["cue_baselines"]["nyu_test"], review["pixels"]["depth-anything-v2-small-hf"]
    for label, cue, cue_name in (("boundary_mixing", "depth_edge_fraction", "depth-edge share"), ("thin_object", "thin_image_fraction", "thin-structure share"),
                                 ("textureless", "log_area_alone", "region size"), ("reflection", "sensor_hole_fraction", "sensor holes")):
        ratio = small["base_rates"][label]["rate_ratio"]
        rows.append(f"| `{label}` | {cues[label]['label_model']['auroc']:.2f} | {cues[label][cue]['auroc']:.2f} ({cue_name}) | "
                    f"{review['top_label']['nyu_test'][label]['precision']:.2f} | {ratio:.2f}× |".replace(f"{ratio:.2f}", f"{ratio:.2f}" if ratio < 1 else f"{ratio:.1f}")
                    + f" {100 * small['edge_tolerance'][label]['kept_at_2px']:.0f}% |")
    partial = review["specular_partial"]
    rows.append(f"partial ρ = {partial['partial_spearman']:.2f}, 95% CI [{partial['lower']:.2f}, {partial['upper']:.2f}]".replace("-", "−"))
    dose = hypersim["dose_response_spearman"]
    rows.append(f"\u03c1 = {dose['estimate']:.2f}".replace("-", "\u2212"))
    rows.append(f"[{dose['lower']:.2f}, {dose['upper']:.2f}]".replace("-", "\u2212"))
    return rows


@pytest.mark.parametrize("row", _readme_rows())
def test_readme_states_current_results(row):
    assert row in README


def test_study_size_in_readme_matches_the_protocol():
    assert "A blinded study of 300 regions" in README
