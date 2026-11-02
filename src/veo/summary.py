"""Render a run's result files into SUMMARY.md, so reported numbers are never typed by hand."""

import json
from pathlib import Path


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _human(result: dict) -> list[str]:
    report = result["veo_vs_rater1"]
    lines = [
        "## Human agreement",
        "",
        f"- All {report['n']} regions vs rater 1: accuracy {_pct(report['accuracy'])}, κ = {report['kappa']:.3f}, macro-F1 = {report['macro_f1']:.3f}",
    ]
    rater2 = result.get("veo_vs_rater2")
    rater2_kappa = rater2["kappa"] if rater2 else result.get("veo_vs_rater2_kappa")
    if rater2_kappa is not None:
        lines.append(f"- VEO vs rater 2: κ = {rater2_kappa:.3f}")
    if result.get("inter_rater_kappa") is not None:
        lines.append(f"- Inter-rater: κ = {result['inter_rater_kappa']:.3f}")
    selective = report["selective"]
    if selective:
        lines.append(
            f"- Committed {selective['n']}/{report['n']} (abstained {report['abstained']}): accuracy {_pct(selective['accuracy'])}, "
            f"κ = {selective['kappa']:.3f}; majority-class baseline {_pct(selective['majority_baseline_accuracy'])}"
        )
    lines += ["", "| Label | VEO correct / predicted | Rater support |", "|---|---|---|"]
    for label, row in report["per_class"].items():
        lines.append(f"| `{label}` | {row['correct']}/{row['predicted']} | {row['support']} |")
    return lines + [""]


def _model(name: str, summary: dict) -> list[str]:
    total = summary["num_regions"]
    metrics = summary.get("mean_metrics", {})
    lines = [f"## Model: {name}", "", f"- Samples: {summary['num_samples']}, regions: {total:,}"]
    if metrics:
        lines.append(f"- AbsRel {metrics['abs_rel']:.4f}, RMSE {metrics['rmse']:.4f} m, δ1 {metrics['delta1']:.4f}")
    if "mean_failure_pixel_fraction" in summary:
        lines.append(
            f"- Failing pixels: {_pct(summary['mean_failure_pixel_fraction'])} per image on average; "
            f"{_pct(summary['region_pixel_fraction'])} of valid pixels lie in regions; "
            f"{summary['multi_label_regions']:,} regions have more than one label"
        )
    pixel_share = summary.get("top_cause_pixel_fraction", {})
    label_share = summary.get("label_pixel_fraction", {})
    label_counts = summary.get("label_counts", {})
    lines += ["", "| Label | Regions (top label) | Share | Valid pixels (top label) | Regions with label | Valid pixels with label |", "|---|---|---|---|---|---|"]
    labels = set(summary["cause_counts"]) | set(label_counts)
    for label in sorted(labels, key=lambda name: (-summary["cause_counts"].get(name, 0), -label_counts.get(name, 0), name)):
        count = summary["cause_counts"].get(label, 0)
        share = _pct(count / total) if total else "-"
        lines.append(
            f"| `{label}` | {count:,} | {share} | {_pct(pixel_share[label]) if label in pixel_share else '-'} | "
            f"{label_counts.get(label, '-')} | {_pct(label_share[label]) if label in label_share else '-'} |"
        )
    return lines + [""]


def _interventions(result: dict) -> list[str]:
    lines = [
        "## Model-in-the-loop interventions",
        "",
        "Each labelled region is paired with a matched control region; both get the same edit and the model is re-run. "
        "Difference = (change in labelled-region AbsRel) - (change in control AbsRel); negative supports VEO's label.",
        "",
        "| Test | Edit | Pairs | Labelled Δ | Control Δ | Difference [95% CI] | p (Holm) | Supports label |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, test in result["tests"].items():
        stats = test["difference"]
        if stats.get("n", 0) < 2:
            lines.append(f"| {name} | {test['edit']} | {test['pairs']} | - | - | too few pairs | - | - |")
            continue
        lines.append(
            f"| {name} | {test['edit']} | {test['pairs']} | {test['mean_target_change']:+.4f} | {test['mean_control_change']:+.4f} | "
            f"{stats['mean']:+.4f} [{stats['lower']:+.4f}, {stats['upper']:+.4f}] | {stats.get('p_holm', stats['p_value']):.4f} | "
            f"{'yes' if test.get('supports_hypothesis') else 'no'} |"
        )
    return lines + [""]


def _interval(entry: dict | None, percent: bool = True) -> str:
    if not entry:
        return "-"
    fmt = _pct if percent else (lambda value: f"{value:.3f}")
    return f"{fmt(entry['estimate'])} [{fmt(entry['lower'])}, {fmt(entry['upper'])}]"


def _human_study(result: dict) -> list[str]:
    agreement = result["veo_vs_consensus"]
    bootstrap = agreement["bootstrap"]
    included = sum(rater["included"] for rater in result["raters"].values())
    inter = result["inter_rater"]
    lines = [
        "## Human study (Dawid-Skene consensus)",
        "",
        f"- {result['items_rated']} items; {included} of {len(result['raters'])} raters passed the practice round",
        f"- Inter-rater: Krippendorff α = {inter['krippendorff_alpha']:.3f}"
        + (f", Fleiss κ = {inter['fleiss_kappa']:.3f} on {inter['fleiss_items']} items" if inter["fleiss_kappa"] is not None else ""),
        f"- VEO vs consensus: accuracy {_interval(bootstrap['accuracy'])}, κ {_interval(bootstrap['kappa'], percent=False)}, "
        f"weighted to the test set {_interval(bootstrap['weighted_accuracy'])}",
    ]
    if "selective_accuracy" in bootstrap:
        lines.append(
            f"- Where VEO commits ({_pct(agreement['coverage'])} of items): accuracy {_interval(bootstrap['selective_accuracy'])} "
            f"vs majority baseline {_interval(bootstrap['selective_majority_baseline'])}"
        )
    if "v2_rules_vs_consensus" in result:
        lines.append(f"- v2 rules vs consensus: accuracy {_interval(result['v2_rules_vs_consensus']['bootstrap']['accuracy'])}")
    if "calibrated_vs_consensus" in result:
        lines.append(f"- Calibrated model vs consensus: accuracy {_interval(result['calibrated_vs_consensus']['bootstrap']['accuracy'])}")
    sensitivity = result.get("sensitivity_all_raters")
    if sensitivity:
        lines.append(f"- Sensitivity, all raters kept: Krippendorff α = {sensitivity['inter_rater']['krippendorff_alpha']:.3f}, "
                     f"VEO accuracy {_interval(sensitivity['veo_vs_consensus_accuracy'])}")
    checks = result["preregistered_checks"]
    lines += ["", "| Pre-registered check | Result |", "|---|---|"]
    lines.append(f"| Krippendorff α ≥ 0.4 | {'pass' if checks['inter_rater_alpha_at_least_0_4'] else 'fail'} |")
    lines.append(f"| Selective accuracy above majority baseline | {'pass' if checks['selective_accuracy_above_majority_baseline'] else 'fail'} |")
    classes = ", ".join(checks["classes_with_precision_above_base_rate"]) or "none"
    lines.append(f"| ≥ 3 classes with precision above base rate | {'pass' if checks['at_least_three_classes_above_base_rate'] else 'fail'} ({classes}) |")
    return lines + [""]


def _auroc_cell(entry: dict | None) -> str:
    if not entry or entry.get("auroc") is None:
        return "-"
    return f"{entry['auroc']:.3f} [{entry['lower']:.3f}, {entry['upper']:.3f}]"


def _f1_cell(entry: dict | None) -> str:
    if not entry:
        return "-"
    return f"{entry['precision']:.2f} / {entry['recall']:.2f} / {entry['f1']:.2f}"


def _silver(result: dict) -> list[str]:
    test = result["test"]
    evaluation = test["evaluation"]
    lines = [
        "## Objective (silver) labels from NYU annotations",
        "",
        f"- Fitted on {result['train']['regions']:,} train regions; scored on {evaluation['regions']:,} test regions "
        f"from {evaluation['scenes']} scenes (95% intervals resample scenes)",
        "",
        "| Category | Test positives | AUROC v2 rules | AUROC v3 model | AUROC learned | P / R / F1 v2 rules | P / R / F1 v3 model | P / R / F1 learned |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for category, entry in evaluation["categories"].items():
        lines.append(
            f"| `{category}` | {entry['positives']:,} ({_pct(entry['prevalence'])}) | {_auroc_cell(entry['hand_set'])} | "
            f"{_auroc_cell(entry.get('label_model') or entry.get('calibrated'))} | {_auroc_cell(entry.get('learned'))} | {_f1_cell(entry['hand_set'])} | "
            f"{_f1_cell(entry.get('label_model') or entry.get('calibrated'))} | {_f1_cell(entry.get('learned'))} |"
        )
    lines += ["", "| VEO top label | Predicted | Confirmed by silver label |", "|---|---|---|"]
    for category, row in test["top_label_precision"].items():
        share = _pct(row["precision"]) if row["precision"] is not None else "-"
        lines.append(f"| `{category}` | {row['predicted']:,} | {row['confirmed']:,} ({share}) |")
    return lines + [""]


def _geometry_agreement(result: dict) -> list[str]:
    lines = [
        "## Labels with geometry from the aligned prediction",
        "",
        "Not usable without ground truth: regions, signed error and sensor holes still come from the error target.",
        "",
        f"- Same top label as with true geometry on {_pct(result['top_cause_agreement'])} of {result['regions']:,} regions "
        f"(κ = {result['top_cause_kappa']:.3f})",
        "",
        "| Label | Regions (audit) | Regions (prediction only) | Both |",
        "|---|---|---|---|",
    ]
    for label, row in result["per_label"].items():
        lines.append(f"| `{label}` | {row['audit']:,} | {row['prediction_only']:,} | {row['both']:,} |")
    return lines + [""]


def _failure_rules(result: dict) -> list[str]:
    lines = ["## Failure-rule sensitivity", "", "| Rule | Regions | Failing px / image | Top labels (share of regions) |", "|---|---|---|---|"]
    for name, row in result["rules"].items():
        total = row["num_regions"] or 0
        top = ", ".join(f"{label} {_pct(count / total)}" for label, count in sorted(row["cause_counts"].items(), key=lambda item: -item[1])[:3]) if total else "-"
        failing = _pct(row["mean_failure_pixel_fraction"]) if row.get("mean_failure_pixel_fraction") is not None else "-"
        lines.append(f"| {name} | {total:,} | {failing} | {top} |")
    return lines + [""]


def _alignment(result: dict) -> list[str]:
    lines = ["## Alignment protocols", "", "| Protocol | AbsRel | δ1 | s > 0 | s < 0 | mean s ± sd |", "|---|---|---|---|---|---|"]
    for protocol, row in result["summary"].items():
        signs = row["scale_sign_counts"]
        scale = f"{row['mean_scale']:.3f} ± {row['std_scale']:.3f}" if row.get("std_scale") is not None else "-"
        metrics = row["mean_metrics"]
        lines.append(f"| {protocol} | {metrics['abs_rel']:.4f} | {metrics['delta1']:.4f} | {signs['positive']} | {signs['negative']} | {scale} |")
    return lines + [""]


def _ablation(result: dict) -> list[str]:
    lines = ["## Signal ablation", "", "| Signal | Neutral value | Labels changed |", "|---|---|---|"]
    for row in result["rows"]:
        lines.append(f"| `{row['signal']}` | {row['neutral_value']:g} | {row['changed_regions']}/{row['total_regions']} ({_pct(row['changed_fraction'])}) |")
    return lines + [""]


def _sanity(result: dict) -> list[str]:
    if "before" not in result:
        return ["## Scoring sanity check", "", f"- {result.get('skipped', 'not run')}", ""]
    return [
        "## Scoring sanity check (formula only; model not re-run)",
        "",
        f"- {result['sample_id']} region {result['region_id']}: textureless score "
        f"{result['before']['textureless_score']:.3f} → {result['after']['textureless_score']:.3f}",
        "",
    ]


def _oracle(result: dict) -> list[str]:
    mean = result["mean"]
    if not mean:
        return ["## Oracle boundary filter (ground-truth masks)", "", "- No eligible scenes", ""]
    def change(key: str) -> str:
        before, after = mean[f"{key}_before"], mean[f"{key}_after"]
        return f"{before:.4f} → {after:.4f} ({100 * (after - before) / before:+.2f}%)"
    return [
        "## Oracle boundary filter (ground-truth masks; not deployable)",
        "",
        f"- Scenes: {result['evaluated_scenes']}",
        f"- Regional AbsRel {change('region_absrel')}",
        f"- Regional RMSE {change('region_rmse')}",
        f"- Global RMSE {change('global_rmse')}",
        "",
    ]


def _hypersim(result: dict) -> list[str]:
    paired = result["paired_reflection_vs_control"]
    stats = paired["difference"]
    lines = [
        "## Hypersim: removing non-diffuse (specular) light",
        "",
        f"- {result['frames']} frames from {result['scenes']} test scenes; {result['regions']:,} failure regions, "
        f"{result['regions_labelled_reflection']} labelled reflection",
    ]
    if stats.get("n", 0) >= 2:
        lines.append(
            f"- Reflection vs matched controls ({paired['pairs']} pairs): labelled Δ {paired['mean_target_change']:+.4f}, "
            f"control Δ {paired['mean_control_change']:+.4f}, difference {stats['mean']:+.4f} "
            f"[{stats['lower']:+.4f}, {stats['upper']:+.4f}], p = {stats['p_value']:.4f}"
        )
    else:
        lines.append(f"- Reflection vs matched controls: {paired['pairs']} pairs (too few to test)")
    dose = result.get("dose_response_spearman")
    if dose:
        lines.append(f"- Specular light vs change in region error (Spearman): {dose['estimate']:+.3f} [{dose['lower']:+.3f}, {dose['upper']:+.3f}]")
    for name, entry in result["detection_auroc"].items():
        if entry:
            lines.append(f"- VEO reflection score, AUROC for {name.replace('_', ' ')} ({entry['positives']} regions): "
                         f"{entry['estimate']:.3f} [{entry['lower']:.3f}, {entry['upper']:.3f}]")
    change = result.get("global_abs_rel_change")
    if change and change.get("n", 0) >= 2:
        lines.append(f"- Whole-image AbsRel change: {change['mean']:+.4f} [{change['lower']:+.4f}, {change['upper']:+.4f}]")
    check = result.get("external_label_check")
    if check:
        lines += ["", "External check of the labels (Hypersim annotations; v3 was fitted on NYU and not refitted):", "",
                  "| Category | Positives | AUROC v2 rules | AUROC v3 model | P / R / F1 v3 model |", "|---|---|---|---|---|"]
        for category, entry in check["categories"].items():
            lines.append(f"| `{category}` | {entry['positives']:,} ({_pct(entry['prevalence'])}) | {_auroc_cell(entry['hand_set'])} | "
                         f"{_auroc_cell(entry.get('label_model'))} | {_f1_cell(entry.get('label_model'))} |")
    return lines + [""]


def _patch_cnn(result: dict) -> list[str]:
    if "categories" not in result:
        return []
    lines = ["## Small CNN on region patches (learned reference, test split)", "", "| Category | AUROC | P / R / F1 |", "|---|---|---|"]
    for category, entry in result["categories"].items():
        lines.append(f"| `{category}` | {_auroc_cell(entry)} | {_f1_cell(entry)} |")
    return lines + [""]


def _model_comparison(result: dict) -> list[str]:
    lines = ["## Model comparison (scene-level paired tests, Holm-corrected within each contrast)", ""]
    for name, contrast in result["contrasts"].items():
        lines += [f"**{name}**: {contrast['first']} minus {contrast['second']} ({contrast['scenes']} scenes)", "",
                  "| Outcome | First | Second | Difference [95% CI] | p (Holm) | Reliable |", "|---|---|---|---|---|---|"]
        for family in ("outcomes", "composition"):
            rows = contrast.get(family, {})
            rows = rows.get("outcomes", {}) if family == "composition" else rows
            for outcome, row in rows.items():
                name_cell = f"share of region pixels: {outcome}" if family == "composition" else outcome
                lines.append(f"| {name_cell} | {row['first_mean']:.4f} | {row['second_mean']:.4f} | {row['mean']:+.4f} [{row['lower']:+.4f}, {row['upper']:+.4f}] | "
                             f"{row['p_holm']:.4f} | {'yes' if row['reliable'] else 'no'} |")
        lines.append("")
    if result.get("skipped"):
        lines += [f"Not run (models missing): {', '.join(result['skipped'])}", ""]
    return lines


def _finetune(result: dict) -> list[str]:
    lines = ["## Guided fine-tuning (docs/FINETUNE_PROTOCOL.md)", "", "Each row is first minus second, so a negative difference favours the first condition.", "",
             "| Test | Difference [95% CI] | p (Holm) | Seeds agree | Supported |", "|---|---|---|---|---|"]
    for family in ("primary", "secondary"):
        for name, row in result[family].items():
            lines.append(f"| {name} | {row['mean']:+.4f} [{row['lower']:+.4f}, {row['upper']:+.4f}] | {row['p_holm']:.4f} | {'yes' if row['seeds_agree'] else 'no'} | "
                         f"{('yes' if row['supported'] else 'no') if 'supported' in row else '-'} |")
    lines += ["", "| Condition | AbsRel | delta1 | Failing pixels | AbsRel on boundary pixels | AbsRel on wall and ceiling pixels |", "|---|---|---|---|---|---|"]
    for name, row in result["condition_means"].items():
        lines.append(f"| {name} | {row['abs_rel']:.4f} | {row['delta1']:.4f} | {row['failing_pixels']:.4f} | {row['abs_rel_boundary_mixing']:.4f} | {row['abs_rel_textureless']:.4f} |")
    return lines + [""]


def _ibims_labels(result: dict) -> list[str]:
    lines = ["## External label check: iBims-1 (frozen label model v3)", "", f"{result['regions']} regions from {result['images']} images.", "",
             "| Label | Positives | AUROC (v3) [95% CI] | AUROC (v2 rules) |", "|---|---|---|---|"]
    for category, entry in result["categories"].items():
        lines.append(f"| `{category}` | {entry['positives']} | {_auroc_cell(entry['label_model'])} | {_auroc_cell(entry['hand_set'])} |")
    return lines + ["", f"`textureless` against any plane mask: {_auroc_cell(result['textureless_any_plane'])}. "
                    f"`reflection` from `sensor_hole_fraction` alone: {_auroc_cell(result['reflection_sensor_holes_only'])}.", ""]


def write_summary_markdown(results_dir: Path) -> Path:
    """Per-model folders (summary.json, alignment_ablation.json) first, then run-level experiments."""
    lines = [f"# Results: {results_dir.name}", "", "Generated by `veo.summary` from the JSON files in this folder.", ""]
    for model_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        summary = _load(model_dir / "summary.json")
        if summary:
            lines += _model(model_dir.name, summary)
        alignment = _load(model_dir / "alignment_ablation.json")
        if alignment:
            lines += _alignment(alignment)
        rules = _load(model_dir / "failure_rules.json")
        if rules:
            lines += _failure_rules(rules)
        interventions = _load(model_dir / "interventions.json")
        if interventions:
            lines += _interventions(interventions)
        study = _load(model_dir / "human_study.json")
        if study:
            lines += _human_study(study)
        silver = _load(model_dir / "silver_calibration.json")
        if silver:
            lines += _silver(silver)
        cnn = _load(model_dir / "patch_cnn.json")
        if cnn:
            lines += _patch_cnn(cnn)
        agreement = _load(model_dir / "geometry_agreement.json")
        if agreement and agreement.get("regions"):
            lines += _geometry_agreement(agreement)
    sections = [
        ("human_agreement.json", _human),
        ("signal_ablation.json", _ablation),
        ("scoring_sanity_check.json", _sanity),
        ("oracle_boundary_filter.json", _oracle),
        ("hypersim_reflection.json", _hypersim),
        ("model_comparison.json", _model_comparison),
        ("label_check.json", _ibims_labels),
        ("guided_finetune.json", _finetune),
    ]
    for filename, render in sections:
        result = _load(results_dir / filename)
        if result:
            lines += render(result)
    path = results_dir / "SUMMARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
