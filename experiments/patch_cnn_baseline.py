"""A small CNN on region patches, as a learned reference for the label models (trained on train, scored on test).

Input: a crop around the region (RGB plus true disparity), resized to 64x64. Output: one logit per
silver category. This shows how far an image-based model gets against the same silver labels; it is
a reference point and is not part of VEO. Skips itself when PyTorch is not installed.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from veo.calibration import bootstrap_auroc, _label_metrics, build_rows, silver_matrix
from veo.config import config_from_dict
from veo.data import load_rgb
from veo.reports import write_result
from veo.silver import SILVER_CATEGORIES


def patches(rows: list[dict], results_path: Path, size: int = 64) -> np.ndarray:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    split_dir = config_from_dict(payload["config"]).dataset.depth_dir.parent
    batch = np.zeros((len(rows), 4, size, size), dtype=np.float32)
    by_sample: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_sample.setdefault(row["sample_id"], []).append(index)
    for sample_id, indices in by_sample.items():
        rgb = load_rgb(split_dir / "rgb" / f"{sample_id}.png").astype(np.float32) / 255.0
        depth = np.load(split_dir / "depth" / f"{sample_id}.npy")
        disparity = np.where(depth > 0, 1.0 / np.maximum(depth, 1e-3), 0.0).astype(np.float32)
        stacked = np.dstack([rgb, np.clip(disparity, 0, 1)])
        height, width = depth.shape
        for index in indices:
            x, y, w, h = rows[index]["bbox"]
            pad = max(8, max(w, h) // 2)
            crop = stacked[max(y - pad, 0):min(y + h + pad, height), max(x - pad, 0):min(x + w + pad, width)]
            batch[index] = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-results", type=Path, required=True)
    parser.add_argument("--test-results", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True, help="prefix of the silver-label logs from silver_calibration.py")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if importlib.util.find_spec("torch") is None:
        write_result(args.output, {"skipped": "PyTorch is not installed"})
        print("PyTorch is not installed; skipped")
        return
    import torch
    from torch import nn

    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    train_rows, _ = build_rows(args.train_results, args.checkpoints.with_name(args.checkpoints.name + "_train.checkpoint.jsonl"))
    test_rows, _ = build_rows(args.test_results, args.checkpoints.with_name(args.checkpoints.name + "_test.checkpoint.jsonl"))
    x_train = torch.from_numpy(patches(train_rows, args.train_results))
    x_test = torch.from_numpy(patches(test_rows, args.test_results))
    y_train = torch.from_numpy(silver_matrix(train_rows).astype(np.float32))
    y_test = silver_matrix(test_rows)

    def block(channels_in: int, channels_out: int) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(channels_in, channels_out, 3, padding=1), nn.BatchNorm2d(channels_out), nn.ReLU(), nn.MaxPool2d(2))

    model = nn.Sequential(block(4, 32), block(32, 64), block(64, 128), block(128, 256), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, y_train.shape[1])).to(device)
    positive_weight = ((len(y_train) - y_train.sum(0)) / y_train.sum(0).clamp(min=1)).clamp(max=20).to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for indices in torch.randperm(len(x_train)).split(128):
            optimizer.zero_grad()
            loss = loss_function(model(x_train[indices].to(device)), y_train[indices].to(device))
            loss.backward()
            optimizer.step()
            total += float(loss) * len(indices)
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / len(x_train):.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        scores = torch.cat([torch.sigmoid(model(batch.to(device))).cpu() for batch in x_test.split(256)]).numpy()
    groups = np.asarray([row["scene"] for row in test_rows])
    report = {}
    for column, category in enumerate(SILVER_CATEGORIES):
        labels = y_test[:, column]
        report[category] = {"positives": int(labels.sum()), **bootstrap_auroc(labels, scores[:, column], groups), **_label_metrics(labels, scores[:, column] >= 0.5)}
        print(f"{category:<16} AUROC {report[category]['auroc']}")
    write_result(args.output, {"train_regions": len(train_rows), "test_regions": len(test_rows), "epochs": args.epochs, "categories": report})
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
