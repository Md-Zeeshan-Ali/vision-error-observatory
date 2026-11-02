import sys
from pathlib import Path
from typing import Protocol

import numpy as np


class DepthPredictor(Protocol):
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Return a depth array with the same height and width as the RGB image."""


class PrecomputedPredictor:
    def __init__(self, prediction_dir: Path):
        self.prediction_dir = prediction_dir

    def predict_for_sample(self, sample_id: str, expected_shape: tuple[int, int]) -> np.ndarray:
        path = self.prediction_dir / f"{sample_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction for {sample_id}: {path}")
        prediction = np.asarray(np.load(path), dtype=np.float32)
        if prediction.shape != expected_shape:
            raise ValueError(f"Prediction {path} has shape {prediction.shape}; expected {expected_shape}")
        return prediction


# These checkpoints output metric depth; every other supported model outputs relative disparity
METRIC_DEPTH_MODELS = {"Intel/zoedepth-nyu-kitti", "apple/DepthPro-hf"}


class TransformersDepthPredictor:
    """Hugging Face depth pipeline; always returns disparity-like values, inverting metric-depth models."""

    def __init__(self, model_id: str, device: str = "auto"):
        self.inverts_output = model_id in METRIC_DEPTH_MODELS
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError("Install the model extra with: pip install -e '.[model]'") from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline_device = 0 if device == "cuda" else -1
        self._pipe = pipeline("depth-estimation", model=model_id, device=pipeline_device)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        result = self._pipe(Image.fromarray(rgb))
        return self._as_disparity(self._prediction_from_result(result, rgb.shape[:2]))

    def _as_disparity(self, prediction: np.ndarray) -> np.ndarray:
        return 1.0 / np.clip(prediction, 1e-3, None) if self.inverts_output else prediction

    @staticmethod
    def _prediction_from_result(result, expected_shape: tuple[int, int]) -> np.ndarray:
        prediction = result["predicted_depth"].squeeze().detach().cpu().numpy().astype(np.float32)
        if prediction.shape != expected_shape:
            import cv2

            prediction = cv2.resize(prediction, (expected_shape[1], expected_shape[0]))
        return prediction

    def predict_batch(self, rgb_images: list[np.ndarray], batch_size: int = 4) -> list[np.ndarray]:
        from PIL import Image

        results = self._pipe([Image.fromarray(rgb) for rgb in rgb_images], batch_size=batch_size)
        return [
            self._as_disparity(self._prediction_from_result(result, rgb.shape[:2]))
            for result, rgb in zip(results, rgb_images, strict=True)
        ]


# This checkpoint outputs affine-invariant depth, so it is aligned in depth space and never inverted
AFFINE_DEPTH_MODELS = {"prs-eth/marigold-depth-v1-1"}


class MarigoldDepthPredictor:
    """Diffusion depth model with the settings in docs/MODEL_COMPARISON_PROTOCOL.md; returns affine-invariant depth."""

    def __init__(self, model_id: str, device: str = "auto", seed: int = 42):
        try:
            import torch
            from diffusers import MarigoldDepthPipeline
        except ImportError as exc:
            raise ImportError("Install the model extra with: pip install -e '.[model]'") from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        half = device == "cuda"
        self._torch, self._device, self._seed = torch, device, seed
        self._pipe = MarigoldDepthPipeline.from_pretrained(model_id, variant="fp16" if half else None, torch_dtype=torch.float16 if half else torch.float32).to(device)
        self._pipe.set_progress_bar_config(disable=True)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        # A new generator per image, so a resumed run gives the same result as an uninterrupted one
        generator = self._torch.Generator(device=self._device).manual_seed(self._seed)
        output = self._pipe(Image.fromarray(rgb), num_inference_steps=4, ensemble_size=1, processing_resolution=768, match_input_resolution=True, generator=generator)
        return np.asarray(output.prediction, dtype=np.float32).squeeze()


def load_predictor(model_id: str, device: str = "auto") -> DepthPredictor:
    if model_id in AFFINE_DEPTH_MODELS:
        return MarigoldDepthPredictor(model_id, device)
    return TransformersDepthPredictor(model_id, device)


class OfficialMetricDepthPredictor:
    """Official Depth Anything V2 metric-depth implementation and checkpoint."""

    def __init__(self, official_repo: Path, checkpoint: Path, encoder: str = "vits", max_depth: float = 20.0, device: str = "auto"):
        try:
            import cv2
            import torch
        except ImportError as exc:
            raise ImportError("Install model dependencies with: uv sync --extra model") from exc
        if not checkpoint.exists():
            raise FileNotFoundError(f"Metric checkpoint not found: {checkpoint}")
        metric_depth = official_repo / "metric_depth"
        for path in (official_repo, metric_depth):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from depth_anything_v2.dpt import DepthAnythingV2

        configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        if encoder not in configs:
            raise ValueError(f"Unsupported official metric encoder: {encoder}")
        self._device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self._device == "auto":
            self._device = "cpu"
        self._cv2 = cv2
        self._model = DepthAnythingV2(**{**configs[encoder], "max_depth": max_depth})
        self._model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self._model = self._model.to(self._device).eval()

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        bgr = self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR)
        return self._model.infer_image(bgr).astype(np.float32)