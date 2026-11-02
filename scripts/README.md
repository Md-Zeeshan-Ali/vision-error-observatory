# Scripts

Analysis lives in `src/veo` and the per-result experiments live in [`experiments/`](../experiments/). This folder holds presentation tooling only.

| Script | Output | Notes |
|---|---|---|
| `site/export_data.py` | `web_app/data/app_data_v2.js` | Needs a finished `veo reproduce` run; picks demo scenes from the test split. |
| `site/build_index.py` | `web_app/index.html` | Static explorer page. |
| `site/build_model_lab.py` | `web_app/model_lab.html` | 3D teaching lab page. |
| `site/generate_video.py` | `web_app/veo_explainer.mp4` | Slide video rendered with OpenCV/PIL. |

Run every script from the repository root, for example `uv run python scripts/site/export_data.py`.

`web_app/veo_explainer.mp4` and `web_app/data/app_data_v2.js` are not in git; they are assets of the `site-assets-v1` GitHub release, downloaded by `python web_app/fetch_assets.py` (the Dockerfile runs it at build time). After regenerating either file, publish it as a new release (for example `site-assets-v2`) and update `RELEASE` and the SHA-256 values in `web_app/fetch_assets.py`; `tests/test_fetch_assets.py` fails if the checksums and the local files disagree.
