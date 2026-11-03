import http.server
import io
import socketserver
import threading
import zipfile

import numpy as np
import pytest

from veo.download import HttpRangeFile
from veo.hypersim import select_frames, standard_camera_scenes, tonemap


def test_tonemap_maps_90th_percentile_to_0_8():
    color = np.random.default_rng(0).uniform(0, 2, (64, 64, 3)).astype(np.float32)
    mapped, scale = tonemap(color)
    brightness = 0.3 * color[..., 0] + 0.59 * color[..., 1] + 0.11 * color[..., 2]
    assert scale == pytest.approx(0.8**2.2 / np.percentile(brightness, 90))
    assert mapped.dtype == np.uint8


def test_same_exposure_keeps_diffuse_part_identical_and_darker_overall():
    rng = np.random.default_rng(1)
    diffuse = rng.uniform(0.1, 1, (32, 32, 3)).astype(np.float32)
    residual = np.zeros_like(diffuse)
    residual[8:16, 8:16] = 3.0  # a specular highlight
    full, scale = tonemap(diffuse + residual)
    without, _ = tonemap(diffuse, scale)
    assert np.array_equal(full[:8], without[:8])  # untouched where there is no residual
    assert without[8:16, 8:16].mean() < full[8:16, 8:16].mean()


def _write_csv(path, header, rows):
    path.write_text("\n".join([",".join(header)] + [",".join(map(str, row)) for row in rows]) + "\n", encoding="utf-8")


def test_frame_selection_uses_standard_cameras_and_the_split(tmp_path):
    header = ["scene_name", "settings_output_img_width", "M_proj_00", "M_proj_02", "M_proj_12"]
    _write_csv(tmp_path / "cams.csv", header, [["a", 1024.0, 1.7320508, 0, 0], ["b", 1024.0, 1.7320508, 0.2, 0], ["c", 1024.0, 1.7320508, 0, 0]])
    assert standard_camera_scenes(tmp_path / "cams.csv") == {"a", "c"}
    split_header = ["scene_name", "camera_name", "frame_id", "included_in_public_release", "exclude_reason", "split_partition_name"]
    rows = [[scene, "cam_00", frame, "True", "", split] for scene, split in (("a", "test"), ("b", "test"), ("c", "train")) for frame in range(10)]
    rows.append(["a", "cam_00", 99, "False", "excluded", "test"])
    _write_csv(tmp_path / "split.csv", split_header, rows)
    frames = select_frames(tmp_path / "split.csv", tmp_path / "cams.csv", "test", per_scene=3, seed=0)
    assert len(frames) == 3 and {scene for scene, _, _ in frames} == {"a"}
    assert all(frame != 99 for _, _, frame in frames)


@pytest.fixture
def zip_server():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("scene/images/a.bin", bytes(range(256)) * 2000)
        archive.writestr("scene/images/b.txt", "hello")
    payload = buffer.getvalue()

    class RangeHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()

        def do_GET(self):
            start, end = (int(v) for v in self.headers["Range"].split("=")[1].split("-"))
            body = payload[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/scene.zip"
    server.shutdown()
    server.server_close()


def test_remote_zip_members_are_read_with_range_requests(zip_server):
    archive = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(zip_server), buffer_size=4096))
    assert sorted(archive.namelist()) == ["scene/images/a.bin", "scene/images/b.txt"]
    assert archive.read("scene/images/b.txt") == b"hello"
    assert archive.read("scene/images/a.bin") == bytes(range(256)) * 2000
