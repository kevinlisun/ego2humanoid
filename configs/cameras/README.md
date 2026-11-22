# `configs/cameras/` — robot camera models

| File | Purpose |
|---|---|
| `g1_head_cam_v1.json` | Intel D435 [[Intel RealSense](../../docs/REFERENCES.md#realsense-d435)] head camera of the Unitree G1 [[Unitree Robotics](../../docs/REFERENCES.md#unitree-g1)], shared by planner, renderer and exporter |

## `g1_head_cam_v1.json` keys

| Key | Value | Unit | Meaning | Read by |
|---|---|---|---|---|
| `name` | `g1_head_cam_v1` | – | Identifier | not read |
| `mjcf_camera` | `d435` | – | Name of the camera in the MJCF | not read by `HeadCamera` (the camera name is defined in `stages/manipulation/mjcf_scene.py`) |
| `parent_body` | `torso_link` | – | Body the camera is rigidly attached to (G1 has no neck) | `HeadCamera.parent_body` |
| `image_hw` | `[360, 640]` | px (height, width) | Render / intrinsics size | `HeadCamera.height`, `.width` |
| `fovy_deg` | `42.5` | deg | Vertical field of view; focal length `f = 0.5 * H / tan(fovy/2)`, principal point at image centre | `HeadCamera.K`, `.fovx_deg` |
| `note` | text | – | D435 colour 16:9 (69.4 x 42.5 deg); gaze follows waist pose and base stance; mounting pose constants `D435_*` live in `stages/manipulation/mjcf_scene.py` | not read |

## Code

- Loader: `src/robot/head_camera.py` — `HeadCamera.load(path=CONFIG_PATH)`, properties `K`, `fovx_deg`;
  `T_world_cam(model, data)` (OpenCV axes, from measured torso FK), `project(T_world_cam, pts_world) -> (uv, depth)`.
- Pixel convention: OpenCV (x right, y down, z forward); `MJ_TO_CV = diag(1, -1, -1)`.
- Users: `src/stages/locomotion/episode.py` (`HeadCamera.load()`), planning visibility in `stages/locomotion/plan/`.
  Note `stages/reconstruction/align/camera_fit.py` defines its own `HeadCamera(model)` class (not this loader).

## Gotchas

- Changing `fovy_deg`/`image_hw` changes visibility scoring and rendered views but not the MJCF camera, which has its own
  constants in `mjcf_scene.py`; keep them consistent.
- Tests: `tests/unit/test_head_camera.py`.
- Parent: [`../README.md`](../README.md).

## Related docs

- [`docs/TECH_DESIGN.md`](../../docs/TECH_DESIGN.md) — camera/gaze design.
- [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) — where the planner, renderer and exporter sit.
- Scene that places the camera: [`../scenarios/README.md`](../scenarios/README.md).

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](../../docs/REFERENCES.md).

- <span id="cite-realsense-d435"></span>[Intel RealSense](../../docs/REFERENCES.md#realsense-d435). *Intel RealSense Depth Camera D435.*
- <span id="cite-unitree-g1"></span>[Unitree Robotics](../../docs/REFERENCES.md#unitree-g1). *G1 humanoid robot and Dex3-1 dexterous hand.*
