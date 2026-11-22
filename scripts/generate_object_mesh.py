"""Image -> 3D mesh with Hunyuan3D-2mini (shape only; physics needs geometry, not texture).

Run inside the FAH venv (needs ``hy3dgen``). Usage: generate_object_mesh.py <clip_dir> [--scene]
Writes ``<clip_dir>/object_mesh_raw.glb`` (unit scale, 4k faces; best of NUM_SEEDS by silhouette IoU).
With ``--scene`` it meshes every crop of the tabletop inventory instead (``scene/inventory.json`` from
``perceive_scene_objects.py``) into ``scene/mesh_<id>.glb``, with fewer seeds: these objects are
only scenery and obstacles, the held object keeps the full search.
Weights: ``data/weights/hunyuan3d/Hunyuan3D-2mini`` (gitignored; tencent/Hunyuan3D-2mini, NC licence).
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from hy3dgen.shapegen import FaceReducer, FloaterRemover, Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.shapegen.models.conditioner import SingleImageEncoder
from PIL import Image

NUM_SEEDS = 4
SCENE_SEEDS = 1
SCENE_MIN_SCORE = 0.2  # matches the sim gate (inventory.SCENERY_MIN_SCORE); weaker ones stay boxes
_RENAMES = (
    ("attention.attention.query", "attention.q_proj"),
    ("attention.attention.key", "attention.k_proj"),
    ("attention.attention.value", "attention.v_proj"),
    ("attention.output.dense", "attention.o_proj"),
)
_orig_load = SingleImageEncoder.load_state_dict


def _load_renamed(self, state_dict, *args, **kwargs):
    # Hunyuan3D checkpoints use the transformers-4 DINOv2 key names; transformers 5 renamed them.
    def fix(k):
        for old, new in _RENAMES:
            k = k.replace(old, new)
        return k.replace("mlp.weights_out", "mlp.down_proj")

    out = {}
    for k, v in state_dict.items():
        if "mlp.weights_in" in k:  # fused SwiGLU input projection: first half = gate, second half = up
            gate, up = v.chunk(2, dim=0)
            out[fix(k.replace("weights_in", "gate_proj"))] = gate
            out[fix(k.replace("weights_in", "up_proj"))] = up
        else:
            out[fix(k)] = v
    return _orig_load(self, out, *args, **kwargs)


SingleImageEncoder.load_state_dict = _load_renamed


def generate(pipe, mesh_texture, crop: Path, seeds: int):
    image = Image.open(crop).convert("RGBA")
    rgba = np.array(image)
    best = (-1.0, None, -1)
    for seed in range(seeds):
        generator = torch.Generator("cuda").manual_seed(seed)
        cand = pipe(image=image, num_inference_steps=30, octree_resolution=256, generator=generator)[0]
        cand = FaceReducer()(FloaterRemover()(cand), max_facenum=4000)
        # Keep the sample whose silhouette agrees best with the crop (render-and-compare).
        _, iou = mesh_texture.fit_view(cand.sample(20000), rgba)
        print(f"{crop.name} seed={seed} silhouette_iou={iou:.3f}")
        if iou > best[0]:
            best = (iou, cand, seed)
    return best


def main():
    clip = Path(sys.argv[1])
    scene = "--scene" in sys.argv[2:]
    weights = Path(__file__).resolve().parents[1] / "data/weights/hunyuan3d/Hunyuan3D-2mini"
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(weights), subfolder="hunyuan3d-dit-v2-mini", variant="fp16", use_safetensors=True, device="cuda"
    )
    spec = importlib.util.spec_from_file_location(
        "mesh_texture",
        Path(__file__).resolve().parents[1] / "src/stages/reconstruction/objects/mesh_texture.py",
    )
    mesh_texture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mesh_texture)
    if not scene:
        iou, mesh, seed = generate(pipe, mesh_texture, clip / "object_crop.png", NUM_SEEDS)
        mesh.export(clip / "object_mesh_raw.glb")
        print(
            f"seed={seed} iou={iou:.3f} vertices={len(mesh.vertices)} faces={len(mesh.faces)}",
            f"extents={mesh.extents}",
        )
        return
    inv = json.loads((clip / "scene/inventory.json").read_text())
    for f in (clip / "scene").glob("mesh_*.glb"):
        f.unlink()
    for o in inv["objects"]:
        o.pop("mesh", None)
        if o.get("score", 1.0) < SCENE_MIN_SCORE:
            continue
        iou, mesh, _ = generate(pipe, mesh_texture, clip / "scene" / o["crop"], SCENE_SEEDS)
        o["mesh"], o["mesh_iou"] = f"mesh_{o['id']}.glb", round(float(iou), 3)
        mesh.export(clip / "scene" / o["mesh"])
    (clip / "scene/inventory.json").write_text(json.dumps(inv, indent=1))


if __name__ == "__main__":
    main()
