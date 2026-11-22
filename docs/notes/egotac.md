# EgoTac integration notes

EgoTac (Mr-Zwkid/EgoTac) predicts per-frame tactile force and contact probability for both hands on the
778-vertex MANO topology from RGB. It does not estimate hand pose, so FAH stays the pose backend and EgoTac is
a configurable tactile head (`configs/hands.yaml`, `tactile.backend: none|egotac`).

## Setup (hot disk, ephemeral)

The NVMe (`/dev/nvme0n1`) is not persistent. Recreate when needed (or use the `nodes/tactile` image via
`DOCKER=1 scripts/run_pipeline.sh`; the tactile stage only runs when `EGOTAC_PY` is set, and `tactile.backend`
in `configs/hands.yaml` is `none` by default):

```bash
sudo mkfs.ext4 -L nvme_hot /dev/nvme0n1 && sudo mkdir -p /mnt/nvme_hot && sudo mount -o noatime /dev/nvme0n1 /mnt/nvme_hot
uv venv --python 3.10 /mnt/nvme_hot/envs/egotac
uv pip install --python /mnt/nvme_hot/envs/egotac/bin/python torch==2.5.1 torchvision==0.20.1 numpy==1.26.4 timm==1.0.24 \
  hydra-core==1.3.2 omegaconf opencv-python-headless==4.11.0.86 "Pillow<12" huggingface_hub==0.36.0 scipy==1.15.3 \
  scikit-learn==1.7.2 zarr==2.16.0 numcodecs==0.12.1 diffusers==0.35.2 dill==0.4.0 accelerate==1.11.0 matplotlib tqdm
hf download wkzhang/EgoTac egotac_base_h4.pt --local-dir /mnt/nvme_hot/weights/egotac   # Apache-2.0, not gated
ln -s /mnt/nvme_hot/weights/egotac data/weights/egotac
/mnt/nvme_hot/envs/egotac/bin/python scripts/estimate_tactile_egotac.py data/derived/egodex/<clip>
```

## Output

`tactile_egotac.npz`: `force (N,2,778)`, `contact_prob (N,2,778)` (hand order [left, right], verified against the GT
wrist motion on 10 clips), `frame_idx`, `source`. Force is a normalised model output (about -0.56..0.27), not Newtons.
