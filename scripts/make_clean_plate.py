"""Clean background plate: remove people and objects from one source frame with local FLUX.1-Kontext-dev.

Run with the ``simfoundry`` conda env python, e.g.
``HF_HOME=/mnt/nvme_hot/hf /mnt/nvme_hot/envs/conda_envs/simfoundry/bin/python scripts/make_clean_plate.py
data/scenes/egodex_104/s1_video/frames_subsampled_15/frame_0081.png data/scenes/egodex_104/plate.png``
"""

import sys

import torch
from diffusers import FluxKontextPipeline
from PIL import Image

PROMPT = (
    "Remove both human arms, hands, sleeves and every object (bowl, ball, sponge, jar, toys) from the table. "
    "Keep the pink tablecloth, the wall and the backdrop exactly as they are, "
    "with the same lighting and camera view."
)

src, dst = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
img = Image.open(src).convert("RGB")
w, h = img.size
w2, h2 = 1280, 720
pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")
out = pipe(
    image=img.resize((w2, h2)),
    prompt=PROMPT,
    height=h2,
    width=w2,
    guidance_scale=2.5,
    num_inference_steps=28,
    generator=torch.Generator("cuda").manual_seed(seed),
).images[0]
out.resize((w, h)).save(dst)
print(dst)
