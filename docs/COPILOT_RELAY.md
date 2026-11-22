# Copilot relay: answering SimFoundry VLM requests

> **Scope.** Optional scene-reconstruction path only. SimFoundry [[NVIDIA GEAR, 2026](REFERENCES.md#simfoundry2026)] queries a vision-language model
> at several stages; this relay lets a human-supervised assistant session answer those queries offline, so that
> pipeline code never calls an external API and every answer is stored with the run for replay.

Pipeline code never calls an external VLM API (D8). When a SimFoundry stage needs a VLM
answer that is not in the inbox yet, it writes a bundle and exits with code 75
(`src/stages/reconstruction/scene/vlm_relay.py`, `PENDING_RELAY_EXIT_CODE`). The same relay carries the
RSI Mode B requests (`scripts/real2sim_loop.py`, `src/rsi/loop.py`).

## Loop

1. Run the stage. If it exits with 75, run `ego2humanoid relay status <run>/scene/relay`.
2. For each pending id, read `outbox/<id>/prompt.md` and view `outbox/<id>/images/NN.png`
   (in order; they precede the prompt text). `request.json` holds the model label.
3. Answer exactly as the prompt demands (format, JSON schema, no extra commentary) and
   write `inbox/<id>.json`:

   ```json
   {"text": "<answer>", "model": "<model used>", "answered_by": "copilot-relay"}
   ```

   Text only; image-editing calls use the local FLUX.1 Kontext [[Black Forest Labs, 2025](REFERENCES.md#batifol2025flux)] model instead.
4. `ego2humanoid relay validate <relay_dir>` then re-run the stage. Repeat until status
   reports 0 pending.

## Rules

- Only frames approved for the relay (licence review Q6, see `LICENSES.md`) may be viewed; never copy them elsewhere.
- Keep the inbox with the run: it makes the scene replayable with `TEST_MODE=1`.

## Appearance loop (Track BT)

`scripts/twin_appearance_rsi.py <clip_dir>` uses the same relay (`<clip>/review/twin_appearance/relay`).
The request image is `real | twin | diff` for the human-camera view; answer with JSON only:
`{"edits": [{"path": "objects.<name>.<field>" | "table.<field>" | "light.<field>" | "room.<field>", "value": ...}], "rationale": "..."}`
(`src/contracts/twin_appearance.py` lists the fields and clamps the ranges). Rejected edits are printed, the
round is rolled back if the score does not improve. Same rules as above: view only approved frames.

## References

Works cited in this document; full entries in [`docs/REFERENCES.md`](REFERENCES.md).

- <span id="cite-batifol2025flux"></span>[Black Forest Labs, 2025](REFERENCES.md#batifol2025flux). *FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing.*
- <span id="cite-simfoundry2026"></span>[NVIDIA GEAR, 2026](REFERENCES.md#simfoundry2026). *SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation.*
