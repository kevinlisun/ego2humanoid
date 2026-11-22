"""Ego2Humanoid v0.2 stage CLI.

Preflight is implemented; upstream GPU adapters remain explicit integration TODOs.
No command that relies on unimplemented adapters silently emits fake trajectories.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import typer

from contracts import ROBOT_ID, EpisodeInput
from stages.adapters import FAHAdapter, SimFoundryAdapter

app = typer.Typer(help="Ego2Humanoid research pipeline - v0.2 contracts; model adapters pending")


def load_episode(path: Path) -> EpisodeInput:
    return EpisodeInput.model_validate_json(path.read_text())


@app.command()
def validate(episode: Path):
    item = load_episode(episode)
    typer.echo(
        f"VALID {item.episode_id}: {item.robot.robot_id}, pure RGB, physics_required={item.physics_required}"
    )


@app.command()
def preflight(episode: Path, video: Path, out: Path):
    """A real, GPU-free input validation & content hash stage."""
    item = load_episode(episode)
    if not video.is_file():
        raise typer.BadParameter(f"Video missing: {video}")
    if video.stat().st_size < 32:
        raise typer.BadParameter("Video too short to be an MP4")
    h = hashlib.sha256()
    with video.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    digest = h.hexdigest()
    if item.video.video.sha256 and item.video.video.sha256 != digest:
        raise typer.BadParameter("Input video SHA256 mismatch")
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,codec_name:format=duration",
                "-of",
                "json",
                str(video),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        probe = json.loads(result.stdout)
        if not probe.get("streams"):
            raise ValueError("No video stream")
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
        raise typer.BadParameter(f"ffprobe video validation failed: {exc}") from exc
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.1",
        "episode_id": item.episode_id,
        "video_sha256": digest,
        "video_bytes": video.stat().st_size,
        "media_probe": probe,
        "robot_id": ROBOT_ID,
        "usage_class": item.usage_class,
        "note": "Preflight passed; this does NOT mean the AI/physics adapters are implemented",
    }
    (out / "preflight.json").write_text(json.dumps(manifest, indent=2) + "\n")
    typer.echo(f"PREFLIGHT {item.episode_id} SHA256 {digest}")


@app.command()
def hand(episode: Path, video: Path, preflight: Path, out: Path):
    FAHAdapter().estimate_hands(load_episode(episode), video, out)


@app.command()
def scene(episode: Path, video: Path, preflight: Path, out: Path):
    SimFoundryAdapter().reconstruct_scene(load_episode(episode), video, out)


@app.command()
def objects(episode: Path, video: Path, hands: Path, scene: Path, out: Path):
    raise NotImplementedError("M2: deserialize hand/scene manifests and implement RGB object-motion tracking")


@app.command()
def align(episode: Path, hands: Path, scene: Path, objects: Path, out: Path):
    raise NotImplementedError("M3: implement Sim3 alignment, scale uncertainty and G1 reachability")


@app.command()
def retarget(episode: Path, hands: Path, scene: Path, objects: Path, aligned: Path, out: Path):
    raise NotImplementedError("M3: implement SPIDER combined G1+Dex3 adapter")


@app.command()
def replay(episode: Path, scene: Path, robot: Path, out: Path):
    raise NotImplementedError("M4: implement action-controlled G1 Dex3 OmniGibson replay")


@app.command()
def qa(episode: Path, sim: Path, robot: Path, out: Path):
    raise NotImplementedError("M4: implement independent, event-based physics quality verification")


relay_app = typer.Typer(help="Copilot VLM relay for SimFoundry (see docs/COPILOT_RELAY.md)")
app.add_typer(relay_app, name="relay")


@relay_app.command("status")
def relay_status_cmd(relay_dir: Path):
    """List pending relay bundles; exit code 75 when some await an answer."""
    from stages.reconstruction.scene.vlm_relay import PENDING_RELAY_EXIT_CODE, relay_status

    st = relay_status(relay_dir)
    for rid in st.pending:
        typer.echo(f"PENDING {rid}  {relay_dir / 'outbox' / rid}")
    typer.echo(f"{len(st.pending)} pending, {len(st.answered)} answered")
    if st.pending:
        raise typer.Exit(PENDING_RELAY_EXIT_CODE)


@relay_app.command("validate")
def relay_validate_cmd(relay_dir: Path):
    """Check that every inbox answer is well-formed."""
    from stages.reconstruction.scene.vlm_relay import validate_inbox

    problems = validate_inbox(relay_dir)
    for p in problems:
        typer.echo(f"INVALID {p}")
    if problems:
        raise typer.Exit(1)
    typer.echo("inbox OK")


scene_app = typer.Typer(help="Reconstructed-scene tools")
app.add_typer(scene_app, name="scene")


@scene_app.command("state")
def scene_state_cmd(
    scene_dir: Path, duration_s: float = 1.0, table_top_z_m: float = 0.8, decompose: bool = True
):
    """Create the objects of ``<scene_dir>/objects.json`` in MuJoCo, settle them and write state + report."""
    import json

    from stages.manipulation.mjcf_scene import SceneConfig
    from stages.reconstruction.scene.export_mjcf import load_scene_objects
    from stages.reconstruction.scene.scene_state import record_scene_state, settle_report

    cfg = SceneConfig(table_top_z_m=table_top_z_m, objects=load_scene_objects(scene_dir, decompose))
    state = record_scene_state(cfg, duration_s)
    state.save(scene_dir / "scene_state.npz")
    report = settle_report(state)
    (scene_dir / "scene_report.json").write_text(json.dumps(report, indent=2))
    typer.echo(f"{len(state.names)} objects, all_stable={report['all_stable']}")
    if not report["all_stable"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()


@app.command("synthetic-run")
def synthetic_run(out: Path):
    """Run the scripted pick-place through physics replay + QA (needs `sim` and `video` extras)."""
    from orchestration.local_runner import run_dag, synthetic_replay_dag

    outs = run_dag(synthetic_replay_dag(), out)
    typer.echo(f"OK: artefacts in {outs['replay']}")
