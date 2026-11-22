from stages.qa.review_paths import clean_review, panels_dir


def test_clean_review_removes_only_this_hands_files(tmp_path, monkeypatch):
    monkeypatch.setenv("E2H_BULK_DIR", str(tmp_path / "bulk"))
    clip = tmp_path / "basic_pick_place_1"
    (clip / "review").mkdir(parents=True)
    panels = panels_dir(clip)
    assert panels == tmp_path / "bulk" / "basic_pick_place_1" / "panels"
    mine = [clip / "review" / "result_grid_sharpa.mp4", panels / "ego_d435_sharpa.npz"]
    keep = [clip / "review" / "result_grid.mp4", panels / "tl.mp4"]
    for f in mine + keep:
        f.write_bytes(b"x")
    removed = clean_review(clip, "_sharpa")
    assert set(removed) == set(mine)
    assert all(f.exists() for f in keep)
