"""scripts/merge_shards.py: concat + dedupe on episode_id. No habitat."""
import importlib.util
import os

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "merge_shards",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "merge_shards.py"),
)
merge_shards = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(merge_shards)


def _write(path, rows):
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")


def test_merge_dedupes_on_episode_id(tmp_path):
    a = tmp_path / "episodes.shard0.csv"
    b = tmp_path / "episodes.shard1.csv"
    _write(str(a), ["episode_id,success", "1,1.0", "3,0.0"])
    _write(str(b), ["episode_id,success", "3,0.0", "2,1.0"])  # 3 is a duplicate
    n = merge_shards._merge_csv([str(a), str(b)], str(tmp_path / "episodes.csv"))
    assert n == 3
    out = (tmp_path / "episodes.csv").read_text().splitlines()
    assert out[0] == "episode_id,success"
    assert sorted(line.split(",")[0] for line in out[1:]) == ["1", "2", "3"]


def test_merge_rejects_mismatched_headers(tmp_path):
    a = tmp_path / "episodes.shard0.csv"
    b = tmp_path / "episodes.shard1.csv"
    _write(str(a), ["episode_id,success", "1,1.0"])
    _write(str(b), ["episode_id,spl", "2,0.5"])
    with pytest.raises(SystemExit):
        merge_shards._merge_csv([str(a), str(b)], str(tmp_path / "episodes.csv"))
