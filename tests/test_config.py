"""Config loading + $VLNSAFE_* expansion + the Phase C agent registry. No habitat."""
import contextlib
import io
import json
import os

import pytest

from vlnsafe import config

ALL_AGENTS = ["seq2seq", "cma", "navila", "streamvln", "bevbert", "etpnav", "etpr1"]


def test_env_var_expansion(monkeypatch):
    monkeypatch.setenv("VLNSAFE_CHECKPOINTS", "/tmp/ckpts")
    agent = config.load_agent("etpnav")
    assert agent["checkpoint"]["r2r"].startswith("/tmp/ckpts/")
    assert "$" not in agent["checkpoint"]["r2r"]


def test_agent_id_must_match_filename():
    assert config.load_agent("etpnav")["id"] == "etpnav"
    with pytest.raises(ValueError):
        config.load_agent("does_not_exist")


def test_load_run_rejects_bad_regime():
    with pytest.raises(ValueError):
        config.load_run("etpnav", "r2r", "sideways")


def test_load_run_shape():
    run = config.load_run("etpnav", "r2r", "block")
    assert set(run) == {"defaults", "benchmark", "agent", "regime"}
    assert run["benchmark"]["id"] == "r2r"
    assert run["defaults"]["seed"] == 20260704


# --- Phase C: one config per agent, all valid, no agent-specific literal in the harness ---

@pytest.mark.parametrize("agent_id", ALL_AGENTS)
def test_every_agent_config_loads_and_validates(agent_id):
    cfg = config.load_agent(agent_id)
    assert cfg["id"] == agent_id
    assert cfg["tier"] in config.AGENT_TIERS
    assert cfg["family"] in config.AGENT_FAMILIES
    for key in ("display_name", "sensor_rig"):
        assert cfg.get(key), "{} missing {}".format(agent_id, key)
    # every \nmark the paper draws must be declared with a reason
    for reason in cfg.get("unavailable", {}).values():
        assert isinstance(reason, str) and reason.strip()


def test_registry_families_come_from_config():
    from vlnsafe.agents import registry
    assert registry.FAMILIES is config.AGENT_FAMILIES


def test_agent_config_rejects_missing_required_key(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "broken.yaml").write_text("id: broken\nfamily: vlnce_policy\nsensor_rig: x\n")
    (agents / "bad_family.yaml").write_text(
        "id: bad_family\ndisplay_name: X\ntier: T1\nfamily: nope\nsensor_rig: x\n")
    monkeypatch.setattr(config, "CONFIGS", str(tmp_path))
    with pytest.raises(ValueError):
        config.load_agent("broken")        # missing tier + display_name
    with pytest.raises(ValueError):
        config.load_agent("bad_family")    # family not in AGENT_FAMILIES


def test_dataset_key_selection():
    r2r = config.load_benchmark("r2r")
    assert config.agent_dataset_key(config.load_agent("etpnav"), "r2r") == "bert"
    assert config.agent_dataset_key(config.load_agent("etpr1"), "r2r") == "xlmr"
    # a VLA has no `dataset:` block -> the untokenised default
    assert config.agent_dataset_key(config.load_agent("navila"), "r2r") == "official"
    assert config.benchmark_dataset_root(r2r, "bert").endswith("BERTidx")
    with pytest.raises(ValueError):
        config.benchmark_dataset_root(r2r, "no_such_key")


def _describe(agent_id, benchmark="r2r"):
    from vlnsafe import cli

    class Args(object):
        agent = agent_id

    Args.benchmark = benchmark
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.cmd_describe(Args())
    return rc, json.loads(buf.getvalue())


@pytest.mark.parametrize("agent_id", ALL_AGENTS)
def test_describe_shape_and_exit_contract(agent_id):
    """describe emits the documented block for every agent, and its exit code is 0 iff
    every resolved path exists or the config is assets_pending."""
    rc, out = _describe(agent_id)
    assert set(out) >= {"agent", "display_name", "tier", "family", "sensor_rig",
                        "paths", "unavailable", "assets_pending", "missing_paths", "ok"}
    assert out["agent"] == agent_id
    assert "dataset" in out["paths"] and out["paths"]["dataset"]["key"]
    expect_ok = out["assets_pending"] or not out["missing_paths"]
    assert out["ok"] is expect_ok
    assert rc == (0 if expect_ok else 1)


@pytest.mark.skipif(not os.path.isdir(os.path.expandvars("$VLNSAFE_CHECKPOINTS/etpr1")),
                    reason="ETP-R1 assets not on this machine")
def test_describe_etpr1_all_paths_present_here():
    """Gate C, run for real when the data mount is available: every etpr1 path exists."""
    rc, out = _describe("etpr1")
    assert rc == 0 and out["ok"] and out["missing_paths"] == []
    for field in ("worker_python", "code_root", "checkpoint", "pretrained", "dataset"):
        assert out["paths"][field]["exists"] is True
