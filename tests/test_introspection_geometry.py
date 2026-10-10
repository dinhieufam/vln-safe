"""attach_replay_geometry: joins the T1 replay sidecars without ever overwriting a logged
column or inventing values for episodes the replay could not reproduce."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from introspection import attach_replay_geometry  # noqa: E402


def _steps(values):
    return pd.DataFrame({
        "episode_id": ["1", "1", "2"], "step_index": [0, 1, 0],
        "predicted_clearance_m": values})


def test_fills_an_empty_column_and_leaves_unreplayed_rows_nan(tmp_path):
    pd.DataFrame({"episode_id": ["1", "1"], "step_index": [0, 1],
                  "predicted_clearance_m": [0.4, "UNAVAILABLE"]}).to_csv(
        tmp_path / "s0.csv", index=False)   # episode 2 failed replay: absent
    df, n = attach_replay_geometry(_steps([np.nan] * 3), [str(tmp_path / "s0.csv")])
    assert n == 1
    assert df["predicted_clearance_m"].iloc[0] == 0.4
    assert df["predicted_clearance_m"].iloc[1:].isna().all()


def test_never_overwrites_a_logged_column(tmp_path):
    pd.DataFrame({"episode_id": ["1"], "step_index": [0],
                  "predicted_clearance_m": [9.9]}).to_csv(tmp_path / "s0.csv", index=False)
    df, n = attach_replay_geometry(_steps([0.1, 0.2, 0.3]), [str(tmp_path / "s0.csv")])
    assert n == 0 and df["predicted_clearance_m"].tolist() == [0.1, 0.2, 0.3]


def test_no_sidecars_is_a_no_op():
    df, n = attach_replay_geometry(_steps([np.nan] * 3), [])
    assert n == 0 and df["predicted_clearance_m"].isna().all()


def _walk(actions, disps, coll, eid="1"):
    return pd.DataFrame({"episode_id": eid, "step_index": range(len(actions)),
                         "executed_action": actions, "sim_displacement_m": disps,
                         "engine_collision_flag": coll})


def test_stall_history_flags_only_a_blocked_pose():
    from introspection import add_stall_history
    # fwd stalls at pose A; turn away and back (heading returns); fwd from A is flagged.
    # A clean fwd moves to pose B: a fwd from B is not flagged.
    df = _walk(["move_forward", "turn_left", "turn_right", "move_forward",
                 "turn_left", "move_forward", "move_forward"],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25],
                [1, 0, 0, 1, 0, 0, 0])
    out = add_stall_history(df, "r2r")
    # the signal marks the pose, whatever the next action: the turns taken while standing
    # at blocked pose A (steps 1 and 4) are flagged too; score_stall_forward keeps forwards
    assert out["stall_history"].tolist() == [0, 1, 0, 1, 1, 0, 0]


def test_stall_history_full_turn_returns_to_the_blocked_heading():
    from introspection import add_stall_history
    n = 24  # R2R: 15 degree turns
    df = _walk(["move_forward"] + ["turn_left"] * n + ["move_forward"],
                [0.0] + [0.0] * n + [0.0], [1] + [0] * n + [1])
    assert add_stall_history(df, "r2r")["stall_history"].iloc[-1] == 1
    # RxR turns 30 degrees: 24 turns is two revolutions, still the same heading
    assert add_stall_history(df, "rxr")["stall_history"].iloc[-1] == 1
    # 12 R2R turns face the other way: not blocked
    df12 = _walk(["move_forward"] + ["turn_left"] * 12 + ["move_forward"],
                  [0.0] * 14, [1] + [0] * 13)
    assert add_stall_history(df12, "r2r")["stall_history"].iloc[-1] == 0


def test_stall_history_resets_per_episode():
    from introspection import add_stall_history
    df = pd.concat([_walk(["move_forward"], [0.0], [1], "1"),
                    _walk(["move_forward"], [0.0], [1], "2")], ignore_index=True)
    assert add_stall_history(df, "r2r")["stall_history"].tolist() == [0, 0]
