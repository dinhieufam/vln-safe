"""Worker JSON-lines protocol validators. No habitat."""
import pytest

from vlnsafe.workers import protocol


def test_shutdown_request_is_valid():
    protocol.validate_request({"cmd": "shutdown"})


def test_decide_request_shape():
    ok = {
        "cmd": "decide",
        "agent_position": [0.0, 0.0, 0.0],
        "agent_rotation": [0.0, 0.0, 0.0, 1.0],
        "observation": {},
    }
    protocol.validate_request(ok)
    with pytest.raises(ValueError):
        protocol.validate_request({"cmd": "decide", "agent_position": [0, 0, 0]})
    with pytest.raises(ValueError):
        bad = dict(ok, agent_rotation=[0, 0, 1])
        protocol.validate_request(bad)
    with pytest.raises(ValueError):
        protocol.validate_request({"cmd": "wat"})


def test_response_validation():
    protocol.validate_response({"ok": True, "primitive_actions": [1, 2, 0]})
    with pytest.raises(ValueError):
        protocol.validate_response({"ok": False, "error": "boom"})
    with pytest.raises(ValueError):
        protocol.validate_response({"ok": True})
