from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dlt.naming import ExperimentNamer
from dlt.yaml_logging import (
    YamlLogger,
    filter_env,
    latest_log,
    load_log,
    yaml_dump,
)


def test_log_roundtrip_and_latest(tmp_path: Path) -> None:
    log = YamlLogger("exp", logdir=tmp_path, save_every=0, log_env=False)
    log.log(step=0, loss=np.float32(1.5), note="a")
    log.log(step=1, loss=0.5)
    log.add_info("done")
    path = log.save()
    log.close()

    data = load_log(path)
    assert data["experiment"]["name"] == "exp"
    assert data["metrics"]["loss"] == [1.5, 0.5]
    assert data["latest"]["loss"] == 0.5
    assert data["config"] == {}
    assert data["info"][-1].endswith("done")
    assert "env" not in data


def test_config_and_env_filtering(tmp_path: Path) -> None:
    env = {
        "PATH": "/bin",
        "MY_API_KEY": "secret",
        "HF_TOKEN": "secret",
        "SECRET": "secret",
        "VISIBLE": "yes",
    }
    filtered = filter_env(env)
    assert filtered == {"PATH": "/bin", "VISIBLE": "yes"}
    assert filter_env(env, allowlist=["MY_API_KEY"]) == {}

    log = YamlLogger("e", logdir=tmp_path, save_every=0, log_env=True, config={"lr": 2})
    log.save()
    log.close()
    data = load_log(log.path)
    assert data["config"] == {"lr": 2}
    assert isinstance(data["env"], dict)
    assert all("TOKEN" not in k for k in data["env"])


def test_keep_last_trims_metric_history(tmp_path: Path) -> None:
    log = YamlLogger("e", logdir=tmp_path, save_every=0, keep_last=3)
    for i in range(10):
        log.log(loss=float(i))
    log.save()
    log.close()
    data = load_log(log.path)
    assert data["metrics"]["loss"] == [7.0, 8.0, 9.0]


def test_save_every_writes_periodically(tmp_path: Path) -> None:
    log = YamlLogger("e", logdir=tmp_path, save_every=2)
    log.log(loss=1.0)  # not yet
    assert log.snapshot()["experiment"]["save_count"] == 0
    log.log(loss=2.0)  # triggers save
    assert log.snapshot()["experiment"]["save_count"] == 1
    assert log.path.is_file()
    log.close()


def test_keyboard_interrupt_context_saves(tmp_path: Path) -> None:
    log = YamlLogger("k", logdir=tmp_path, save_every=0)
    with pytest.raises(KeyboardInterrupt):
        with log:
            log.log(loss=1.0)
            raise KeyboardInterrupt
    data = load_log(log.path)
    assert any("KeyboardInterrupt" in line for line in data["info"])
    assert data["latest"]["loss"] == 1.0


def test_exception_is_recorded_but_not_swallowed(tmp_path: Path) -> None:
    log = YamlLogger("e", logdir=tmp_path, save_every=0)
    with pytest.raises(ValueError):
        with log:
            raise ValueError("boom")
    data = load_log(log.path)
    assert any("ValueError" in line for line in data["info"])


def test_jsonl_sidecar(tmp_path: Path) -> None:
    log = YamlLogger("j", logdir=tmp_path, save_every=0, jsonl_metrics=True)
    for i in range(3):
        log.log(step=i, loss=float(i))
    log.save()
    log.close()
    lines = log.jsonl_path.read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["loss"] == 0.0


def test_yaml_dump_handles_arbitrary_objects() -> None:
    class Weird:
        def __repr__(self) -> str:
            return "<weird>"

    text = yaml_dump({"obj": Weird(), "path": Path("a/b"), "arr": np.array([1, 2])})
    assert "<weird>" in text and "a/b" in text and "- 1" in text


def test_run_integration_and_latest_log(tmp_path: Path) -> None:
    run = ExperimentNamer(tmp_path / "runs", name="integrated").new_run({"lr": 1e-4})
    log = YamlLogger("integrated", run=run, save_every=0)
    log.log(loss=0.1)
    log.close()
    assert log.path.parent == run.logs
    assert str(run.run_dir) in yaml_load_text(log.path)
    assert latest_log(run.logs) == log.path


def yaml_load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_latest_log_missing_dir(tmp_path: Path) -> None:
    assert latest_log(tmp_path / "nope") is None


def test_async_save_coalesces_and_flush_persists(tmp_path: Path) -> None:
    log = YamlLogger("async", logdir=tmp_path, save_every=1, async_save=True)
    for i in range(100):
        log.log(loss=float(i))
    log.flush()
    data = load_log(log.path)
    assert data["latest"]["loss"] == 99.0
    assert data["metrics"]["loss"][-1] == 99.0
    log.close()
    assert load_log(log.path)["metrics"]["loss"] == [float(i) for i in range(100)]


def test_async_context_manager_flushes_on_exit(tmp_path: Path) -> None:
    with YamlLogger("async", logdir=tmp_path, save_every=0, async_save=True) as log:
        log.log(loss=1.5)
        log.add_info("hi")
    assert log.path.is_file()
    data = load_log(log.path)
    assert data["latest"]["loss"] == 1.5
    assert any(line.endswith("hi") for line in data["info"])


def test_async_write_error_is_recorded(tmp_path: Path) -> None:
    target_dir = tmp_path / "a-directory"
    target_dir.mkdir()
    log = YamlLogger("async-err", logdir=tmp_path, save_every=0, async_save=True)
    log.log(loss=1.0)
    log.path = target_dir  # replacing a directory with a file must fail
    log.save()
    log.flush()
    assert log._write_error is not None
    log.close()
