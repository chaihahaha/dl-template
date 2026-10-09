"""A minimalist, crash-safe YAML experiment logger.

Why YAML?
---------
A training log is most useful when a human can ``grep`` it while the run is
still going: the current loss, the exact hyper-parameters, the code that
produced the run, environment variables, git commit, ... all in one
human-readable file.  YAML gives you that for the price of one dependency
(``PyYAML``), and it round-trips to plain Python objects.

What this module fixes compared with the original ``yaml_logging.py``
--------------------------------------------------------------------
* The original was a *snippet* with undefined names (``name``, ``config``,
  ``N``, ``dataset``, ``model``); it could not be imported.  This is a real
  class with an explicit API.
* ``logger['env_vars'] = os.environ`` dumped **every** environment variable,
  including API keys and tokens, into a file.  Environment capture is now
  opt-in and filtered.
* Writing the log with ``open(path, "w")`` on every save meant that (a) a
  crash mid-write could corrupt the log and (b) with ``never_overwrite``
  active, *every single save created a backup copy*.  Saves are now atomic
  (temp file + :func:`os.replace`) and never trigger the backup logic.
* ``pdb.set_trace()`` ran unconditionally in ``finally`` and would hang any
  non-interactive/CI run.  Debugging hooks are opt-in.
* NumPy scalars/arrays and torch tensors are serialised properly instead of
  raising ``RepresenterError``.
* Data is saved on ``KeyboardInterrupt``, on uncaught exceptions and at
  interpreter exit, so ``Ctrl+C`` never loses the last few hundred steps.
* A ``latest:`` section always shows the most recent value of each metric,
  which is what you actually want to grep for.
* Every log file gets a sortable, collision-free name and can live in a run
  directory created by :mod:`dlt.naming`.

Only hard dependency: ``PyYAML``.  ``tqdm`` is used if available.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ._atomic import atomic_write_text
from .naming import RunPaths, slugify, timestamp, utcnow

__all__ = [
    "YamlLogger",
    "load_log",
    "load_logs",
    "latest_log",
    "install_excepthook",
    "yaml_dump",
    "yaml_load",
]

_LOG = logging.getLogger("dlt.yaml_logging")

#: substrings that make an environment variable look sensitive
SENSITIVE_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "APIKEY",
    "API_KEY",
    "CREDENTIAL",
    "PRIVATE",
    "AUTH",
    "SESSION",
    "ACCESS_KEY",
    "COOKIE",
)


# --------------------------------------------------------------------------- #
# YAML plumbing: a dumper that understands scientific Python objects
# --------------------------------------------------------------------------- #
def _build_dumper():
    """Return a Dumper class that never crashes on numpy/torch/pathlib objects."""
    try:
        import yaml

        base = getattr(yaml, "CSafeDumper", None) or yaml.SafeDumper
    except Exception:  # pragma: no cover - yaml is a hard dependency
        raise

    class _Dumper(base):  # type: ignore[misc, valid-type]
        pass

    def _represent_path(dumper, data):
        return dumper.represent_str(os.fspath(data))

    def _represent_enum(dumper, data):
        return dumper.represent_data(data.value)

    def _represent_repr(dumper, data):
        try:
            import dataclasses

            if dataclasses.is_dataclass(data) and not isinstance(data, type):
                return dumper.represent_dict(dataclasses.asdict(data).items())
        except Exception:
            pass
        return dumper.represent_str(repr(data))

    _Dumper.add_multi_representer(object, _represent_repr)
    try:
        from pathlib import PurePath

        _Dumper.add_multi_representer(PurePath, _represent_path)
    except Exception:  # pragma: no cover
        pass
    try:
        import enum

        _Dumper.add_multi_representer(enum.Enum, _represent_enum)
    except Exception:  # pragma: no cover
        pass
    try:
        import numpy as np

        _Dumper.add_multi_representer(np.ndarray, lambda d, o: d.represent_data(o.tolist()))
        _Dumper.add_multi_representer(np.generic, lambda d, o: d.represent_data(o.item()))
    except Exception:  # pragma: no cover - numpy optional
        pass
    try:
        import torch

        def _represent_tensor(dumper, data):
            try:
                return dumper.represent_data(
                    data.detach().cpu().reshape(-1).tolist()
                    if data.numel() != 0
                    else []
                )
            except Exception:
                return _represent_repr(dumper, data)

        _Dumper.add_multi_representer(torch.Tensor, _represent_tensor)
    except Exception:  # pragma: no cover - torch optional
        pass
    return _Dumper


_DUMPER: Any = None


def _dumper():
    global _DUMPER
    if _DUMPER is None:
        _DUMPER = _build_dumper()
    return _DUMPER


def yaml_dump(obj: Any) -> str:
    """``yaml.safe_dump`` that tolerates NumPy/torch/pathlib/dataclasses."""
    import yaml

    return yaml.dump(
        obj,
        Dumper=_dumper(),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=1000,
    )


def yaml_load(text: str) -> Any:
    import yaml

    return yaml.safe_load(text)


# --------------------------------------------------------------------------- #
# metadata collectors
# --------------------------------------------------------------------------- #
def describe_system() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
    }
    for var in ("USER", "USERNAME", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV"):
        value = os.environ.get(var)
        if value:
            info[var.lower()] = value
    return info


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    from importlib import metadata

    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except Exception:
            out[name] = None
    return out


def _git_info(cwd: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    import subprocess

    def run(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return None
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "short_commit": commit[:8],
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "remote": run("config", "--get", "remote.origin.url"),
        "dirty": bool(status),
    }


def filter_env(
    environ: Mapping[str, str] | None = None,
    *,
    allowlist: Iterable[str] | None = None,
    denylist: Iterable[str] | None = None,
) -> dict[str, str]:
    """Return a copy of ``environ`` with credential-looking variables removed."""
    environ = os.environ if environ is None else environ
    deny = {k.upper() for k in (denylist or ())}
    out: dict[str, str] = {}
    for key, value in environ.items():
        if key.upper() in deny or any(marker in key.upper() for marker in SENSITIVE_MARKERS):
            continue
        if allowlist is not None and key not in allowlist:
            continue
        out[key] = value
    return out


def _capture_source() -> dict[str, Any] | None:
    """Best-effort capture of the *user's* training script."""
    candidates: list[Path] = []
    main = sys.modules.get("__main__")
    main_file = getattr(main, "__file__", None)
    if main_file:
        candidates.append(Path(main_file))
    if sys.argv and sys.argv[0] and not sys.argv[0].startswith("-"):
        candidates.append(Path(sys.argv[0]))
    for candidate in candidates:
        try:
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8", errors="replace")
                if len(text) > 4_000_000:
                    text = text[:4_000_000] + "\n# ... truncated ...\n"
                return {
                    "path": str(candidate.resolve()),
                    "sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest(),
                    "source": text,
                }
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- #
# the logger
# --------------------------------------------------------------------------- #
class YamlLogger:
    """Human-readable, crash-safe YAML log for a single experiment run.

    Parameters
    ----------
    name:
        Human-readable experiment name (used in the file name).
    config:
        Hyper-parameters / configuration mapping.
    logdir:
        Directory for the log file (created on demand).  Default ``./logs``.
    path:
        Explicit log path; overrides ``logdir``.
    run:
        A :class:`dlt.naming.RunPaths`; if given, log into ``run.logs`` and
        attach the run metadata.
    save_every:
        Auto-save every N calls to :meth:`log` (0 disables periodic saving).
    log_code:
        Embed the source of the main training script.
    log_git:
        Embed git commit/branch/dirty state (runs ``git`` with a timeout).
    log_env:
        Embed environment variables; sensitive ones are always filtered.
    jsonl_metrics:
        Additionally append one JSON object per :meth:`log` call to
        ``<logfile>.jsonl``.  Useful for very long runs where rewriting the
        whole YAML each time becomes expensive.
    keep_last:
        Keep only the last N values per metric in the YAML (memory friendly).
    """

    def __init__(
        self,
        name: str = "experiment",
        *,
        config: Mapping[str, Any] | None = None,
        logdir: str | os.PathLike[str] | None = None,
        path: str | os.PathLike[str] | None = None,
        run: RunPaths | None = None,
        save_every: int = 100,
        log_code: bool = True,
        log_git: bool = False,
        log_env: bool = False,
        env_allowlist: Iterable[str] | None = None,
        env_denylist: Iterable[str] | None = None,
        log_packages: Iterable[str] | None = ("numpy", "torch", "ray", "gymnasium"),
        jsonl_metrics: bool = False,
        keep_last: int | None = None,
        fsync: bool = False,
        flush_stdout: bool = True,
        autosave_atexit: bool = True,
    ) -> None:
        self.name = slugify(name)
        self.save_every = int(save_every)
        self.fsync = bool(fsync)
        self.keep_last = keep_last
        self.run = run

        self.experiment_id = uuid.uuid4().hex[:8]
        self.created_at = utcnow()

        if path is not None:
            self.path = Path(path)
        else:
            directory = Path(logdir) if logdir is not None else (
                run.logs if run is not None else Path("logs")
            )
            self.path = directory / (
                f"{timestamp(self.created_at)}_{self.name}_{self.experiment_id}.yaml"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_metrics = bool(jsonl_metrics)
        self.jsonl_path = self.path.with_suffix(self.path.suffix + ".jsonl")
        self._jsonl_handle = None

        # --- collected state ------------------------------------------------
        self.config: dict[str, Any] = dict(config) if config else {}
        self.metrics: dict[str, list[float]] = {}
        self.latest: dict[str, Any] = {}
        self.info: list[str] = []
        self.events: list[dict[str, Any]] = []
        self.artifacts: list[str] = []
        self.extra: dict[str, Any] = {}
        self.step: int | None = None

        self._system = describe_system()
        self._packages = _package_versions(log_packages) if log_packages else {}
        self._git = _git_info() if log_git else None
        self._env = (
            filter_env(allowlist=env_allowlist, denylist=env_denylist) if log_env else None
        )
        self._code = _capture_source() if log_code else None

        self._save_count = 0
        self._log_count = 0
        self._closed = False
        self.flush_stdout = flush_stdout

        if autosave_atexit:
            atexit.register(self.save)
        _LOG.debug("experiment %s logging to %s", self.experiment_id, self.path)

    # -- metadata -------------------------------------------------------- #
    @property
    def run_dir(self) -> Path | None:
        return self.run.run_dir if self.run is not None else None

    def set_config(self, config: Mapping[str, Any]) -> None:
        """Replace/merge the configuration mapping."""
        self.config.update(dict(config))

    def update(self, mapping: Mapping[str, Any]) -> None:
        """Merge arbitrary top-level keys into the log (advanced)."""
        self.extra.update(dict(mapping))

    def add_info(self, message: str) -> None:
        """Append a free-form human-readable line to the ``info`` list."""
        self.info.append(f"[{utcnow().isoformat()}] {message}")

    # alias
    info_msg = add_info

    def add_event(self, name: str, **fields: Any) -> None:
        self.events.append({"time": utcnow().isoformat(), "name": name, **fields})

    def add_artifact(self, path: str | os.PathLike[str]) -> None:
        entry = os.fspath(path)
        if entry not in self.artifacts:
            self.artifacts.append(entry)

    # -- metrics --------------------------------------------------------- #
    def log(self, metrics: Mapping[str, Any] | None = None, /, **kwargs: Any) -> "YamlLogger":
        """Record one step of metrics and maybe auto-save.

        ``logger.log(loss=0.1, acc=0.9)`` and ``logger.log({"loss": 0.1})`` are
        both accepted.  Values are stored as numbers when possible; anything
        else is stringified so the YAML stays valid.
        """
        record: dict[str, Any] = {}
        if metrics:
            record.update(metrics)
        record.update(kwargs)
        if not record:
            return self
        for key, value in record.items():
            key = str(key)
            scalar = self._coerce(value)
            self.metrics.setdefault(key, []).append(scalar)
            if self.keep_last is not None and len(self.metrics[key]) > self.keep_last:
                del self.metrics[key][: len(self.metrics[key]) - self.keep_last]
            self.latest[key] = scalar
        if "step" in record:
            try:
                self.step = int(record["step"])
            except (TypeError, ValueError):
                self.step = record["step"]  # type: ignore[assignment]
        self._log_count += 1
        if self.jsonl_metrics:
            self._append_jsonl(record)
        if self.save_every and self._log_count % self.save_every == 0:
            self.save()
        return self

    @staticmethod
    def _coerce(value: Any) -> Any:
        """Make a metric value YAML/JSON friendly (unwrap 0-d tensors/arrays)."""
        if isinstance(value, (bool, int, float, str)) or value is None:
            return value
        try:
            import numpy as np

            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray) and value.ndim == 0:
                return value.item()
        except Exception:
            pass
        try:
            import torch

            if isinstance(value, torch.Tensor) and value.numel() == 1:
                return value.detach().cpu().item()
        except Exception:
            pass
        return value

    def _append_jsonl(self, record: Mapping[str, Any]) -> None:
        try:
            payload = dict(record)
            if self.step is not None:
                payload.setdefault("step", self.step)
            if self._jsonl_handle is None:
                self._jsonl_handle = open(self.jsonl_path, "a", encoding="utf-8")
            self._jsonl_handle.write(json.dumps(payload, ensure_ascii=False, default=repr) + "\n")
            self._jsonl_handle.flush()
        except Exception as exc:  # never let logging kill training
            _LOG.warning("could not write jsonl metrics: %s", exc)

    # -- serialisation --------------------------------------------------- #
    def snapshot(self) -> dict[str, Any]:
        """Return the full log as a plain dict."""
        data: dict[str, Any] = {
            "experiment": {
                "id": self.experiment_id,
                "name": self.name,
                "created_at": self.created_at.isoformat(),
                "updated_at": utcnow().isoformat(),
                "save_count": self._save_count,
                "log_count": self._log_count,
                "path": str(self.path),
            },
            "system": self._system,
        }
        if self._packages:
            data["packages"] = self._packages
        if self.run is not None:
            data["run_dir"] = str(self.run.run_dir)
        if self._git:
            data["git"] = self._git
        if self._code:
            data["code"] = self._code
        if self._env is not None:
            data["env"] = self._env
        data["config"] = dict(self.config)
        data["step"] = self.step
        data["latest"] = dict(self.latest)
        data["metrics"] = {k: list(v) for k, v in self.metrics.items()}
        data["info"] = list(self.info)
        data["events"] = list(self.events)
        if self.artifacts:
            data["artifacts"] = list(self.artifacts)
        data.update(self.extra)
        return data

    def save(self, *, raise_on_error: bool = False) -> Path:
        """Atomically write the whole log to :attr:`path`."""
        if self._closed:
            return self.path
        try:
            text = yaml_dump(self.snapshot())
            atomic_write_text(self.path, text, fsync=self.fsync)
            self._save_count += 1
        except Exception as exc:
            _LOG.error("failed to save log to %s: %s", self.path, exc)
            if raise_on_error:
                raise
            return self.path
        if self.flush_stdout:
            try:
                sys.stdout.flush()
            except Exception:
                pass
        return self.path

    def close(self) -> Path:
        """Final save and detach the ``atexit`` hook."""
        path = self.save()
        if self._jsonl_handle is not None:
            try:
                self._jsonl_handle.close()
            except Exception:
                pass
            self._jsonl_handle = None
        self._closed = True
        with contextlib.suppress(Exception):
            atexit.unregister(self.save)
        return path

    # -- context manager / error handling -------------------------------- #
    def __enter__(self) -> "YamlLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is KeyboardInterrupt:
            self.add_info("KeyboardInterrupt: saving before exit")
        elif exc is not None:
            self.add_info("".join(traceback.format_exception(exc_type, exc, tb)))
        self.save()
        return False  # never swallow exceptions

    @contextlib.contextmanager
    def catch(self, *, reraise: bool = True) -> Iterator["YamlLogger"]:
        """Context manager that saves on interrupt/exception and re-raises."""
        try:
            yield self
        except KeyboardInterrupt:
            self.add_info("KeyboardInterrupt")
            self.save()
            if reraise:
                raise
        except BaseException as exc:  # noqa: BLE001 - logging should see everything
            self.add_info("exception: " + "".join(traceback.format_exception_only(type(exc), exc)))
            self.save()
            if reraise:
                raise

    def track(
        self,
        iterable: Iterable[Any],
        *,
        total: int | None = None,
        desc: str | None = None,
    ) -> Iterator[Any]:
        """Wrap an iterable with optional tqdm progress and periodic logging."""
        if total is None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None
        try:
            from tqdm import tqdm

            iterator: Iterable[Any] = tqdm(iterable, total=total, desc=desc)
        except Exception:
            iterator = iterable
        yield from iterator

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"YamlLogger(name={self.name!r}, path={str(self.path)!r})"


# --------------------------------------------------------------------------- #
# reading logs back
# --------------------------------------------------------------------------- #
def load_log(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML log produced by :class:`YamlLogger`."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path!r} does not contain a YAML mapping")
    return data


def load_logs(paths: Iterable[str | os.PathLike[str]]) -> list[dict[str, Any]]:
    return [load_log(p) for p in paths]


def latest_log(directory: str | os.PathLike[str] = "logs") -> Path | None:
    """Return the most recent ``*.yaml`` log in ``directory`` (or ``None``)."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    candidates = sorted(p for p in directory.glob("*.yaml") if p.is_file())
    return candidates[-1] if candidates else None


def install_excepthook(
    logger: YamlLogger,
    *,
    interactive: bool | None = None,
) -> None:
    """Save ``logger`` and print a traceback when an exception escapes ``main``.

    An interactive ``pdb`` prompt is only opened when explicitly requested (or
    when ``interactive=None`` and ``DLT_DEBUG=1`` **and** stdin is a TTY), so
    batch/CI runs never hang waiting for input.
    """
    if interactive is None:
        interactive = (
            sys.stdin.isatty() and os.environ.get("DLT_DEBUG", "0").lower() in ("1", "true", "yes")
        )

    def hook(exc_type, exc, tb):
        try:
            sys.__excepthook__(exc_type, exc, tb)
        except Exception:
            pass
        logger.add_info(
            "uncaught: " + "".join(traceback.format_exception_only(exc_type, exc)).strip()
        )
        logger.save()
        if interactive:
            import pdb

            pdb.post_mortem(tb)

    sys.excepthook = hook


# --------------------------------------------------------------------------- #
# demo / smoke test
# --------------------------------------------------------------------------- #
def _demo(tmp_root: Path | None = None) -> Path:
    """Tiny runnable example; also used by the test-suite."""
    import math

    root = Path(tmp_root) if tmp_root else Path("runs")
    from .naming import ExperimentNamer

    run = ExperimentNamer(root, name="demo").new_run({"lr": 1e-3, "epochs": 3})
    with YamlLogger("demo", config={"lr": 1e-3}, run=run, save_every=5) as logger:
        for step in range(20):
            logger.log(step=step, loss=math.exp(-step / 5), acc=step / 20)
        logger.add_info("finished demo")
        return logger.path


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print("wrote", _demo())
