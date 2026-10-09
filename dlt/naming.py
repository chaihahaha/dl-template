"""Automatic, collision-free naming of experiments, runs and output files.

Motivation
----------
Deep-learning runs are usually identified by three things at once:

* **when** it started (sortable, tells you which run is newer),
* **what** it is (a human-readable name / the config), and
* **a short unique id** (so two runs of the same thing never collide).

This module builds paths of the form ``runs/20261010T002915Z_resnet50_a1b2c3d4``
and gives every run standard sub-directories (``logs/``, ``ckpts/``,
``figures/``, ``results/``).  It depends only on the standard library, so it can
be imported from anywhere without heavy imports.

Nothing here ever guesses: ``unique_path`` checks the filesystem and appends a
counter (or a timestamp+id) until the name is free, so you cannot silently
overwrite a previous experiment.
"""

from __future__ import annotations

import os
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "utcnow",
    "timestamp",
    "slugify",
    "short_id",
    "unique_path",
    "auto_name",
    "RunPaths",
    "ExperimentNamer",
]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def utcnow() -> datetime:
    """Timezone-aware current UTC time (never use the naive ``datetime.now()``)."""
    return datetime.now(timezone.utc)


def timestamp(now: datetime | None = None, *, utc: bool = True) -> str:
    """Return a compact, lexicographically sortable timestamp.

    >>> len(timestamp()) 
    16
    """
    now = now or utcnow()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc) if utc else now
    if not utc:
        now = now.astimezone()
    return now.strftime("%Y%m%dT%H%M%SZ" if utc else "%Y%m%dT%H%M%S%z")


def short_id(length: int = 8) -> str:
    """Random hex id.  Uses :mod:`secrets` (not ``uuid1``, which leaks the MAC)."""
    if length <= 0:
        raise ValueError("length must be positive")
    return secrets.token_hex((length + 1) // 2)[:length]


def slugify(text: Any, *, max_length: int = 48, separator: str = "-") -> str:
    """Turn arbitrary text into a safe, readable path component.

    Unicode letters (e.g. Chinese) are preserved; whitespace and punctuation
    collapse into ``separator``.
    """
    text = str(text).strip()
    out: list[str] = []
    pending_sep = False
    for char in text:
        if char.isalnum() or char in "-_.":
            if pending_sep and out:
                out.append(separator)
            pending_sep = False
            out.append(char)
        else:
            pending_sep = True
    slug = "".join(out).strip(separator + ".")
    if not slug:
        slug = "run"
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip(separator + ".") or "run"
    return slug


def unique_path(
    path: str | os.PathLike[str],
    *,
    strategy: str = "counter",
    create: bool = False,
    max_tries: int = 100_000,
) -> Path:
    """Return a path that does not exist yet.

    Parameters
    ----------
    strategy:
        ``"counter"``    -> ``name_001.ext``, ``name_002.ext`` ...
        ``"timestamp"``  -> ``name_20261010T002915Z_ab12.ext``
    create:
        When true, reserve the name atomically with ``O_CREAT | O_EXCL`` so
        concurrent processes cannot race onto the same path.
    """
    path = Path(path)
    if strategy not in ("counter", "timestamp"):
        raise ValueError(f"unknown strategy: {strategy!r}")

    def candidate(i: int) -> Path:
        if i == 0 and strategy == "counter":
            return path
        if strategy == "counter":
            return path.with_name(f"{path.stem}_{i:03d}{path.suffix}")
        return path.with_name(f"{path.stem}_{timestamp()}_{short_id(4)}{path.suffix}")

    for i in range(max_tries):
        cand = candidate(i)
        if create:
            try:
                fd = os.open(cand, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
            except FileExistsError:
                continue
            os.close(fd)
            return cand
        if not cand.exists() and not os.path.lexists(cand):
            return cand
    raise FileExistsError(f"could not find a free name for {path} after {max_tries} tries")


def auto_name(
    prefix: str,
    *,
    ext: str = "",
    directory: str | os.PathLike[str] | None = None,
    with_timestamp: bool = True,
    create: bool = False,
) -> Path:
    """Build ``[directory/]<timestamp>_<slug-prefix>[_id]<ext>`` uniquely.

    This is the "one-liner" file-naming helper::

        p = auto_name("loss curve", ext=".png", directory="imgs")
        # imgs/20261010T002915Z_loss-curve_ab12.png
    """
    stem = slugify(prefix)
    if with_timestamp:
        stem = f"{timestamp()}_{stem}_{short_id(4)}"
    name = stem + (ext if ext.startswith(".") or not ext else "." + ext)
    base = Path(directory) if directory else Path(".")
    return unique_path(base / name, create=create)


# --------------------------------------------------------------------------- #
# run directories
# --------------------------------------------------------------------------- #
@dataclass
class RunPaths:
    """Paths belonging to a single experiment run."""

    root: Path
    run_dir: Path
    name: str
    run_id: str
    created_at: datetime
    subdirs: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.run_dir = Path(self.run_dir)

    # -- standard sub-directories ----------------------------------------- #
    def subdir(self, key: str, *, create: bool = True) -> Path:
        if key not in self.subdirs:
            raise KeyError(f"unknown sub-directory {key!r}; known: {sorted(self.subdirs)}")
        path = self.subdirs[key]
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def logs(self) -> Path:
        return self.subdir("logs")

    @property
    def ckpts(self) -> Path:
        return self.subdir("ckpts")

    @property
    def figures(self) -> Path:
        return self.subdir("figures")

    @property
    def results(self) -> Path:
        return self.subdir("results")

    # -- convenience ------------------------------------------------------ #
    def path(self, *parts: str | os.PathLike[str], create_parents: bool = False) -> Path:
        target = self.run_dir.joinpath(*[os.fspath(p) for p in parts])
        if create_parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def file(
        self,
        stem: str,
        *,
        ext: str = "",
        subdir: str | None = None,
        unique: bool = True,
        create: bool = False,
    ) -> Path:
        """A uniquely named file, optionally inside one of the standard sub-dirs."""
        base = self.subdir(subdir) if subdir else self.run_dir
        name = stem + (ext if ext.startswith(".") or not ext else "." + ext)
        target = base / name
        return unique_path(target, create=create) if unique else target

    def save_config(self, config: Mapping[str, Any] | Any, filename: str = "config.yaml") -> Path:
        """Persist the run configuration next to the run (YAML, JSON fallback)."""
        target = self.run_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        from ._atomic import atomic_write_text

        try:
            import yaml

            text = yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True)
        except Exception:  # pragma: no cover - yaml is a hard dep, but be safe
            import json

            text = json.dumps(config, indent=2, ensure_ascii=False, default=repr)
        atomic_write_text(target, text)
        return target

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return os.fspath(self.run_dir)


class ExperimentNamer:
    """Create run directories with sortable, collision-free names.

    >>> namer = ExperimentNamer("runs", name="mnist-baseline")
    >>> paths = namer.new_run({"lr": 1e-3})   # doctest: +SKIP
    >>> paths.run_dir.name                     # doctest: +SKIP
    '20261010T002915Z_mnist-baseline_ab12cd34'
    """

    SUBDIRS = ("logs", "ckpts", "figures", "results")

    def __init__(
        self,
        root: str | os.PathLike[str] = "runs",
        *,
        name: str | None = None,
        id_length: int = 8,
        subdirs: Iterable[str] = SUBDIRS,
        unique: bool = True,
    ) -> None:
        self.root = Path(root)
        self.name = name
        self.id_length = int(id_length)
        self.subdirs = tuple(subdirs)
        self.unique = unique

    def run_name(self, name: str | None = None, now: datetime | None = None) -> str:
        return f"{timestamp(now)}_{slugify(name or self.name or 'run')}_{short_id(self.id_length)}"

    def new_run(
        self,
        name: str | Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | Any | None = None,
        *,
        now: datetime | None = None,
        make: bool = True,
        save_config: bool = True,
        run_id: str | None = None,
    ) -> RunPaths:
        # Allow ``new_run(config)`` as a shorthand for ``new_run(config=config)``.
        if isinstance(name, Mapping) and config is None:
            name, config = None, name
        run_id = run_id or short_id(self.id_length)
        run_name = f"{timestamp(now)}_{slugify(name or self.name or 'run')}_{run_id}"
        run_dir = self.root / run_name
        if self.unique:
            run_dir = unique_path(run_dir)
        subdirs = {key: run_dir / key for key in self.subdirs}
        paths = RunPaths(
            root=self.root,
            run_dir=run_dir,
            name=slugify(name or self.name or "run"),
            run_id=run_id,
            created_at=now or utcnow(),
            subdirs=subdirs,
        )
        if make:
            run_dir.mkdir(parents=True, exist_ok=True)
            for path in subdirs.values():
                path.mkdir(parents=True, exist_ok=True)
        if save_config and config is not None:
            paths.save_config(dict(config))
        return paths

    def latest_run(self, *, must_exist: bool = True) -> Path | None:
        """Return the most recently created run directory (by name order)."""
        if not self.root.is_dir():
            if must_exist:
                raise FileNotFoundError(f"run root does not exist: {self.root}")
            return None
        runs = sorted((p for p in self.root.iterdir() if p.is_dir()), key=lambda p: p.name)
        return runs[-1] if runs else None
