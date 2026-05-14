#!/usr/bin/env python3
"""Build a Kaggle-compatible multi-file archive: main.py + orbit_submit/ (+ extras).

Default: copy ``submission_<version>.py`` → ``main.py`` and bundle ``orbit_submit/``.
Optional overrides: ``tools/orbit_submission_pack.yaml`` (versions.<slug>.entry|packages|files).
"""

from __future__ import annotations

import argparse
import compileall
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "tools" / "orbit_submission_pack.yaml"


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    txt = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(txt)
    except ImportError:
        print(f"WARN: PyYAML missing — ignoring manifest {path}", file=sys.stderr)
        return {}
    return data if isinstance(data, dict) else {}


def _version_block(manifest: dict, version: str) -> dict:
    vers = manifest.get("versions") or {}
    block = vers.get(version)
    return block if isinstance(block, dict) else {}


def _resolve_paths(
    version: str, manifest_path: Path
) -> tuple[Path, list[Path], list[tuple[Path, str]]]:
    """Returns (entry_py, package_dirs, extra_files) where extra_files are (abs, arcname)."""
    manifest = _load_yaml(manifest_path)
    block = _version_block(manifest, version)

    entry = block.get("entry")
    if entry:
        ep = Path(entry)
        entry_path = ep if ep.is_absolute() else ROOT / ep
    else:
        entry_path = ROOT / f"submission_{version}.py"
    if not entry_path.is_file():
        raise SystemExit(f"Entry missing: {entry_path}")

    packages = block.get("packages")
    pkgs: list[Path] = []
    if packages is None:
        pkgs = [ROOT / "orbit_submit"]
    else:
        if not isinstance(packages, list):
            raise SystemExit("manifest packages must be a list")
        pkgs = [ROOT / p for p in packages]

    extras: list[tuple[Path, str]] = []
    for item in block.get("files") or []:
        if isinstance(item, str):
            p = ROOT / item
            extras.append((p, Path(item).name))
        elif isinstance(item, dict) and "src" in item:
            src = ROOT / item["src"]
            dest = item.get("dest") or Path(item["src"]).name
            extras.append((src, str(dest)))
        else:
            raise SystemExit(f"Bad manifest files entry: {item!r}")

    return entry_path, pkgs, extras


def _inspect_tree(base: Path) -> None:
    for p in sorted(base.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(base)
        print(rel.as_posix())


def _compile_staging(base: Path) -> None:
    ok = compileall.compile_dir(str(base), quiet=1)
    if not ok:
        raise SystemExit("compileall failed — fix syntax errors above")
    for pycache in sorted(base.rglob("__pycache__"), reverse=True):
        shutil.rmtree(pycache, ignore_errors=True)


def _write_zip(base: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in base.rglob("*"):
            if p.is_dir():
                continue
            arc = p.relative_to(base).as_posix()
            z.write(p, arcname=arc)


def _write_tar_gz(base: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tf:
        for p in sorted(base.rglob("*")):
            if p.is_dir():
                continue
            arc = str(p.relative_to(base))
            tf.add(p, arcname=arc)


def _write_7z(base: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file())
    if not names:
        raise SystemExit("nothing to archive")
    for exe in ("7zz", "7z"):
        try:
            subprocess.run(
                [exe, "a", "-t7z", str(dest), *names],
                cwd=str(base),
                check=True,
                capture_output=True,
                text=True,
            )
            return
        except FileNotFoundError:
            continue
    raise SystemExit("7z / 7zz not found; install p7zip or choose zip / tar.gz")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack Orbit Wars submission (+ orbit_submit).")
    ap.add_argument("--version", required=True, help="Matches submission_<version>.py basename")
    ap.add_argument(
        "--format",
        choices=("zip", "tar.gz", "tgz", "7z"),
        default="zip",
        help="Archive format",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path or directory (default: dist/orbit_submit_<version>.<ext>)",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"YAML manifest (default {DEFAULT_MANIFEST})",
    )
    ap.add_argument("--inspect", action="store_true", help="Print staged file list and exit")
    ap.add_argument("--no-compile", action="store_true", help="Skip compileall on staging dir")
    args = ap.parse_args()

    ver = args.version
    fmt = "tar.gz" if args.format == "tgz" else args.format
    ext = "tar.gz" if fmt == "tar.gz" else fmt

    out_path = args.out
    if out_path is None:
        out_path = ROOT / "dist" / f"orbit_submit_{ver}.{ext}"
    elif out_path.is_dir() or str(out_path).endswith("/"):
        out_path = Path(out_path) / f"orbit_submit_{ver}.{ext}"

    entry, pkgs, extras = _resolve_paths(ver, Path(args.manifest))

    with tempfile.TemporaryDirectory(prefix="orbit_pack_") as td:
        base = Path(td)
        shutil.copy2(entry, base / "main.py")
        seen_pkgs = set()
        for d in pkgs:
            rp = d.resolve()
            if not rp.is_dir():
                raise SystemExit(f"Package dir missing: {rp}")
            if rp in seen_pkgs:
                continue
            seen_pkgs.add(rp)
            dest = base / rp.name
            shutil.copytree(
                rp,
                dest,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.py[cod]", "*.pyd"),
            )
        for abs_path, arc in extras:
            if not abs_path.is_file():
                raise SystemExit(f"Extra file missing: {abs_path}")
            tgt = base / arc
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(abs_path, tgt)

        if args.inspect:
            _inspect_tree(base)
            return

        if not args.no_compile:
            _compile_staging(base)

        if fmt == "zip":
            _write_zip(base, out_path.resolve())
        elif fmt == "tar.gz":
            _write_tar_gz(base, out_path.resolve())
        else:
            _write_7z(base, out_path.resolve())

    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
