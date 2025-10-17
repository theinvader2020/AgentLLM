"""Utilities for resilient file transfers.

This module exposes a small command line interface that can mirror a source
directory (or a single file) to a destination while handling intermittent
disconnects.  It keeps track of partially copied files, resumes them whenever
possible, and performs integrity checks so corrupted copies are detected
immediately.

Example usage::

    python scripts/robust_transfer.py /path/to/source \
        \\This PC\\Lenovo Yoga Tab 11\\Internal shared storage\\Download\\BiglyBT

The script only depends on the Python standard library and works on Windows,
macOS, and Linux.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


BUFFER_SIZE = 64 * 1024 * 1024  # 64 MiB per read – large enough for throughput


@dataclass
class TransferResult:
    """Summarises the outcome of a single copy operation."""

    source: Path
    destination: Path
    skipped: bool = False
    verified: bool = False
    error: Optional[str] = None


def iter_source_files(source: Path) -> Iterator[Path]:
    """Yield all files contained in *source*.

    If ``source`` is a directory, the iterator walks the tree depth first. When
    it is a file, the iterator yields the file itself.
    """

    if source.is_dir():
        for path in source.rglob("*"):
            if path.is_file():
                yield path
    else:
        yield source


def compute_hash(path: Path, chunk_size: int = BUFFER_SIZE) -> str:
    """Compute a SHA256 digest for *path* in a memory efficient way."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def copy_file_with_resume(
    source: Path,
    destination: Path,
    chunk_size: int = BUFFER_SIZE,
    verify: bool = True,
) -> TransferResult:
    """Copy *source* into *destination*, resuming partial transfers when possible.

    The copy is performed in ``chunk_size`` byte slices. If a previous transfer
    created a partial destination file, the function resumes copying from the
    last written offset instead of starting from scratch. When ``verify`` is
    ``True`` a post-transfer SHA256 check is executed.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        src_size = source.stat().st_size
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        return TransferResult(source, destination, error=str(exc))

    transferred = 0
    if destination.exists():
        dst_size = destination.stat().st_size
        if dst_size > src_size:
            # Destination is larger than the source, start from scratch.
            destination.unlink()
        else:
            transferred = dst_size

    mode = "r+b" if destination.exists() else "wb"
    with source.open("rb") as src_handle, destination.open(mode) as dst_handle:
        if transferred:
            src_handle.seek(transferred)
            dst_handle.seek(transferred)

        while transferred < src_size:
            chunk = src_handle.read(chunk_size)
            if not chunk:
                break
            dst_handle.write(chunk)
            transferred += len(chunk)

    result = TransferResult(source, destination)

    if verify:
        src_hash = compute_hash(source, chunk_size)
        dst_hash = compute_hash(destination, chunk_size)
        result.verified = src_hash == dst_hash
        if not result.verified:
            result.error = (
                "Hash mismatch after transfer – destination copy may be corrupted."
            )

    return result


def should_skip(source: Path, destination: Path, verify: bool) -> bool:
    """Return ``True`` when *destination* already matches *source*."""

    if not destination.exists():
        return False

    if source.stat().st_size != destination.stat().st_size:
        return False

    if not verify:
        return True

    return compute_hash(source) == compute_hash(destination)


def copy_tree(
    source: Path,
    destination: Path,
    *,
    chunk_size: int = BUFFER_SIZE,
    verify: bool = True,
    dry_run: bool = False,
) -> Iterable[TransferResult]:
    """Copy ``source`` to ``destination`` while yielding :class:`TransferResult`.

    ``destination`` is treated as a directory. When ``dry_run`` is ``True`` the
    function only reports what would be copied without touching the file system.
    """

    base = source if source.is_dir() else source.parent

    for path in iter_source_files(source):
        relative = path.relative_to(base)
        dst_path = destination / relative

        if dry_run:
            yield TransferResult(path, dst_path, skipped=True, verified=False)
            continue

        if should_skip(path, dst_path, verify):
            yield TransferResult(path, dst_path, skipped=True, verified=verify)
            continue

        yield copy_file_with_resume(path, dst_path, chunk_size=chunk_size, verify=verify)


def configure_logging(verbose: bool, log_file: Optional[Path]) -> None:
    """Configure the root logger according to the CLI flags."""

    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resiliently mirror one path into another with resume support.",
    )
    parser.add_argument("source", type=Path, help="File or directory to copy.")
    parser.add_argument("destination", type=Path, help="Target directory.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=BUFFER_SIZE,
        metavar="BYTES",
        help="Amount of data to read per iteration (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Do not perform post-transfer hash verification.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned work without copying anything.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional file that will receive a persistent log.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose, args.log_file)

    verify = not args.skip_verification
    destination = args.destination
    if destination.is_file():
        logging.error("Destination must be a directory, not an existing file.")
        return os.EX_USAGE

    try:
        results = list(
            copy_tree(
                args.source,
                destination,
                chunk_size=args.chunk_size,
                verify=verify,
                dry_run=args.dry_run,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.exception("Transfer aborted: %s", exc)
        return os.EX_SOFTWARE

    skipped = sum(1 for result in results if result.skipped)
    errors = [result for result in results if result.error]

    for result in results:
        if result.skipped:
            logging.info("Skipped %s", result.source)
        elif result.error:
            logging.error("Failed %s -> %s: %s", result.source, result.destination, result.error)
        else:
            status = "verified" if result.verified else "copied"
            logging.info("Transferred %s -> %s (%s)", result.source, result.destination, status)

    if errors:
        logging.error("Completed with %d errors", len(errors))
        return os.EX_DATAERR

    logging.info(
        "Completed successfully (%d files, %d skipped)",
        len(results),
        skipped,
    )
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())

