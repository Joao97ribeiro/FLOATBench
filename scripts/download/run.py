"""Download the FLOATBench dataset into data/.

Fetches the dataset archive, checks its SHA-256 and unpacks it to
``data/{ref,opt1,opt2}/`` (``train_damage.csv``, ``test_damage.csv``,
``data.csv`` and ``metadata.json`` per tower). Standard library only,
apart from absl for the flags.

Run::

    python scripts/download/run.py --flagfile=scripts/download/config.cfg
"""

import hashlib
import pathlib
import shutil
import tempfile
import urllib.request
import zipfile

from absl import app, flags, logging

FLAGS = flags.FLAGS
flags.DEFINE_string("url", None, "Download link of the dataset archive.")
flags.DEFINE_string("sha256", None, "Expected SHA-256 of the archive.")
flags.DEFINE_string("data_dir", "data",
                    "Where the per-tower folders are written.")
flags.DEFINE_list("towers", "ref,opt1,opt2", "Tower folders to expect.")
flags.DEFINE_bool("force", False, "Download again even if the files exist.")

FILES = ("train_damage.csv", "test_damage.csv", "data.csv", "metadata.json")


def _download(url: str, path: pathlib.Path) -> str:
    """Streams ``url`` to ``path``.

    Args:
        url: Source URL.
        path: Destination file.

    Returns:
        SHA-256 hex digest of the downloaded bytes.
    """
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as resp, open(path, "wb") as out:
        while chunk := resp.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def _move_into(src: pathlib.Path, data_dir: pathlib.Path) -> None:
    """Moves every entry of ``src`` into ``data_dir``, replacing old ones.

    Args:
        src: Unpacked archive root.
        data_dir: Destination folder.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dst = data_dir / item.name
        if dst.is_dir():
            shutil.rmtree(dst)
        elif dst.exists():
            dst.unlink()
        shutil.move(str(item), dst)


def main(_):
    """Downloads, verifies and unpacks the dataset."""
    data_dir = pathlib.Path(FLAGS.data_dir)
    expected = [data_dir / t / f for t in FLAGS.towers for f in FILES]
    if all(p.exists() for p in expected) and not FLAGS.force:
        logging.info(
            "%s/ already holds the dataset (use --force to "
            "download again).", data_dir)
        return
    with tempfile.TemporaryDirectory() as tmp:
        archive = pathlib.Path(tmp) / "FLOATBench.zip"
        logging.info("Downloading the dataset archive ...")
        sha = _download(FLAGS.url, archive)
        if FLAGS.sha256 and sha != FLAGS.sha256:
            raise SystemExit(f"Checksum mismatch: {sha} "
                             f"(expected {FLAGS.sha256}).")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        _move_into(pathlib.Path(tmp) / "FLOATBench", data_dir)
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise SystemExit(f"Missing after unpacking: {missing}")
    logging.info("Dataset ready in %s/ (%d files, checksum verified).",
                 data_dir, len(expected))


if __name__ == "__main__":
    flags.mark_flag_as_required("url")
    app.run(main)
