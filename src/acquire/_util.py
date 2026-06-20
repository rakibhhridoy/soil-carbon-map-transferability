"""Shared helpers for MDBC acquisition scripts."""
import json, os, time, hashlib
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]          # .../BlueCarbon
MANIFESTS = ROOT / "manifests"
LOGS = ROOT / "logs"
MANIFESTS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)


def stream_download(url: str, dest: Path, chunk=1 << 20, retries=3) -> Path:
    """Download url -> dest with streaming, resume-safe overwrite, simple retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                got = 0
                with open(tmp, "wb") as f:
                    for c in r.iter_content(chunk):
                        f.write(c); got += len(c)
                if total and got < total * 0.99:
                    raise IOError(f"short read {got}/{total}")
            tmp.replace(dest)
            return dest
        except Exception as e:
            print(f"  [retry {attempt}/{retries}] {dest.name}: {e}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"failed: {url}")


def sha256(path: Path, limit=1 << 26) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:16]


def write_manifest(name: str, source: str, files: dict, extra: dict | None = None):
    out = {"source": source, "generated": time.time(), "files": files}
    if extra:
        out.update(extra)
    p = MANIFESTS / f"{name}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"  manifest -> {p}")
    return p


def record_file(path: Path, url: str = "") -> dict:
    st = path.stat()
    return {"bytes": st.st_size, "mtime": round(st.st_mtime, 2),
            "path": str(path.relative_to(ROOT)), "sha256_head": sha256(path), "url": url}
