"""Download the official CIFAR-10 binary archive, verify it, and decode it."""
import hashlib
import concurrent.futures
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URL = "https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz"
MD5 = "c32a1d4ab5d03f1284b67883e8d87530"


def main():
    DATA.mkdir(exist_ok=True)
    archive = DATA / "cifar-10-binary.tar.gz"
    if not archive.exists():
        print("Downloading", URL, flush=True)
        temporary = archive.with_suffix(".download")
        prefix = temporary.stat().st_size if temporary.exists() else 0
        probe = urllib.request.Request(URL, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(probe, timeout=30) as response:
            assert response.status == 206
            total = int(response.headers["Content-Range"].split("/")[-1])
        parts = []
        chunk_size = max(1, (total - prefix + 7) // 8)
        for start in range(prefix, total, chunk_size):
            end = min(total, start + chunk_size) - 1
            parts.append((start, end, DATA / f"cifar-part-{start}-{end}"))
        def fetch(part):
            start, end, path = part
            if path.exists() and path.stat().st_size == end - start + 1:
                return
            request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as stream:
                assert response.status == 206
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            assert path.stat().st_size == end - start + 1
            print("Downloaded range", start, end, flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(fetch, parts))
        assembled = archive.with_suffix(".assembled")
        with assembled.open("wb") as stream:
            if prefix:
                with temporary.open("rb") as old:
                    shutil.copyfileobj(old, stream)
            for _, _, path in parts:
                with path.open("rb") as chunk:
                    shutil.copyfileobj(chunk, stream)
        assert assembled.stat().st_size == total
        if hashlib.md5(assembled.read_bytes()).hexdigest() != MD5:
            raise RuntimeError("Assembled CIFAR archive checksum mismatch")
        assembled.rename(archive)
        for _, _, path in parts:
            path.unlink()
        if temporary.exists():
            temporary.unlink()
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != MD5:
        raise RuntimeError(f"CIFAR archive checksum mismatch: {digest}")
    with tarfile.open(archive, "r:gz") as tar:
        def batch(name):
            stream = tar.extractfile("cifar-10-batches-bin/" + name)
            assert stream is not None
            records = np.frombuffer(stream.read(), dtype=np.uint8).reshape(-1, 3073)
            return records[:, 1:].reshape(-1, 3, 32, 32).copy(), records[:, 0].astype(np.int64)
        batches = [batch(f"data_batch_{i}.bin") for i in range(1, 6)]
        train_x = np.concatenate([x for x, _ in batches])
        train_y = np.concatenate([y for _, y in batches])
        test_x, test_y = batch("test_batch.bin")
    for name, values in (("train_images", train_x), ("train_labels", train_y),
                         ("test_images", test_x), ("test_labels", test_y)):
        np.save(DATA / f"{name}.npy", values)
    print("Verified CIFAR-10:", train_x.shape, test_x.shape, "md5", digest, flush=True)


if __name__ == "__main__":
    main()
