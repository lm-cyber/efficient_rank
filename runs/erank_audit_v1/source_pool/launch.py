"""Launch one experiment phase on the UUID belonging to physical NVIDIA GPU 0."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("smoke", "base", "pool"))
    args = parser.parse_args()
    config = json.loads((ROOT / "experiments/config.json").read_text())
    run = ROOT / "runs" / config["protocol"]
    run.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "data/train_images.npy").exists():
        raise RuntimeError("Finish experiments/download_data.py before launching GPU experiments")
    if args.phase == "pool" and not (run / "base_complete.json").exists():
        raise RuntimeError("Finish the base phase before launching the experimental pool")
    gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    marker = run / "process.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        proc = Path(f"/proc/{previous['pid']}/cmdline")
        if proc.exists() and str(ROOT / "experiments").encode() in proc.read_bytes():
            raise RuntimeError(f"Experiment phase still running: {previous}")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu_uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    script = "pipeline.py" if args.phase == "pool" else "base.py"
    command = [sys.executable, "-u", str(ROOT / "experiments" / script)]
    if args.phase == "smoke":
        command.append("--smoke")
    logfile = run / f"{args.phase}.log"
    with logfile.open("a") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    record = {"pid": process.pid, "phase": args.phase, "gpu_index": 0, "gpu_uuid": gpu_uuid,
              "log": str(logfile), "started_at": time.time()}
    marker.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
