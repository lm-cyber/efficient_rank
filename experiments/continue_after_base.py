"""Continue the same authorized run after the base-training process exits."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/erank_audit_v1"


def main():
    record = json.loads((RUN / "process.json").read_text())
    if record["phase"] != "base":
        raise RuntimeError("Expected the base phase")
    command = Path(f"/proc/{record['pid']}/cmdline")
    while command.exists() and b"experiments/base.py" in command.read_bytes():
        time.sleep(10)
    if not (RUN / "base_complete.json").exists():
        raise RuntimeError("Base process exited without completing; pool was not started")
    subprocess.run([sys.executable, str(ROOT / "experiments/launch.py"), "pool"], check=True)


if __name__ == "__main__":
    main()
