"""Run factor diagnostics after a materialization process completes."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def alive(pid: int) -> bool:
    result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, errors="replace")
    return str(pid) in result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    args = parser.parse_args()
    while alive(args.wait_pid):
        print(f"waiting_for_materialization pid={args.wait_pid}", flush=True)
        time.sleep(args.interval_seconds)
    print("materialization_finished; running_factor_research", flush=True)
    subprocess.run([sys.executable, "-m", "quant_research.pipeline", "factor-research"], check=True)


if __name__ == "__main__":
    main()
