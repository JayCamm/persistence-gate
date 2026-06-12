from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Persistence Gate product benchmark suite.")
    parser.add_argument("--profile", default="conservative")
    parser.add_argument("--top-k", default="3")
    parser.add_argument("--candidate-pool", default="8")
    parser.add_argument("--shuffle-replicates", default="10")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out-dir", default="benchmark_results/product_suite")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    dirty_cmd = [
        sys.executable,
        "benchmarks/dirty_metadata_trap_benchmark.py",
        "--profile",
        args.profile,
        "--top-k",
        args.top_k,
        "--candidate-pool",
        args.candidate_pool,
        "--out-dir",
        str(out / "dirty_metadata_trap"),
    ]
    if args.refresh:
        dirty_cmd.append("--refresh")
    run(dirty_cmd, root)

    audit_cmd = [
        sys.executable,
        "benchmarks/dirty_metadata_bias_audit.py",
        "--profile",
        args.profile,
        "--top-k",
        args.top_k,
        "--candidate-pool",
        args.candidate_pool,
        "--shuffle-replicates",
        args.shuffle_replicates,
        "--out-dir",
        str(out / "dirty_metadata_bias_audit"),
    ]
    if args.refresh:
        audit_cmd.append("--refresh")
    run(audit_cmd, root)

    manifest = out / "BENCHMARK_MANIFEST.txt"
    manifest.write_text(
        "Persistence Gate product benchmark suite\n"
        f"profile={args.profile}\n"
        f"top_k={args.top_k}\n"
        f"candidate_pool={args.candidate_pool}\n"
        f"shuffle_replicates={args.shuffle_replicates}\n"
        "outputs=dirty_metadata_trap, dirty_metadata_bias_audit\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
