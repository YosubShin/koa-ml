from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import Config, load_config
from .slurm import cancel_job, list_jobs, run_health_checks, submit_job
from .ssh import SSHError


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the KOA config file (defaults to ~/.config/koa-ml/config.yaml).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="koa-ml",
        description="Utilities for running KOA HPC (Slurm) jobs from your local machine.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check", help="Run KOA connectivity health checks."
    )
    _add_common_arguments(check_parser)

    jobs_parser = subparsers.add_parser(
        "jobs", help="List active KOA jobs for the configured user."
    )
    _add_common_arguments(jobs_parser)

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a KOA job by id.")
    _add_common_arguments(cancel_parser)
    cancel_parser.add_argument("job_id", help="Slurm job id to cancel.")

    submit_parser = subparsers.add_parser(
        "submit", help="Submit a job script via sbatch."
    )
    _add_common_arguments(submit_parser)
    submit_parser.add_argument(
        "job_script", type=Path, help="Path to the local job script."
    )
    submit_parser.add_argument("--remote-name", help="Override the filename on KOA.")
    submit_parser.add_argument(
        "--partition", help="Slurm partition (queue) to submit to."
    )
    submit_parser.add_argument("--time", help="Walltime request (e.g. 02:00:00).")
    submit_parser.add_argument("--gpus", type=int, help="Number of GPUs to request.")
    submit_parser.add_argument("--cpus", type=int, help="Number of CPUs to request.")
    submit_parser.add_argument("--memory", help="Memory request (e.g. 32G).")
    submit_parser.add_argument("--account", help="Slurm account if required.")
    submit_parser.add_argument("--qos", help="Quality of service if required.")
    submit_parser.add_argument(
        "--sbatch-arg",
        action="append",
        default=[],
        help="Additional raw sbatch arguments. Repeat for multiple flags.",
    )

    return parser


def _load(args: argparse.Namespace) -> Config:
    return load_config(args.config)


def _submit(args: argparse.Namespace, config: Config) -> int:
    sbatch_args: list[str] = []
    if args.partition:
        sbatch_args.extend(["--partition", args.partition])
    if args.time:
        sbatch_args.extend(["--time", args.time])
    if args.gpus:
        sbatch_args.append(f"--gres=gpu:{args.gpus}")
    if args.cpus:
        sbatch_args.extend(["--cpus-per-task", str(args.cpus)])
    if args.memory:
        sbatch_args.extend(["--mem", args.memory])
    if args.account:
        sbatch_args.extend(["--account", args.account])
    if args.qos:
        sbatch_args.extend(["--qos", args.qos])
    sbatch_args.extend(args.sbatch_arg or [])

    job_id = submit_job(
        config,
        args.job_script,
        sbatch_args=sbatch_args,
        remote_name=args.remote_name,
    )
    print(f"Submitted KOA job {job_id}")
    return 0


def _cancel(args: argparse.Namespace, config: Config) -> int:
    cancel_job(config, args.job_id)
    print(f"Cancelled KOA job {args.job_id}")
    return 0


def _jobs(_: argparse.Namespace, config: Config) -> int:
    print(list_jobs(config), end="")
    return 0


def _check(_: argparse.Namespace, config: Config) -> int:
    print(run_health_checks(config), end="")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = _load(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    try:
        if args.command == "submit":
            return _submit(args, config)
        if args.command == "cancel":
            return _cancel(args, config)
        if args.command == "jobs":
            return _jobs(args, config)
        if args.command == "check":
            return _check(args, config)
    except (SSHError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unhandled command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
