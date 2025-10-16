from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

from .config import Config
from .ssh import SSHError, copy_to_remote, run_ssh

SBATCH_JOB_ID_PATTERN = re.compile(r"Submitted batch job (\d+)")


def ensure_remote_workspace(config: Config) -> None:
    run_ssh(config, ["mkdir", "-p", str(config.remote_workdir)])


def submit_job(
    config: Config,
    local_job_script: Path,
    *,
    sbatch_args: Optional[Iterable[str]] = None,
    remote_name: Optional[str] = None,
) -> str:
    """
    Copy a job script to KOA and submit it with sbatch. Returns the job id.
    """
    if not local_job_script.exists():
        raise FileNotFoundError(f"Job script not found: {local_job_script}")

    ensure_remote_workspace(config)

    remote_script = config.remote_workdir / (remote_name or local_job_script.name)
    copy_to_remote(config, local_job_script, remote_script)

    args = ["sbatch"]
    if sbatch_args:
        args.extend(sbatch_args)
    args.append(str(remote_script))

    result = run_ssh(config, args, capture_output=True)
    output = result.stdout.strip() if result.stdout else ""
    match = SBATCH_JOB_ID_PATTERN.search(output)
    if not match:
        raise SSHError(f"Unable to parse sbatch output for job id: {output}")
    return match.group(1)


def cancel_job(config: Config, job_id: str) -> None:
    run_ssh(config, ["scancel", job_id])


def list_jobs(config: Config) -> str:
    result = run_ssh(
        config,
        [
            "squeue",
            "-u",
            config.user,
            "-o",
            r"%i|%j|%T|%M|%l|%D|%R",
        ],
        capture_output=True,
    )
    return result.stdout


def run_health_checks(config: Config) -> str:
    result = run_ssh(
        config,
        [
            "bash",
            "-lc",
            (
                "set -euo pipefail;"
                "echo '== hostname =='; hostname;"
                "echo '== sinfo =='; sinfo -o '%P %a %l %D %G %m'"
            ),
        ],
        capture_output=True,
    )
    return result.stdout
