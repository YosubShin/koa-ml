# koa-ml

---

Thin helper utilities for managing KOA HPC GPU jobs from a local workstation. The initial goal is to get comfortable allocating and cancelling GPU jobs on the KOA cluster; model training and experiment tracking can layer on afterwards.

## Quick start

1. Create a Python environment (Python 3.9 or newer; 3.10+ recommended) and install the package in editable mode:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip setuptools wheel
   pip install -e .
   ```
   If `pip install -e .` complains about Python 3.9, recreate the virtual environment with a newer interpreter (e.g., `pyenv install 3.11.9`).
2. Copy the example configuration and fill in your KOA account details:
   ```bash
   cp .koa-config.example.yaml ~/.config/koa-ml/config.yaml
   ```
   - `user`: KOA username (netid) – e.g., `mburiek`
   - `host`: KOA login node hostname (`koa.its.hawaii.edu`)
   - `identity_file`: absolute path to the SSH private key you use for KOA (optional if you rely on your default SSH config)
   - `remote_workdir`: directory on KOA where job scripts should live (defaults to `/home/<user>/koa-ml`)
3. Verify connectivity:
   ```bash
   koa-ml check
   ```
   This runs `sinfo` on KOA and prints the available partitions. If it fails, double-check SSH connectivity and that your key is loaded (`ssh user@host`).

## CLI usage

After installation you can run the CLI as `koa-ml <command>` (the `koa_ml` alias also works if you prefer underscores).

- `koa-ml check` — run a lightweight KOA health check (`hostname`, `sinfo`)
- `koa-ml submit jobs/example_job.slurm --partition gpu --gpus 1` — copy the job script to KOA, submit it with `sbatch`, and print the job id
- `koa-ml jobs` — list your active jobs using `squeue`
- `koa-ml cancel <job_id>` — cancel an active job (`scancel`)

Each command accepts `--config` if you want to point at a custom configuration path.

## Testing

The project includes comprehensive unit and integration tests. See [TESTING.md](TESTING.md) for detailed testing instructions.

Quick test commands:
```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run unit tests (no KOA connection needed)
pytest tests/ -m "not integration" -v

# Run all tests including integration tests (requires KOA credentials)
pytest tests/ -v
```

## Jobs folder

- `jobs/example_job.slurm` contains a minimal template that requests a single GPU and runs `nvidia-smi`.
- This is intended as a sanity check that you can grab a GPU allocation and release it.
- Update the script with the modules and conda environments you normally use.

## KOA quick reference

- Login node: `koa.its.hawaii.edu` (SSH/MFA required); Open OnDemand is available at https://koa.its.hawaii.edu
- Storage: `/home/<user>` (50 GB, snapshotted daily) and `/mnt/lustre/koa/scratch/<user>` (90-day purge); copy results off scratch promptly
- Common partitions: `gpu` (up to 3 days), `gpu-sandbox` (4 h, unique job slot), `sandbox` (CPU, 4 h), `shared` (3 d), `shared-long` (7 d)
- Slurm commands of interest: `sinfo`, `squeue -u <user>`, `sbatch`, `srun`, `scancel`
- Interactive test session: `srun -I30 -p sandbox -N 1 -c 1 --mem=6G -t 0-01:00:00 --pty /bin/bash`
- Support: uh-hpc-help@lists.hawaii.edu (include job id, script path, error/output files when reporting issues)

## Next steps

- Add higher-level job builders/templates once the base GPU allocation flow feels solid
- Integrate experiment logging (e.g., MLflow) after confirming KOA ↔ local data sync patterns
- Consider wrapping SSH auth around `sshing`'s `ProxyJump` or campus VPN requirements if needed
