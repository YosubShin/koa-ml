"""Integration tests for koa-ml.

These tests require actual KOA credentials and connectivity.
Run with: pytest tests/test_integration.py -v

Skip these tests by default with: pytest tests/ -v -m "not integration"
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from koa_ml import cancel_job, list_jobs, run_health_checks, submit_job
from koa_ml.config import load_config

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def config():
    """Load the actual KOA configuration.

    This will fail if you haven't set up ~/.config/koa-ml/config.yaml
    """
    try:
        return load_config()
    except FileNotFoundError:
        pytest.skip("KOA configuration not found. Set up ~/.config/koa-ml/config.yaml")


@pytest.fixture(scope="module")
def test_job_script(tmp_path_factory):
    """Create a minimal test job script."""
    tmp_path = tmp_path_factory.mktemp("jobs")
    script = tmp_path / "integration_test.slurm"
    script.write_text(
        """#!/bin/bash
#SBATCH --job-name=koa-ml-test
#SBATCH --partition=sandbox
#SBATCH --time=00:02:00
#SBATCH --output=koa-ml-test-%j.out

echo "Integration test job running on $(hostname)"
echo "Current time: $(date)"
sleep 30
echo "Test complete"
"""
    )
    return script


class TestHealthChecks:
    """Test KOA connectivity and health checks."""

    def test_run_health_checks(self, config):
        """Test that we can connect to KOA and run basic commands."""
        output = run_health_checks(config)

        assert output, "Health check returned empty output"
        assert "hostname" in output.lower(), "Health check missing hostname"
        assert "sinfo" in output.lower(), "Health check missing sinfo"

        print("\n=== Health Check Output ===")
        print(output)


class TestJobManagement:
    """Test job submission, listing, and cancellation."""

    def test_list_jobs(self, config):
        """Test listing current jobs."""
        output = list_jobs(config)

        # Output should at least have headers even if no jobs running
        assert output, "list_jobs returned empty output"
        print("\n=== Current Jobs ===")
        print(output)

    @pytest.mark.slow
    def test_submit_and_cancel_job(self, config, test_job_script):
        """Test the full job lifecycle: submit, verify, cancel.

        This test actually submits a job to KOA and cancels it.
        Marked as 'slow' since it interacts with a real cluster.
        """
        # Submit the job
        job_id = submit_job(
            config,
            test_job_script,
            sbatch_args=["--partition", "sandbox"],
        )

        print(f"\n=== Submitted job {job_id} ===")
        assert job_id.isdigit(), f"Expected numeric job ID, got: {job_id}"

        try:
            # Give the job a moment to appear in the queue
            time.sleep(2)

            # Verify the job appears in the queue
            jobs_output = list_jobs(config)
            assert job_id in jobs_output, f"Job {job_id} not found in queue"
            print(f"Job {job_id} confirmed in queue")

        finally:
            # Always cancel the job, even if assertions fail
            cancel_job(config, job_id)
            print(f"Cancelled job {job_id}")

            # Verify cancellation
            time.sleep(2)
            jobs_output = list_jobs(config)
            # Job should either be gone or show as CANCELLED
            if job_id in jobs_output:
                assert "CANCEL" in jobs_output.upper(), f"Job {job_id} still active"


class TestJobSubmissionVariants:
    """Test different job submission configurations."""

    @pytest.mark.slow
    def test_submit_with_custom_remote_name(self, config, test_job_script):
        """Test submitting with a custom remote filename."""
        job_id = submit_job(
            config,
            test_job_script,
            sbatch_args=["--partition", "sandbox"],
            remote_name="custom_test.slurm",
        )

        try:
            assert job_id.isdigit()
            print(f"Submitted job {job_id} with custom remote name")
        finally:
            cancel_job(config, job_id)

    @pytest.mark.slow
    def test_submit_with_gpu_request(self, config, test_job_script):
        """Test submitting with GPU resource requests.

        Note: This uses gpu-sandbox which has limited availability.
        The job may queue for a while before running.
        """
        job_id = submit_job(
            config,
            test_job_script,
            sbatch_args=[
                "--partition",
                "gpu-sandbox",
                "--gres=gpu:1",
                "--time",
                "00:02:00",
            ],
        )

        try:
            assert job_id.isdigit()
            print(f"Submitted GPU job {job_id}")

            # Check it's in the queue (might be pending)
            time.sleep(2)
            jobs_output = list_jobs(config)
            assert job_id in jobs_output
            print(f"GPU job {job_id} in queue")
        finally:
            cancel_job(config, job_id)


@pytest.mark.manual
class TestManualVerification:
    """Tests that require manual verification.

    These are marked with @pytest.mark.manual and won't run automatically.
    Run explicitly with: pytest tests/test_integration.py -v -m manual
    """

    def test_job_output_file_created(self, config, test_job_script):
        """Submit a job and verify the output file is created on KOA.

        This test submits a job and lets it run to completion.
        You'll need to manually check KOA for the output file.
        """
        job_id = submit_job(
            config,
            test_job_script,
            sbatch_args=["--partition", "sandbox"],
        )

        print(f"\n{'='*60}")
        print(f"Submitted job {job_id}")
        print(f"Output file will be: koa-ml-test-{job_id}.out")
        print(f"Location: {config.remote_workdir}/koa-ml-test-{job_id}.out")
        print(f"\nTo check the job status:")
        print(f"  koa-ml jobs")
        print(f"\nTo view output when complete:")
        print(f"  ssh {config.login} cat {config.remote_workdir}/koa-ml-test-{job_id}.out")
        print(f"\nTo cancel early:")
        print(f"  koa-ml cancel {job_id}")
        print(f"{'='*60}\n")

        # Don't auto-cancel for manual verification
