"""OpenCode -> mock OpenAI integration (no GPU, no user config touched)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from slurmhelps.serving.mock import MockOpenAIServer

OPENCODE = shutil.which("opencode")


@pytest.mark.skipif(not OPENCODE, reason="opencode CLI is not installed")
def test_opencode_calls_mock_through_openai_compatible_provider(tmp_path):
    server = MockOpenAIServer(model="glm53-reap50", max_model_len=786432).start()
    try:
        xdg = tmp_path / "xdg"
        (xdg / "opencode").mkdir(parents=True)
        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "local": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "slurmhelps local",
                    "options": {"baseURL": server.base_url + "/v1", "apiKey": "dummy"},
                    "models": {"glm53-reap50": {"name": "GLM local"}},
                }
            },
        }
        (xdg / "opencode" / "opencode.json").write_text(json.dumps(config))
        env = dict(
            os.environ,
            XDG_CONFIG_HOME=str(xdg),
            XDG_DATA_HOME=str(tmp_path / "data"),
            XDG_CACHE_HOME=str(tmp_path / "cache"),
            XDG_STATE_HOME=str(tmp_path / "state"),
        )
        proc = subprocess.run(
            [
                OPENCODE,
                "run",
                "--format",
                "json",
                "--model",
                "local/glm53-reap50",
                "hello from the slurmhelps smoke",
            ],
            text=True,
            capture_output=True,
            timeout=120,
            env=env,
        )
        assert proc.returncode == 0, proc.stderr + proc.stdout
        assert '"type":"error"' not in proc.stdout
        assert '"type":"text"' in proc.stdout
        # The mock records the request; prove it actually hit our backend.
        assert server.requests
    finally:
        server.stop()
