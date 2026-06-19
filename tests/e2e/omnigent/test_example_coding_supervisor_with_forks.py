"""End-to-end test for ``examples/agents/coding_supervisor_with_forks``.

Supervisor + two worker sub-agents, each with a forked os_env
(hardlink-tree COW). The test runs against the mock LLM server
so no real harness CLI binary is required.

YAML has ``sandbox: type: none`` everywhere so the sandbox is
off; the fork mode itself works cross-platform.

**What breaks if this fails:**
- Sub-agent ``os_env.fork`` propagation regresses.
- Per-worker harness specification is lost during spec translation.
- The ``sys_session_*`` + forked-env combination stops wiring
  the symlinks under ``.sessions/<worker>/`` that the supervisor
  reads to diff worker output.
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e.omnigent._example_helpers import (
    assert_completed_one_shot,
    run_one_shot,
)
from tests.e2e.omnigent.conftest import configure_mock_llm


def test_coding_supervisor_with_forks_one_shot(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    mock_credentials_env: dict[str, str],
    mock_llm_server_url: str,
) -> None:
    """
    Run the forked coding-supervisor one-shot against the mock LLM
    server and verify the run completes cleanly.

    Uses ``openai-agents`` harness with the mock LLM for
    deterministic responses. Provides canned replies for the
    supervisor turn and any worker sub-agent turns that may fire.

    :param omnigent_python: Interpreter with omnigent +
        openai-agents installed.
    :param omnigent_repo_root: Repo root for subprocess cwd.
    :param mock_credentials_env: Mock-LLM env vars.
    :param mock_llm_server_url: Mock server URL for configuring
        response queues.
    """
    # The supervisor may spawn one or both workers, each consuming
    # a response. Provide enough canned replies to cover the
    # supervisor turn plus potential worker turns and auto-wake.
    configure_mock_llm(
        mock_llm_server_url,
        [
            {"text": "I have delegated the work to the workers. Task complete."},
            {"text": "Worker A finished."},
            {"text": "Worker B finished."},
            {"text": "Both workers done. Summary: OK."},
        ],
    )
    result = run_one_shot(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        omnigent_credentials_env=mock_credentials_env,
        example_name="coding_supervisor_with_forks",
        harness="openai-agents",
        model="mock-model",
    )
    assert_completed_one_shot(result, "coding_supervisor_with_forks")
