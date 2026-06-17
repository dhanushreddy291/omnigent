"""
Behavioral I/O tests for :class:`omnigent.terminals.TerminalRegistry`
driven against a real tmux.

These complement :mod:`tests.terminals.test_registry` (which pins the
registry's *lifecycle* invariants — launch / get / close / list /
cleanup / shutdown — but never sends a keystroke or reads a byte) and
:mod:`tests.tools.builtins.test_sys_terminal` (which drives the same
behaviors through the ``sys_terminal_*`` tool envelopes). The module
here exercises the load-bearing capabilities that
``sys_terminal_*`` adds over a one-shot ``sys_os_shell`` — interactive
state that survives across calls, cwd anchoring of the live shell,
control-key delivery, and per-session isolation — at the tightest
layer that still touches a real tmux: ``TerminalRegistry.launch`` →
``TerminalInstance.send`` / ``.read``.

Why this layer (not the e2e suite): the equivalent end-to-end
coverage in ``tests/e2e/test_sys_terminal_e2e.py`` is fully suppressed
in ``tests/known_failures.yaml`` (cluster ``terminal-d6``, "requires
running runner") because it needs a live runner shard and a real LLM.
These tests reach the same tmux behaviors deterministically — no
runner, no LLM, no flakiness from prose — so the capability keeps
durable coverage in the normal ``tests/terminals`` CI shard, which
installs tmux (see ``.github/workflows/ci.yml``).

Every test launches a real tmux and is skipped when tmux is not on
PATH. ``send`` is asynchronous from the shell's perspective (tmux
delivers keystrokes, the shell renders on its own clock), so reads
poll with a bounded budget rather than asserting on a single capture.
"""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from omnigent.inner.datamodel import OSEnvSandboxSpec, OSEnvSpec, TerminalEnvSpec
from omnigent.inner.terminal import TerminalInstance
from omnigent.terminals import TerminalRegistry

pytestmark = pytest.mark.skipif(
    shutil.which("tmux") is None,
    reason="tmux not installed; registry I/O tests need a real tmux on PATH",
)

# Bounded poll budget for a marker to render in the pane. tmux delivers
# send-keys and the shell echoes on its own clock; a fixed sleep either
# wastes time or races. 5s comfortably covers a cold bash prompt plus
# echo on a loaded CI box while keeping a hung shell from stalling the
# suite. The matching round-trip test in
# ``tests/tools/builtins/test_sys_terminal.py`` uses a ~2s budget; this
# is deliberately roomier because some tests here chain two sends.
_MARKER_BUDGET_S = 5.0
_POLL_INTERVAL_S = 0.1


def _bash_spec(cwd: Path, *, allow_cwd_override: bool = False) -> TerminalEnvSpec:
    """A minimal bash :class:`TerminalEnvSpec` anchored at *cwd*, sandbox off.

    Sandbox is forced to ``none`` so the test doesn't depend on
    bwrap / seatbelt availability — these tests exercise tmux I/O,
    not sandboxing (covered elsewhere). The cwd is set on the
    terminal's own ``os_env`` so the launched shell starts there.

    :param cwd: Directory the spawned shell starts in.
    :param allow_cwd_override: Whether per-launch ``cwd_override`` is
        permitted. Defaults to ``False`` (matches the spec default).
    :returns: A :class:`TerminalEnvSpec` ready to launch.
    """
    return TerminalEnvSpec(
        command="bash",
        allow_cwd_override=allow_cwd_override,
        os_env=OSEnvSpec(
            type="caller_process",
            cwd=str(cwd),
            sandbox=OSEnvSandboxSpec(type="none"),
        ),
    )


async def _read_until(
    instance: TerminalInstance,
    needle: str,
    *,
    budget_s: float = _MARKER_BUDGET_S,
) -> str:
    """Poll ``instance.read`` until *needle* shows in the pane or budget elapses.

    :param instance: The launched terminal to read from.
    :param needle: Substring expected to appear in the pane capture.
    :param budget_s: Total seconds to poll before giving up.
    :returns: The last pane ``screen`` text observed — contains
        *needle* on success, or the final capture (for a useful
        failure message) on timeout.
    """
    import asyncio

    waited = 0.0
    screen = ""
    while waited < budget_s:
        result = await instance.read()
        screen = result.get("screen", "")
        if needle in screen:
            return screen
        await asyncio.sleep(_POLL_INTERVAL_S)
        waited += _POLL_INTERVAL_S
    return screen


@pytest.fixture
def reg() -> TerminalRegistry:
    """A fresh registry per test."""
    return TerminalRegistry()


@pytest.fixture
async def cleanup(reg: TerminalRegistry) -> AsyncIterator[None]:
    """Close every terminal at test exit, even when an assertion fails.

    Declared to depend on ``reg`` so its teardown runs before the
    registry fixture is discarded — otherwise a failed assertion
    mid-test would leak a live tmux server.

    :param reg: The registry fixture whose terminals get torn down.
    :yields: ``None`` — the value is unused; the fixture exists for
        its teardown side effect.
    """
    yield
    await reg.shutdown()


# ── Interactive state persistence ─────────────────────────────


async def test_shell_state_persists_across_separate_sends(
    reg: TerminalRegistry, cleanup: None, tmp_path: Path
) -> None:
    """A variable set in one ``send`` is still set in a later ``send``.

    This is the load-bearing capability ``sys_terminal_*`` adds over a
    one-shot ``sys_os_shell``: the shell is a single long-lived process
    across calls, so state (here a shell variable) set by the first
    send is visible to the second. A naive "spawn a fresh shell per
    send" implementation would pass every lifecycle test in
    :mod:`tests.terminals.test_registry` yet fail here, because the
    second send would run in a shell that never saw the assignment.

    Mirrors the suppressed
    ``test_sys_terminal_persists_across_turns_e2e`` (which proves the
    same thing across two LLM turns) at the registry→tmux layer.

    :param reg: Fresh registry.
    :param cleanup: Teardown side-effect fixture.
    :param tmp_path: Per-test working directory for the shell.
    """
    del cleanup
    instance = await reg.launch("conv_a", "bash", "s1", _bash_spec(tmp_path))

    # First send establishes state; second send observes it. Two
    # distinct send() calls — not one compound command — so the test
    # actually proves cross-call persistence.
    await instance.send(text="MARKER_VAR=persisted_value", keys="Enter")
    await instance.send(text="echo VAR_IS_$MARKER_VAR", keys="Enter")

    screen = await _read_until(instance, "VAR_IS_persisted_value")
    assert "VAR_IS_persisted_value" in screen, (
        "Shell variable set in the first send was not visible in the "
        "second. The shell is supposed to be one long-lived process "
        "across send() calls; if this fails, each send is spawning a "
        f"fresh shell (state lost). Last pane:\n{screen!r}"
    )


async def test_working_directory_change_persists_across_sends(
    reg: TerminalRegistry, cleanup: None, tmp_path: Path
) -> None:
    """A ``cd`` in one send is reflected by ``pwd`` in a later send.

    The directory-state analogue of the variable-persistence test —
    and the exact behavior the suppressed persistence e2e asserts
    (``cd /tmp`` in turn 1, ``pwd`` shows ``/tmp`` in turn 2). Proving
    it here at the tmux layer removes the dependency on a live runner.

    :param reg: Fresh registry.
    :param cleanup: Teardown side-effect fixture.
    :param tmp_path: Working directory; a ``subdir`` is created to cd into.
    """
    del cleanup
    subdir = tmp_path / "nested_dir"
    subdir.mkdir()
    instance = await reg.launch("conv_a", "bash", "s1", _bash_spec(tmp_path))

    await instance.send(text=f"cd {subdir}", keys="Enter")
    await instance.send(text="pwd", keys="Enter")

    # macOS resolves /var -> /private/var; accept either spelling by
    # matching the leaf directory name, which is unambiguous here.
    screen = await _read_until(instance, "nested_dir")
    assert "nested_dir" in screen, (
        "cd from the first send did not persist into the second send's "
        f"pwd. The session lost its working directory. Last pane:\n{screen!r}"
    )


# ── cwd anchoring of the live shell ───────────────────────────


async def test_launched_shell_starts_in_spec_cwd(
    reg: TerminalRegistry, cleanup: None, tmp_path: Path
) -> None:
    """``pwd`` in a freshly launched shell reports the spec's cwd.

    Unit tests in ``tests/tools/builtins/test_sys_terminal.py`` pin
    ``_resolve_cwd``'s precedence logic in isolation, but nothing at
    this layer proves the resolved cwd actually anchors the *live*
    shell. This drives ``pwd`` against the real tmux and asserts the
    shell landed where the spec said — the behavioral half of the
    suppressed ``test_sys_terminal_cwd_default_is_workspace_e2e``.

    :param reg: Fresh registry.
    :param cleanup: Teardown side-effect fixture.
    :param tmp_path: The directory the spec anchors the shell to.
    """
    del cleanup
    instance = await reg.launch("conv_a", "bash", "s1", _bash_spec(tmp_path))

    await instance.send(text="pwd", keys="Enter")

    leaf = tmp_path.name
    screen = await _read_until(instance, leaf)
    assert leaf in screen, (
        f"Launched shell's pwd did not report the spec cwd {tmp_path}. "
        f"The terminal landed somewhere other than its configured cwd. "
        f"Last pane:\n{screen!r}"
    )


async def test_cwd_override_anchors_live_shell_in_subdirectory(
    reg: TerminalRegistry, cleanup: None, tmp_path: Path
) -> None:
    """A per-launch ``cwd_override`` starts the live shell in that subdir.

    Complements the disallowed-override *rejection* test in the tool
    suite: here the override is allowed and we prove it actually moves
    the spawned shell's cwd (``pwd`` reports the subdirectory and the
    instance records it on ``launch_cwd``). The override is contained
    to a subdirectory of the spec cwd, matching
    ``build_terminal_os_env_spec``'s containment guard.

    :param reg: Fresh registry.
    :param cleanup: Teardown side-effect fixture.
    :param tmp_path: Spec cwd; a ``workdir`` subdir is the override target.
    """
    del cleanup
    override_dir = tmp_path / "workdir"
    override_dir.mkdir()
    spec = _bash_spec(tmp_path, allow_cwd_override=True)

    instance = await reg.launch(
        "conv_a",
        "bash",
        "s1",
        spec,
        cwd_override=str(override_dir),
    )

    # The instance records the resolved launch cwd; assert it points
    # at the override target (resolved for the macOS /var symlink).
    assert instance.launch_cwd is not None
    assert Path(instance.launch_cwd).name == "workdir", (
        f"launch_cwd did not reflect the cwd_override; got "
        f"{instance.launch_cwd!r}, expected a path ending in 'workdir'."
    )

    await instance.send(text="pwd", keys="Enter")
    screen = await _read_until(instance, "workdir")
    assert "workdir" in screen, (
        "cwd_override did not anchor the live shell: pwd never reported "
        f"the override subdirectory. Last pane:\n{screen!r}"
    )


# ── Control-key delivery ──────────────────────────────────────


async def test_ctrl_c_interrupts_running_command(
    reg: TerminalRegistry, cleanup: None, tmp_path: Path
) -> None:
    """``keys="C-c"`` interrupts a running foreground command.

    Exercises the ``send`` path's control-key handling (``text=None``,
    ``keys="C-c"``) against a live shell — the interactive capability
    that distinguishes ``sys_terminal_*`` from a one-shot command run.
    A long ``sleep`` is started, interrupted with C-c, and a marker
    echo afterward proves the shell returned to its prompt rather than
    staying blocked in the sleep.

    :param reg: Fresh registry.
    :param cleanup: Teardown side-effect fixture.
    :param tmp_path: Working directory for the shell.
    """
    import asyncio

    del cleanup
    instance = await reg.launch("conv_a", "bash", "s1", _bash_spec(tmp_path))

    # Start a long-running foreground command, then let it actually
    # begin before interrupting (a C-c that lands before the shell
    # forks `sleep` would just edit the command line).
    await instance.send(text="sleep 120", keys="Enter")
    await asyncio.sleep(0.4)

    # Interrupt: no literal text, just the control key.
    interrupt = await instance.send(text=None, keys="C-c")
    assert interrupt.get("status") == "sent", (
        f"send of C-c did not report status='sent': {interrupt!r}"
    )

    # If the interrupt worked, the shell is back at a prompt and this
    # echo runs promptly. If C-c was swallowed, the echo sits behind a
    # 120s sleep and the marker never shows within the budget.
    await instance.send(text="echo INTERRUPT_RECOVERED_OK", keys="Enter")
    screen = await _read_until(instance, "INTERRUPT_RECOVERED_OK")
    assert "INTERRUPT_RECOVERED_OK" in screen, (
        "Marker echo did not appear after C-c, so the interrupt didn't "
        "return the shell to its prompt — the foreground `sleep` was "
        f"never interrupted. Last pane:\n{screen!r}"
    )


# ── Parallel-session isolation (driven by real I/O) ───────────


async def test_parallel_sessions_have_isolated_shell_state(
    reg: TerminalRegistry, cleanup: None, tmp_path: Path
) -> None:
    """Two sessions of the same terminal don't share shell state.

    ``test_registry.py`` proves distinct session keys yield distinct
    instances and sockets (object identity); this proves the stronger
    *behavioral* property the design promises — independent tmux
    sessions running in parallel — by setting the same variable name to
    different values in each and confirming each session reports only
    its own value. A registry bug that collapsed ``(name, key)`` to
    ``name`` (one shared tmux) would surface here as cross-talk even if
    the socket-identity assertions elsewhere still passed.

    :param reg: Fresh registry.
    :param cleanup: Teardown side-effect fixture.
    :param tmp_path: Working directory shared by both sessions (state
        isolation is per-shell-process, not per-cwd).
    """
    del cleanup
    spec = _bash_spec(tmp_path)
    s1 = await reg.launch("conv_a", "bash", "s1", spec)
    s2 = await reg.launch("conv_a", "bash", "s2", spec)

    assert s1 is not s2
    assert s1.socket_path != s2.socket_path

    # Same variable name, different values, one per session.
    await s1.send(text="SESSION_TAG=alpha_one", keys="Enter")
    await s2.send(text="SESSION_TAG=beta_two", keys="Enter")
    await s1.send(text="echo TAG=$SESSION_TAG", keys="Enter")
    await s2.send(text="echo TAG=$SESSION_TAG", keys="Enter")

    s1_screen = await _read_until(s1, "TAG=alpha_one")
    s2_screen = await _read_until(s2, "TAG=beta_two")

    assert "TAG=alpha_one" in s1_screen, (
        f"Session s1 did not report its own value. Pane:\n{s1_screen!r}"
    )
    assert "TAG=beta_two" in s2_screen, (
        f"Session s2 did not report its own value. Pane:\n{s2_screen!r}"
    )
    # The load-bearing isolation assertion: neither session sees the
    # other's value. Cross-talk here means the two session keys are
    # backed by a single shared tmux session.
    assert "beta_two" not in s1_screen, (
        "Session s1's pane shows session s2's value — the two session "
        f"keys are not isolated. s1 pane:\n{s1_screen!r}"
    )
    assert "alpha_one" not in s2_screen, (
        "Session s2's pane shows session s1's value — the two session "
        f"keys are not isolated. s2 pane:\n{s2_screen!r}"
    )


# ── Read-after-close error contract ───────────────────────────


async def test_send_and_read_after_close_report_not_running(
    reg: TerminalRegistry, cleanup: None, tmp_path: Path
) -> None:
    """Once closed, the instance's ``send`` / ``read`` error cleanly.

    ``test_registry.py`` proves ``close`` removes the registry *entry*;
    this proves the *instance* object refuses I/O afterward instead of
    talking to a dead tmux socket. The ``send`` tool path surfaces this
    as a "not running" error to the LLM, so the instance-level contract
    matters: a stale read against a killed server must not hang or
    return garbage.

    :param reg: Fresh registry.
    :param cleanup: Teardown side-effect fixture (idempotent with the
        explicit close below).
    :param tmp_path: Working directory for the shell.
    """
    del cleanup
    instance = await reg.launch("conv_a", "bash", "s1", _bash_spec(tmp_path))
    assert await instance.is_alive()

    closed = await reg.close("conv_a", "bash", "s1")
    assert closed is True

    assert instance.running is False
    assert await instance.is_alive() is False

    send_result = await instance.send(text="echo too_late", keys="Enter")
    assert "error" in send_result, (
        f"send after close should return an error envelope, got {send_result!r}"
    )
    read_result = await instance.read()
    assert "error" in read_result, (
        f"read after close should return an error envelope, got {read_result!r}"
    )
