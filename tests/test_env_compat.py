"""Unit tests for :mod:`omnigent._env_compat` (legacy env-prefix mirroring)."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from omnigent import _env_compat


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give each test a clean slate: reset the module's once-only guard and
    strip any pre-existing ``OMNIGENT_*`` / ``OMNIGENTS_*`` / ``OMNIAGENTS_*``
    vars so a real environment can't leak into (or out of) the test."""
    monkeypatch.setattr(_env_compat, "_mirrored", False)
    for name in list(os.environ):
        if name.startswith(("OMNIGENT_", "OMNIGENTS_", "OMNIAGENTS_")):
            monkeypatch.delenv(name, raising=False)
    yield


def test_legacy_var_mirrored_when_new_name_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIGENTS_SKIP_WEB_UI", "1")

    _env_compat.mirror_legacy_env()

    assert os.environ["OMNIGENT_SKIP_WEB_UI"] == "1"
    # The legacy var itself is left in place, only mirrored.
    assert os.environ["OMNIGENTS_SKIP_WEB_UI"] == "1"


def test_oldest_legacy_prefix_also_mirrored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIAGENTS_TOKEN", "abc")

    _env_compat.mirror_legacy_env()

    assert os.environ["OMNIGENT_TOKEN"] == "abc"


def test_explicit_new_value_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIGENT_HOST", "explicit")
    monkeypatch.setenv("OMNIGENTS_HOST", "legacy")

    _env_compat.mirror_legacy_env()

    # An explicitly-set new-name var is never overwritten by a legacy one.
    assert os.environ["OMNIGENT_HOST"] == "explicit"


def test_newest_legacy_prefix_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both legacy prefixes set for the same suffix; the newer OMNIGENTS_ should win.
    monkeypatch.setenv("OMNIGENTS_PORT", "new")
    monkeypatch.setenv("OMNIAGENTS_PORT", "old")

    _env_compat.mirror_legacy_env()

    assert os.environ["OMNIGENT_PORT"] == "new"


def test_once_only_guard_skips_rescan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIGENTS_FIRST", "1")
    _env_compat.mirror_legacy_env()
    assert os.environ["OMNIGENT_FIRST"] == "1"

    # A legacy var introduced after the first call must NOT be mirrored: the
    # guard short-circuits the second call before it rescans the environment.
    monkeypatch.setenv("OMNIGENTS_SECOND", "2")
    _env_compat.mirror_legacy_env()

    assert "OMNIGENT_SECOND" not in os.environ


def test_once_only_guard_does_not_overwrite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIGENTS_VALUE", "from-legacy")
    _env_compat.mirror_legacy_env()
    assert os.environ["OMNIGENT_VALUE"] == "from-legacy"

    # Even if the mirrored value is changed afterwards, a second call is a no-op.
    monkeypatch.setenv("OMNIGENT_VALUE", "mutated")
    _env_compat.mirror_legacy_env()

    assert os.environ["OMNIGENT_VALUE"] == "mutated"


def test_non_prefixed_vars_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH_LIKE_VAR", "keep-me")
    monkeypatch.setenv("OMNIGENTSX_NOT_A_PREFIX", "also-keep")

    _env_compat.mirror_legacy_env()

    assert os.environ["PATH_LIKE_VAR"] == "keep-me"
    # ``OMNIGENTSX_`` starts with ``OMNIGENTS_``? No -- the underscore matters.
    assert os.environ["OMNIGENTSX_NOT_A_PREFIX"] == "also-keep"
    # Nothing was mirrored from a non-legacy name.
    assert "OMNIGENT_NOT_A_PREFIX" not in os.environ
