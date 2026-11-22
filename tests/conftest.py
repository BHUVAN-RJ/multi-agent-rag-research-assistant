"""Shared fixtures: offline tests run against the code defaults, not the developer's .env."""

from dataclasses import replace

import pytest

from src.llm import providers


@pytest.fixture(autouse=True)
def default_openrouter_chain(monkeypatch):
    # A local .env can list extra OpenRouter ids; a grader's checkout has no .env at all.
    # Both must see the same chain, or a test passes on one machine and fails on the other.
    default_settings = replace(providers.settings, openrouter_free_models=[])
    monkeypatch.setattr(providers, "settings", default_settings)
