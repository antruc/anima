"""Checks for the UI translations and system language detection."""

import sys

import pytest

from anima.app import TEXT, system_lang


def test_same_keys():
    """Every language defines exactly the same strings."""
    assert TEXT["es"].keys() == TEXT["en"].keys()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows reads the OS UI language")
def test_system_lang_env(monkeypatch):
    """LANGUAGE wins over LANG, and no variable gives ''."""
    for var in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(var, raising=False)
    assert system_lang() == ""
    monkeypatch.setenv("LANG", "es_MX.UTF-8")
    assert system_lang() == "es"
    monkeypatch.setenv("LANGUAGE", "en_US:es")
    assert system_lang() == "en"


def test_system_lang_returns_code():
    """On any OS the result is a short lowercase code."""
    lang = system_lang()
    assert lang == lang.lower() and len(lang) <= 2
