"""Unit tests for utils.fingerprint."""

import re

from utils.fingerprint import code_fingerprint


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode())


def test_format_and_determinism(tmp_path):
    _write(tmp_path, "a.py", "x = 1\n")
    fp = code_fingerprint(tmp_path)
    assert re.fullmatch(r"[0-9a-f]{64}", fp)
    assert fp == code_fingerprint(tmp_path)


def test_content_change_changes_hash(tmp_path):
    _write(tmp_path, "a.py", "x = 1\n")
    before = code_fingerprint(tmp_path)
    _write(tmp_path, "a.py", "x = 2\n")
    assert code_fingerprint(tmp_path) != before


def test_rename_changes_hash(tmp_path):
    _write(tmp_path, "a.py", "x = 1\n")
    before = code_fingerprint(tmp_path)
    (tmp_path / "a.py").rename(tmp_path / "b.py")
    assert code_fingerprint(tmp_path) != before


def test_yaml_included_other_files_ignored(tmp_path):
    _write(tmp_path, "a.py", "x = 1\n")
    base = code_fingerprint(tmp_path)
    _write(tmp_path, "notes.md", "ignored")
    assert code_fingerprint(tmp_path) == base
    _write(tmp_path, "config.yaml", "k: v\n")
    assert code_fingerprint(tmp_path) != base


def test_excluded_dirs_ignored(tmp_path):
    _write(tmp_path, "a.py", "x = 1\n")
    base = code_fingerprint(tmp_path)
    _write(tmp_path, ".venv/lib/x.py", "junk")
    _write(tmp_path, "__pycache__/y.py", "junk")
    _write(tmp_path, ".chroma/z.yaml", "junk")
    assert code_fingerprint(tmp_path) == base


def test_line_endings_normalised(tmp_path):
    _write(tmp_path, "a.py", "x = 1\ny = 2\n")
    lf = code_fingerprint(tmp_path)
    _write(tmp_path, "a.py", "x = 1\r\ny = 2\r\n")
    assert code_fingerprint(tmp_path) == lf


def test_creation_order_irrelevant(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    for root, order in ((one, ["a.py", "b.py"]), (two, ["b.py", "a.py"])):
        for name in order:
            _write(root, name, f"# {name}\n")
    assert code_fingerprint(one) == code_fingerprint(two)
