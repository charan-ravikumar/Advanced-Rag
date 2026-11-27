"""eval/heldout.py: download once, pin, open once. A FAKE archive in tmp_path: no network, and the real held-out data is never touched."""

import ast
import io
import json
import tarfile
from pathlib import Path

import pytest

from eval import heldout as ho

ROOT = Path(__file__).resolve().parents[2]


def archive_bytes(members: dict[str, bytes], *, links: dict[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.SYMTYPE, target
            tar.addfile(info)
    return buf.getvalue()


GOOD = archive_bytes({ho.TEST_FILE: b'{"fake": "test data"}', "qasper_evaluator.py": b"raise SystemExit('never executed')\n"})


class Downloader:
    def __init__(self, payload: bytes = GOOD):
        self.payload, self.urls = payload, []

    def __call__(self, url, target):
        self.urls.append(url)
        Path(target).write_bytes(self.payload)


@pytest.fixture
def world(tmp_path):
    w = {"directory": tmp_path / "data" / "set", "manifest_path": tmp_path / "repo" / "manifest.json", "freeze_path": tmp_path / "repo" / "freeze.json",
         "ledger_path": tmp_path / "repo" / "ledger.jsonl", "prereg": tmp_path / "repo" / "PREREG.md"}
    w["prereg"].parent.mkdir(parents=True)
    w["prereg"].write_text("# plan\n", encoding="utf-8")
    return w


def secure(w, download=None):
    return ho.fetch(directory=w["directory"], manifest_path=w["manifest_path"], download=download or Downloader())


def do_freeze(w, **kw):
    kw.setdefault("fingerprint", "fp-1")
    return ho.freeze(arms=["D"], preregistration=w["prereg"], frozen_by="owner, test", directory=w["directory"], manifest_path=w["manifest_path"],
                     freeze_path=w["freeze_path"], **kw)


def do_open(w, **kw):
    from eval.run_retrieval_eval import CANDIDATE_K, TOP_K

    args = dict(confirmed=True, arms=["D"], top_k=TOP_K, candidate_k=CANDIDATE_K, preregistration=w["prereg"], purpose="final evaluation", fingerprint="fp-1")
    args.update(kw)
    return ho.open_heldout(directory=w["directory"], manifest_path=w["manifest_path"], freeze_path=w["freeze_path"], ledger_path=w["ledger_path"], **args)


# ---- securing -------------------------------------------------------------------------------------------------------------------------

def test_fetch_downloads_the_official_url_stores_read_only_files_and_pins_every_hash(world):
    dl = Downloader()
    state, manifest = secure(world, dl)
    assert state == "secured" and dl.urls == [ho.URL] and ho.URL.endswith("qasper-test-and-evaluator-v0.3.tgz")
    assert set(manifest["files"]) == {ho.TEST_FILE, "qasper_evaluator.py"} and len(manifest["archive"]["sha256"]) == 64
    assert json.loads(world["manifest_path"].read_text()) == manifest
    for name in (ho.ARCHIVE_NAME, *manifest["files"]):
        assert (world["directory"] / name).stat().st_mode & 0o222 == 0, name  # read-only
    assert [p.name for p in world["directory"].parent.iterdir()] == ["set"]  # no staging folder left behind


def test_a_second_fetch_does_not_download_again_or_rewrite_the_manifest(world):
    secure(world)
    before = world["manifest_path"].read_bytes()

    def forbidden(url, target):
        raise AssertionError("downloaded twice")

    assert secure(world, forbidden)[0] == "already secured" and world["manifest_path"].read_bytes() == before


def test_on_another_machine_only_the_identical_archive_is_accepted(world, tmp_path):
    secure(world)
    elsewhere = {**world, "directory": tmp_path / "other" / "set"}
    changed = archive_bytes({ho.TEST_FILE: b'{"fake": "changed"}'})
    with pytest.raises(ho.HeldoutRefused, match="differs from the pinned manifest"):
        secure(elsewhere, Downloader(changed))
    assert not elsewhere["directory"].exists()
    assert secure(elsewhere)[0] == "secured"


@pytest.mark.parametrize("payload", [archive_bytes({"something-else.json": b"{}"}), archive_bytes({ho.TEST_FILE: b"{}"}, links={"x": "/etc/passwd"}),
                                     archive_bytes({f"a/{ho.TEST_FILE}": b"{}", f"b/{ho.TEST_FILE}": b"{}"}), archive_bytes({ho.OPENED_MARKER: b"{}", ho.TEST_FILE: b"{}"})],
                         ids=["no test file", "a link", "two files with one name", "a fake opened marker"])
def test_a_suspicious_archive_is_refused_and_nothing_is_stored(world, payload):
    with pytest.raises(ho.HeldoutRefused):
        secure(world, Downloader(payload))
    assert not world["directory"].exists() and not world["manifest_path"].exists()


def test_path_traversal_in_a_member_name_stays_inside_the_folder(world, tmp_path):
    secure(world, Downloader(archive_bytes({ho.TEST_FILE: b"{}", "../../escaped.txt": b"x"})))
    assert (world["directory"] / "escaped.txt").exists() and not (tmp_path / "escaped.txt").exists() and not (tmp_path / "data" / "escaped.txt").exists()


def test_verify_notices_a_changed_or_missing_file(world):
    secure(world)
    args = dict(directory=world["directory"], manifest_path=world["manifest_path"])
    assert ho.verify(**args)["dataset"] == ho.DATASET
    target = world["directory"] / ho.TEST_FILE
    target.chmod(0o644)
    target.write_bytes(b'{"fake": "TEST data"}')  # same size, different bytes
    with pytest.raises(ho.HeldoutRefused, match="does not match"):
        ho.verify(**args)
    target.unlink()
    with pytest.raises(ho.HeldoutRefused, match="missing"):
        ho.verify(**args)
    with pytest.raises(ho.HeldoutRefused, match="no manifest"):
        ho.verify(directory=world["directory"], manifest_path=world["manifest_path"].with_name("none.json"))


def test_the_download_keeps_certificate_verification_on():
    import ssl

    context = ho._tls_context()
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname and context.cert_store_stats()["x509_ca"] > 0


# ---- freezing and opening --------------------------------------------------------------------------------------------------------------

def test_nothing_opens_without_confirmation_a_purpose_and_a_freeze(world):
    secure(world)
    with pytest.raises(ho.HeldoutRefused, match="no frozen configuration"):
        do_open(world)
    do_freeze(world)
    with pytest.raises(ho.HeldoutRefused, match="explicit confirmation"):
        do_open(world, confirmed=False)
    with pytest.raises(ho.HeldoutRefused, match="what this one opening is for"):
        do_open(world, purpose=" ")
    assert not world["ledger_path"].exists() and not (world["directory"] / ho.OPENED_MARKER).exists()  # a refusal uses nothing up


def test_a_freeze_needs_a_preregistration_and_a_name_and_is_never_overwritten(world):
    secure(world)
    with pytest.raises(ho.HeldoutRefused, match="no pre-registration"):
        ho.freeze(arms=["D"], preregistration=world["prereg"].with_name("missing.md"), frozen_by="owner", directory=world["directory"],
                  manifest_path=world["manifest_path"], freeze_path=world["freeze_path"], fingerprint="fp-1")
    with pytest.raises(ho.HeldoutRefused, match="who asked"):
        ho.freeze(arms=["D"], preregistration=world["prereg"], frozen_by="", directory=world["directory"], manifest_path=world["manifest_path"],
                  freeze_path=world["freeze_path"], fingerprint="fp-1")
    frozen = do_freeze(world)
    assert frozen["data_sha256"] == json.loads(world["manifest_path"].read_text())["files"][ho.TEST_FILE]["sha256"] and list(frozen["arms"]) == ["D"]
    with pytest.raises(ho.HeldoutRefused, match="never edited or recreated"):
        do_freeze(world)


@pytest.mark.parametrize("change,needle", [({"arms": ["A"]}, "arms"), ({"top_k": 20}, "top_k"), ({"candidate_k": 100}, "candidate_k"),
                                           ({"fingerprint": "fp-2"}, "code_fingerprint")])
def test_anything_that_differs_from_the_freeze_is_refused(world, change, needle):
    secure(world)
    do_freeze(world)
    with pytest.raises(ho.HeldoutRefused, match=needle):
        do_open(world, **change)
    assert ho.openings(directory=world["directory"], ledger_path=world["ledger_path"]) == []


def test_an_edited_preregistration_is_refused(world):
    secure(world)
    do_freeze(world)
    world["prereg"].write_text("# a different plan\n", encoding="utf-8")
    with pytest.raises(ho.HeldoutRefused, match="preregistration_sha256"):
        do_open(world)


def test_changed_data_is_refused_at_opening(world):
    secure(world)
    do_freeze(world)
    target = world["directory"] / ho.TEST_FILE
    target.chmod(0o644)
    target.write_bytes(b"{}")
    with pytest.raises(ho.HeldoutRefused, match="does not match"):
        do_open(world)


def test_the_set_opens_exactly_once_and_the_opening_is_recorded_in_both_places(world):
    secure(world)
    do_freeze(world)
    path = do_open(world)
    assert path == world["directory"] / ho.TEST_FILE and path.is_file()
    (line,) = world["ledger_path"].read_text().splitlines()
    entry = json.loads(line)
    assert entry == json.loads((world["directory"] / ho.OPENED_MARKER).read_text()) and entry["arms"] == ["D"] and entry["purpose"] == "final evaluation"
    with pytest.raises(ho.HeldoutRefused, match="opened before"):
        do_open(world)


@pytest.mark.parametrize("deleted", ["ledger", "marker"])
def test_deleting_one_of_the_two_records_does_not_reopen_the_set(world, deleted):
    secure(world)
    do_freeze(world)
    do_open(world)
    if deleted == "ledger":
        world["ledger_path"].unlink()
    else:
        (world["directory"] / ho.OPENED_MARKER).unlink()
    with pytest.raises(ho.HeldoutRefused, match="opened before"):
        do_open(world)


# ---- nothing else may read it ------------------------------------------------------------------------------------------------------------

def test_the_runners_guard_refuses_the_folder_and_byte_identical_copies_but_not_other_files(world, tmp_path):
    secure(world)
    args = dict(directory=world["directory"], manifest_path=world["manifest_path"])
    copy = tmp_path / "innocent_name.json"
    copy.write_bytes((world["directory"] / ho.TEST_FILE).read_bytes())
    other = tmp_path / "other.json"
    other.write_text("{}", encoding="utf-8")
    for bad in (world["directory"] / ho.TEST_FILE, world["directory"] / "not-there.json", copy):
        with pytest.raises(ho.HeldoutRefused, match="held-out"):
            ho.refuse_heldout_inputs([other, bad], **args)
    ho.refuse_heldout_inputs([other, tmp_path / "missing.json"], **args)


def test_the_retrieval_runner_refuses_the_held_out_data_as_its_golden_set(monkeypatch, world):
    from eval import run_retrieval_eval as rre

    secure(world)
    monkeypatch.setattr(ho, "data_dir", lambda: world["directory"])
    with pytest.raises(ho.HeldoutRefused, match="held-out"):
        rre.main(["--split", "dev", "--arm", "D", "--golden", str(world["directory"] / ho.TEST_FILE)])


def test_only_the_heldout_module_names_the_test_data():
    """A tripwire: another module that spells out the held-out folder, file or URL would be a way around the gate."""
    needles = (ho.TEST_FILE, ho.ARCHIVE_NAME, "advrag/heldout", "qasper_test_v0.3")
    offenders = []
    for folder in ("eval", "scripts", "rag", "rag_core", "db", "app", "api", "utils", "evaluation"):
        for path in (ROOT / folder).rglob("*.py"):
            if path.name == "heldout.py":
                continue
            text = path.read_text(encoding="utf-8")
            offenders += [f"{path.relative_to(ROOT)}: {n}" for n in needles if n in text]
    assert offenders == []


def test_importing_the_module_does_no_io_and_imports_nothing_heavy():
    tree = ast.parse((ROOT / "eval" / "heldout.py").read_text(encoding="utf-8"))
    top = {(n.module if isinstance(n, ast.ImportFrom) else n.names[0].name).split(".")[0] for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))}
    assert not top & {"chromadb", "torch", "sentence_transformers", "datasets", "huggingface_hub", "requests", "httpx"}


def test_the_repository_has_no_heldout_freeze_or_ledger_until_the_owner_asks():
    assert not ho.FREEZE_FILE.exists() and not ho.LEDGER_FILE.exists()
