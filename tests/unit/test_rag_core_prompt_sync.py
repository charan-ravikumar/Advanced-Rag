"""rag_core.prompt_sync: compare and seed with a fake Langfuse client. Idempotent, never overwrites by default, dry run changes nothing."""


from rag_core.prompt_sync import compare_all, compare_prompt, seed
from rag_core.prompts import INSUFFICIENT_CONTEXT, PROMPT_NAMES, local_template, template_sha256
from tests.unit.rag_core_fakes import FakePrompt, chat_messages

LONG, SHORT = PROMPT_NAMES["long"], PROMPT_NAMES["short"]


class NotFoundError(Exception):  # same class name the SDK's API error uses
    pass


class FakeServer:
    """get_prompt / create_prompt over an in-memory prompt table: name -> list of versions."""

    def __init__(self, **initial):
        self.versions = {name: list(v) for name, v in initial.items()}
        self.created, self.fetch_error = [], None

    def get_prompt(self, name, **kw):
        if self.fetch_error:
            raise self.fetch_error
        if not self.versions.get(name):
            raise NotFoundError("not found")
        messages = self.versions[name][-1]
        return FakePrompt(chat_messages(messages[0]["content"], messages[1]["content"]), version=len(self.versions[name]))

    def create_prompt(self, **kw):
        self.created.append(kw)
        self.versions.setdefault(kw["name"], []).append([dict(m) for m in kw["prompt"]])


def server(long=True, short=True):
    return FakeServer(**({LONG: [local_template("long")]} if long else {}), **({SHORT: [local_template("short")]} if short else {}))


def test_identical_prompts_are_reported_identical():
    results = compare_all(server())
    assert [(r.name, r.state) for r in results] == [(LONG, "identical"), (SHORT, "identical")] and results[0].remote_sha256 == results[0].local_sha256


def test_a_missing_prompt_is_reported_missing_and_a_changed_one_differs():
    s = server(long=False)
    s.versions[SHORT] = [[{"role": "system", "content": f"changed {INSUFFICIENT_CONTEXT}"}, local_template("short")[1]]]
    long, short = compare_all(s)
    assert long.state == "missing" and long.remote_sha256 is None
    assert short.state == "differs" and short.remote_sha256 != short.local_sha256 and short.remote_version == 1


def test_a_failed_fetch_is_an_error_not_a_missing_prompt():
    s = server()
    s.fetch_error = ConnectionError("down")
    assert all(r.state == "error" and "ConnectionError" in r.detail for r in compare_all(s))
    s.fetch_error = type("E", (Exception,), {"status_code": 404})("x")  # a 404 by status code is also 'missing'
    assert all(r.state == "missing" for r in compare_all(s))


def test_the_sdk_fallback_and_unreadable_prompts_are_never_taken_for_identical():
    class Fallback(FakeServer):
        def get_prompt(self, name, **kw):
            return FakePrompt(chat_messages("x"), is_fallback=True)

    assert compare_prompt(Fallback(), LONG).state == "error"

    class Unreadable(FakeServer):
        def get_prompt(self, name, **kw):
            return FakePrompt(42)

    assert compare_prompt(Unreadable(), LONG).state == "unreadable"


def test_a_dry_run_changes_nothing_even_when_prompts_are_missing():
    s = server(long=False, short=False)
    results = seed(s, apply=False)
    assert [r.state for r in results] == ["missing", "missing"] and s.created == [] and [r.action for r in results] == [None, None]


def test_apply_creates_only_missing_prompts_as_chat_with_the_production_label_and_the_local_text():
    s = server(long=False)
    results = seed(s, apply=True)
    assert [r.action for r in results] == ["created", None]
    (call,) = s.created
    assert call["name"] == LONG and call["type"] == "chat" and call["labels"] == ["production"] and call["prompt"] == local_template("long")
    assert "seeded from rag_core/prompts.py" in call["commit_message"]


def test_seeding_is_idempotent_a_second_run_creates_nothing():
    s = server(long=False, short=False)
    seed(s, apply=True)
    assert len(s.created) == 2
    again = seed(s, apply=True)
    assert len(s.created) == 2 and [r.state for r in again] == ["identical", "identical"]
    assert [template_sha256(s.versions[n][-1]) for n in (LONG, SHORT)] == [template_sha256(local_template("long")), template_sha256(local_template("short"))]


def test_a_differing_prompt_is_never_overwritten_unless_explicitly_requested_and_old_versions_stay():
    s = server()
    edited = [{"role": "system", "content": f"edited in the UI. {INSUFFICIENT_CONTEXT}"}, local_template("long")[1]]
    s.versions[LONG].append(edited)
    results = seed(s, apply=True)
    assert results[0].state == "differs" and results[0].action == "skipped" and s.created == [] and len(s.versions[LONG]) == 2
    results = seed(s, apply=True, update_differing=True)
    assert results[0].action == "created" and len(s.created) == 1 and len(s.versions[LONG]) == 3  # a NEW version; versions 1 and 2 untouched
    assert s.versions[LONG][1] == edited


def test_errors_are_never_turned_into_creations():
    s = server()
    s.fetch_error = RuntimeError("auth")
    results = seed(s, apply=True, update_differing=True)
    assert s.created == [] and all(r.state == "error" and r.action == "skipped" for r in results)
