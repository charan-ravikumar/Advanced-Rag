"""rag_core.claims: the response schema, quote location, parsing (complete, partial, malformed) and the streaming renderer. Pure, no provider."""

import json
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rag_core.claims import (
    MAX_CLAIMS,
    MIN_QUOTE_WORDS,
    STATUS_ANSWERED,
    STATUS_INSUFFICIENT,
    StreamRenderer,
    answer_schema,
    locate_quote,
    parse_answer,
    render_answer,
)
from rag_core.prompts import escape_text
from tests.unit.rag_core_fakes import answer_json

S1 = SimpleNamespace(label="S1", text="We fine-tune BERT on SQuAD 2.0\nand report  F1 & exact match <EM> on the dev set.")
S2 = SimpleNamespace(label="S2", text="The model has 110M parameters.")
SOURCES = [S1, S2]


# ---- the schema -----------------------------------------------------------------------------------------------------------------------

def test_the_schema_allows_exactly_the_sent_labels_and_requires_status_claims_text_citations_source_and_quote():
    schema = answer_schema(["S1", "S2"])
    claim = schema["properties"]["claims"]["items"]
    cite = claim["properties"]["citations"]["items"]
    assert schema["required"] == ["status", "claims"] and schema["properties"]["status"]["enum"] == [STATUS_ANSWERED, STATUS_INSUFFICIENT]
    assert list(schema["properties"]) == ["status", "claims"]  # status first: the stream knows early whether the model abstains
    assert claim["required"] == ["text", "citations"] and claim["properties"]["citations"]["minItems"] == 1
    assert cite["required"] == ["source", "quote"] and cite["properties"]["source"]["enum"] == ["S1", "S2"]
    assert schema["properties"]["claims"]["maxItems"] == MAX_CLAIMS
    json.dumps(schema)  # plain JSON, as the provider needs it


def test_no_schema_without_sources():
    with pytest.raises(ValueError):
        answer_schema([])


# ---- quote location ---------------------------------------------------------------------------------------------------------------------

def test_a_quote_is_found_across_newlines_and_repeated_spaces_and_its_offsets_point_into_the_source():
    start, end = locate_quote("SQuAD 2.0 and report F1", S1.text)
    assert S1.text[start:end] == "SQuAD 2.0\nand report  F1"


def test_a_quote_of_the_escaped_text_the_model_saw_is_found_in_the_raw_source():
    seen = escape_text(S1.text)  # the prompt escapes & < >
    assert "&amp;" in seen and "&lt;EM&gt;" in seen
    start, end = locate_quote("F1 &amp; exact match &lt;EM&gt;", S1.text)
    assert S1.text[start:end] == "F1 & exact match <EM>"


@pytest.mark.parametrize("quote", ["BERT", "fine-tune BERT", "", "   ", "we fine-tune BERT on SQuAD", "fine-tune BERT ... SQuAD", "BERT on SQuAD 3.0"])
def test_short_changed_or_stitched_quotes_are_not_found(quote):
    assert locate_quote(quote, S1.text) is None  # too short (< MIN_QUOTE_WORDS words), case changed, elided or altered


@given(words=st.lists(st.text(alphabet="abcXYZ019.,-", min_size=1, max_size=6), min_size=MIN_QUOTE_WORDS, max_size=25),
       seps=st.lists(st.sampled_from([" ", "  ", "\n", "\t ", " \n "]), min_size=25, max_size=25), data=st.data())
def test_any_run_of_whole_words_from_a_source_is_found_and_maps_back_to_the_same_words(words, seps, data):
    text = "".join(w + s for w, s in zip(words, seps))
    i = data.draw(st.integers(0, len(words) - MIN_QUOTE_WORDS))
    j = data.draw(st.integers(i + MIN_QUOTE_WORDS, len(words)))
    quote = " ".join(words[i:j])
    found = locate_quote(quote, text)
    assert found is not None
    start, end = found
    assert " ".join(text[start:end].split()) == quote and 0 <= start < end <= len(text)


# ---- parsing ----------------------------------------------------------------------------------------------------------------------------

def test_a_complete_answer_is_parsed_with_verified_spans_valid_and_invalid_labels():
    reply = answer_json(("They fine-tune BERT.", [("S1", "We fine-tune BERT on SQuAD")]), ("It is small.", [("S2", "has 110M parameters"), ("S7", "a b c")]))
    p = parse_answer(reply, SOURCES)
    assert p.complete and p.status == STATUS_ANSWERED and not p.abstained and p.cited == ["S1", "S2"] and p.invalid == ["S7"]
    assert [c.text for c in p.claims] == ["They fine-tune BERT.", "It is small."] and p.quotes == (2, 3)
    first = p.claims[0].citations[0]
    assert first.verified and S1.text[first.start:first.end] == "We fine-tune BERT on SQuAD"
    assert render_answer(p.claims) == "They fine-tune BERT. [S1] It is small. [S2][S7]"


def test_a_label_cited_twice_in_one_claim_is_rendered_once():
    p = parse_answer(answer_json(("Claim.", [("S1", "We fine-tune BERT on SQuAD"), ("S1", "report F1 & exact match")])), SOURCES)
    assert render_answer(p.claims) == "Claim. [S1]" and p.quotes == (2, 2)


@pytest.mark.parametrize("text", ["not json at all", "[1, 2]", '"a string"', "", '{"answer": "free text"}'])
def test_text_that_is_not_a_structured_answer_has_no_status_and_no_claims(text):
    p = parse_answer(text, SOURCES)
    assert p.status is None and p.claims == [] and not p.abstained


def test_an_unknown_status_is_not_trusted():
    p = parse_answer(json.dumps({"status": "maybe", "claims": []}), SOURCES)
    assert p.status is None and not p.abstained


def test_truncated_json_keeps_the_complete_claims_and_the_started_one():
    reply = answer_json(("One.", ["S1"]), ("Two is cut", ["S2"]))
    p = parse_answer(reply[: reply.index("cut") + 3], SOURCES)
    assert not p.complete and [c.text for c in p.claims] == ["One.", "Two is cut"]


def test_malformed_items_are_skipped_and_counted():
    reply = json.dumps({"status": "answered", "claims": ["x", {"text": 5}, {"text": "Ok.", "citations": [{"source": 1, "quote": "q"}, "y",
                                                                                                         {"source": "S2", "quote": "has 110M parameters"}]}]})
    p = parse_answer(reply, SOURCES)
    assert [c.text for c in p.claims] == ["Ok."] and [x.source for x in p.claims[0].citations] == ["S2"] and p.malformed_items == 4


# ---- streaming --------------------------------------------------------------------------------------------------------------------------

def stream(reply: str, size: int) -> tuple[str, StreamRenderer]:
    r = StreamRenderer()
    return "".join(r.feed(reply[i:i + size]) for i in range(0, len(reply), size)), r


@pytest.mark.parametrize("size", [1, 2, 5, 13, 1000])
def test_the_streamed_display_is_a_prefix_of_the_final_answer_for_any_chunking(size):
    reply = answer_json(("Claim one with é and \"quotes\".", ["S1"]), ("Claim two.", ["S1", "S2"]), ("Three.", ["S2"]))
    shown, r = stream(reply, size)
    final = render_answer(parse_answer(reply, SOURCES).claims)
    assert final.startswith(shown) and shown.startswith("Claim one") and shown + r.finish(final) == final


def test_an_abstention_is_never_shown_while_streaming():
    shown, _ = stream(answer_json(("Speculation.", ["S1"]), status=STATUS_INSUFFICIENT), 3)
    assert shown == ""


def test_nothing_is_shown_before_the_status_is_known():
    r = StreamRenderer()
    assert r.feed('{"claims": [{"text": "Early text') == "" and r.feed('", "citations": []}], "status": "answered"}') != ""
