"""Offline tests: the Claude API is replaced by a fake client, so these need no key or network.

A live run over the six sample emails is opt-in: set RUN_LIVE_TESTS=1 (and ANTHROPIC_API_KEY).
"""

import json
import os
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2 as httpx  # the HTTP library the anthropic SDK 2.x depends on
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import extract_order as eo  # noqa: E402

TODAY = date(2026, 10, 31)
SAMPLES = Path(__file__).resolve().parent.parent / "samples"

GOOD = {
    "quantity": 2500,
    "unit_price": 0.42,
    "currency": "USD",
    "delivery_date": "2027-03-15",
    "supplier_name": "Northwind Fasteners Ltd.",
    "notes": "",
}


# ---------------------------------------------------------------- fakes


def response(text=None, stop_reason="end_turn", blocks=None, stop_details=None):
    if blocks is None:
        blocks = [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(stop_reason=stop_reason, content=blocks, stop_details=stop_details)


class FakeClient:
    """Mimics client.messages.create: returns a canned response or raises."""

    def __init__(self, result):
        self.result = result
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def http_error(cls, status):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx.Response(status, request=req), body=None)


def issue_for(result, name):
    return next((i for i in result.issues if i["field"] == name), None)


# ---------------------------------------------------------------- happy path


def test_valid_output_is_fully_accepted():
    r = eo.extract("some email", client=FakeClient(response(json.dumps(GOOD))), today=TODAY)
    assert r.ok
    assert r.accepted == {k: v for k, v in GOOD.items() if k != "notes"}


def test_request_shape():
    fake = FakeClient(response(json.dumps(GOOD)))
    eo.extract("Qty 5", client=fake, today=TODAY)
    call = fake.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    # No refusal fallback: no beta flags, no secondary model.
    assert "fallbacks" not in call and "betas" not in call
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "<email>\nQty 5\n</email>" in call["messages"][0]["content"]


def test_whole_float_quantity_becomes_int():
    r = eo.validate({**GOOD, "quantity": 12.0}, today=TODAY)
    assert r.accepted["quantity"] == 12 and isinstance(r.accepted["quantity"], int)


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize("bad", [0, -5, 12.5, "100", True, [3]])
def test_quantity_must_be_positive_integer(bad):
    r = eo.validate({**GOOD, "quantity": bad}, today=TODAY)
    assert "quantity" not in r.accepted
    assert issue_for(r, "quantity")["status"] == "invalid"


@pytest.mark.parametrize("bad", ["0.42", "$0.42", float("nan"), float("inf"), 0, -1, False])
def test_price_must_be_a_positive_finite_number(bad):
    r = eo.validate({**GOOD, "unit_price": bad}, today=TODAY)
    assert "unit_price" not in r.accepted
    assert issue_for(r, "unit_price")["status"] == "invalid"


@pytest.mark.parametrize(
    "bad, reason_fragment",
    [
        ("2027-02-30", "real calendar date"),
        ("2027-13-01", "real calendar date"),
        ("2025-06-12", "future"),
        ("2026-10-31", "future"),  # today is not the future
        ("15/03/2027", "YYYY-MM-DD"),
        ("next Friday", "YYYY-MM-DD"),
        (20270315, "YYYY-MM-DD"),
    ],
)
def test_date_must_be_real_and_future(bad, reason_fragment):
    r = eo.validate({**GOOD, "delivery_date": bad}, today=TODAY)
    assert "delivery_date" not in r.accepted
    issue = issue_for(r, "delivery_date")
    assert issue["status"] == "invalid" and reason_fragment in issue["reason"]


def test_tomorrow_is_accepted():
    r = eo.validate({**GOOD, "delivery_date": "2026-11-01"}, today=TODAY)
    assert r.accepted["delivery_date"] == "2026-11-01"


@pytest.mark.parametrize("name", ["quantity", "unit_price", "delivery_date", "supplier_name"])
@pytest.mark.parametrize("empty", [None, "", "   ", "<absent>"])
def test_missing_fields_are_flagged_not_guessed(name, empty):
    raw = dict(GOOD)
    if empty == "<absent>":
        del raw[name]
    else:
        raw[name] = empty
    r = eo.validate(raw, today=TODAY)
    assert name not in r.accepted
    assert issue_for(r, name) == {"field": name, "status": "missing", "reason": eo.MISSING_REASON}
    assert not r.ok


def test_all_missing():
    r = eo.validate({}, today=TODAY)
    assert r.accepted == {}
    assert {i["field"] for i in r.issues} == set(eo.REQUIRED_FIELDS)


def test_currency_optional_but_validated():
    assert eo.validate({**GOOD, "currency": None}, today=TODAY).ok
    assert eo.validate({**GOOD, "currency": "usd"}, today=TODAY).accepted["currency"] == "USD"
    r = eo.validate({**GOOD, "currency": "dollars"}, today=TODAY)
    assert issue_for(r, "currency")["status"] == "invalid"


# ---------------------------------------------------------------- malformed model output


@pytest.mark.parametrize(
    "resp, fragment",
    [
        (response("not json at all"), "not valid JSON"),
        (response('{"quantity": 5,'), "not valid JSON"),
        (response("[1, 2, 3]"), "not an object"),
        (response('"just a string"'), "not an object"),
        (response(blocks=[]), "no text"),
        (response(blocks=[SimpleNamespace(type="thinking", thinking="")]), "no text"),
        (response("{}", stop_reason="refusal"), "declined"),
        (response('{"quantity"', stop_reason="max_tokens"), "cut off"),
    ],
)
def test_malformed_output_raises_extraction_error(resp, fragment):
    with pytest.raises(eo.ExtractionError, match=fragment):
        eo.extract("email", client=FakeClient(resp), today=TODAY)


def test_refusal_reason_includes_stop_details_and_calls_once():
    details = SimpleNamespace(type="refusal", category="cyber", explanation="flagged by classifier")
    fake = FakeClient(response("", stop_reason="refusal", stop_details=details))
    with pytest.raises(eo.ExtractionError) as exc:
        eo.extract("email", client=fake, today=TODAY)
    assert str(exc.value) == (
        "model declined the request (stop_reason=refusal, category=cyber): flagged by classifier"
    )
    assert len(fake.calls) == 1  # no retry, no second model


@pytest.mark.parametrize(
    "result, expected_error",
    [
        (response("", stop_reason="refusal"), "model declined the request (stop_reason=refusal)"),
        (http_error(anthropic.InternalServerError, 500), "API error (500) after retries"),
        (http_error(anthropic.AuthenticationError, 401), "API key was rejected (401)"),
    ],
)
def test_cli_refusal_or_api_error_exits_2_and_prints_reason(monkeypatch, capsys, tmp_path, result, expected_error):
    fake = FakeClient(result)
    monkeypatch.setattr(eo, "make_client", lambda: fake)
    email = tmp_path / "e.txt"
    email.write_text("Qty 5", encoding="utf-8")
    assert eo.main([str(email)]) == 2
    assert json.loads(capsys.readouterr().out) == {"ok": False, "error": expected_error}
    assert len(fake.calls) == 1


def test_markdown_fenced_json_is_tolerated():
    text = "```json\n" + json.dumps(GOOD) + "\n```"
    assert eo.extract("email", client=FakeClient(response(text)), today=TODAY).ok


def test_wrong_types_from_model_are_flagged_not_crash():
    raw = {"quantity": {"n": 5}, "unit_price": None, "delivery_date": ["2027-01-01"],
           "supplier_name": 42, "currency": 7, "notes": None}
    r = eo.extract("email", client=FakeClient(response(json.dumps(raw))), today=TODAY)
    assert r.accepted == {}
    assert len(r.issues) == 5


# ---------------------------------------------------------------- API failures


@pytest.mark.parametrize(
    "exc, fragment",
    [
        (http_error(anthropic.AuthenticationError, 401), "rejected"),
        (http_error(anthropic.PermissionDeniedError, 403), "permission"),
        (http_error(anthropic.BadRequestError, 400), "400"),
        (http_error(anthropic.RateLimitError, 429), "rate limited"),
        (http_error(anthropic.InternalServerError, 500), "500"),
        (http_error(anthropic.APIStatusError, 529), "529"),
        (anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), "network"),
        (anthropic.APITimeoutError(request=httpx.Request("POST", "https://x")), "network"),
    ],
)
def test_api_failures_become_extraction_errors(exc, fragment):
    with pytest.raises(eo.ExtractionError, match=fragment):
        eo.extract("email", client=FakeClient(exc), today=TODAY)


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(eo.ExtractionError, match="ANTHROPIC_API_KEY is not set"):
        eo.extract("email", today=TODAY)


def test_empty_email_is_rejected_before_calling_api():
    fake = FakeClient(response(json.dumps(GOOD)))
    with pytest.raises(eo.ExtractionError, match="empty"):
        eo.extract("   \n", client=fake)
    assert fake.calls == []


# ---------------------------------------------------------------- CLI


def test_cli_exit_codes(monkeypatch, capsys, tmp_path):
    email = tmp_path / "e.txt"
    email.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(eo, "extract", lambda text: eo.validate(GOOD, today=TODAY))
    assert eo.main([str(email)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    monkeypatch.setattr(eo, "extract", lambda text: eo.validate({}, today=TODAY))
    assert eo.main([str(email)]) == 1
    capsys.readouterr()

    def boom(text):
        raise eo.ExtractionError("API error (500) after retries")

    monkeypatch.setattr(eo, "extract", boom)
    assert eo.main([str(email)]) == 2
    assert json.loads(capsys.readouterr().out) == {"ok": False, "error": "API error (500) after retries"}

    assert eo.main([str(tmp_path / "nope.txt")]) == 2


def test_api_key_never_appears_in_output(monkeypatch, capsys, tmp_path):
    secret = "sk-ant-test-SECRET-DO-NOT-PRINT"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.setattr(eo, "call_model", lambda text, client: (_ for _ in ()).throw(
        eo.ExtractionError("API key was rejected (401)")))
    email = tmp_path / "e.txt"
    email.write_text("hello", encoding="utf-8")
    eo.main([str(email)])
    out = capsys.readouterr()
    assert secret not in out.out and secret not in out.err


# ---------------------------------------------------------------- live (opt-in)

# What each sample should produce. None = must be flagged, never accepted.
EXPECTED = {
    "01_clean.txt": {"quantity": 2500, "unit_price": 0.42, "delivery_date": "2027-03-15"},
    "02_past_date.txt": {"quantity": 300, "unit_price": 18.75, "delivery_date": None},
    "03_messy_thread.txt": {"quantity": 1200, "unit_price": 3.15, "delivery_date": "2027-01-04"},
    "04_messy_missing.txt": {"quantity": None, "unit_price": None, "delivery_date": None},
    "05_impossible_date.txt": {"quantity": 12, "unit_price": 14950, "delivery_date": None},
    "06_deliberately_bad.txt": {"quantity": None, "unit_price": None, "delivery_date": None},
}


@pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call the real API")
@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_live_sample(name):
    r = eo.extract((SAMPLES / name).read_text(encoding="utf-8"), today=TODAY)
    for fld, want in EXPECTED[name].items():
        if want is None:
            assert fld not in r.accepted, f"{fld} should have been flagged: {r.to_dict()}"
        else:
            assert r.accepted.get(fld) == pytest.approx(want), r.to_dict()
    assert r.accepted.get("supplier_name") or name == "04_messy_missing.txt"
