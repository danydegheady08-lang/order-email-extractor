"""Tests for the deterministic unit / partial-confirmation guard (no network, no key)."""

import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_order as eo  # noqa: E402
from test_extract_order import GOOD, FakeClient, response  # noqa: E402

TODAY = date(2026, 10, 31)
SAMPLES = Path(__file__).resolve().parent.parent / "samples"

# (pattern name the guard should report, an email snippet containing it)
POSITIVE = [
    ("dozen", "Confirming 5 dozen gloves at EUR 24.00 per dozen."),
    ("dozen", "we ship two Dozens of bolts"),  # plural + capital
    ("gross", "2 gross of rivets, USD 0.05 each"),
    ("Dutzend", "3 Dutzend Handschuhe zu 18,00 EUR"),
    ("Dutzend", "zwei dutzend Stück"),
    ("per 100", "Price EUR 12.50 per 100 pcs"),
    ("per hundred", "EUR 9 per hundred pieces"),
    ("per thousand", "USD 38 per thousand labels"),
    ("/M", "20,000 labels, USD 38.00/M"),
    ("N of M", "Lot 400 of 1000 is ready"),
    ("N of M", "we can do 1,200 of 5.000 now"),
    ("confirm N of", "I can confirm 400 of the order for 5 Feb"),
    ("confirm N of", "Confirmed 400 of 1000"),
]

# Emails with none of the patterns must pass through untouched.
NEGATIVE = [
    "Quantity: 2,500 units. Unit price: USD 0.42. Delivery 15 March 2027.",
    "Please confirm the delivery date by Friday.",  # 'confirm' without 'N of'
    "per unit price USD 4, per piece, 1000 pcs",  # 'per' without 100/hundred/thousand
    "Order 4000 pcs, ship 100 pallets",  # numbers without 'of'
    "We sent the dozenth reminder",  # 'dozenth' is not 'dozen'
    "ref. 12/Mar and /Mo schedule",  # '/M' must be a whole token, not '/Mar'
]


def run(email, raw=None):
    return eo.extract(email, client=FakeClient(response(json.dumps(raw or GOOD))), today=TODAY)


@pytest.mark.parametrize("name,snippet", POSITIVE)
def test_pattern_is_detected(name, snippet):
    assert name in eo.unit_patterns_found(snippet)


@pytest.mark.parametrize("name,snippet", POSITIVE)
def test_pattern_flags_quantity_and_price_and_withholds_values(name, snippet):
    r = run(snippet)
    assert not r.ok
    for field in ("quantity", "unit_price"):
        assert field not in r.accepted, f"{field} should be withheld"
        issue = next(i for i in r.issues if i["field"] == field)
        assert issue["reason"] == "unit or partial ambiguity, needs human review"
        assert issue["status"] == "needs_review"


def test_other_fields_are_untouched():
    r = run("2 gross of rivets")
    assert r.accepted["supplier_name"] == GOOD["supplier_name"]
    assert r.accepted["delivery_date"] == GOOD["delivery_date"]
    assert r.accepted["currency"] == "USD"


@pytest.mark.parametrize("snippet", NEGATIVE)
def test_no_pattern_means_no_change(snippet):
    assert eo.unit_patterns_found(snippet) == []
    r = run(snippet)
    assert r.ok
    assert r.accepted["quantity"] == 2500 and r.accepted["unit_price"] == 0.42


def test_only_fields_the_model_returned_are_flagged_by_the_guard():
    raw = dict(GOOD, unit_price=None)  # model already gave no price
    r = run("5 dozen gloves", raw)
    prices = [i for i in r.issues if i["field"] == "unit_price"]
    assert len(prices) == 1 and prices[0]["status"] == "missing"  # not double-reported
    assert any(i["field"] == "quantity" and i["status"] == "needs_review" for i in r.issues)


def test_guard_does_not_touch_a_missing_or_invalid_field_reason():
    raw = dict(GOOD, quantity=-5)  # invalid -> never accepted, so nothing for the guard to withhold
    r = run("2 gross", raw)
    qty = [i for i in r.issues if i["field"] == "quantity"]
    assert len(qty) == 1 and qty[0]["status"] == "invalid"


def test_slash_m_is_case_sensitive():
    assert "/M" in eo.unit_patterns_found("USD 38/M")
    assert "/M" not in eo.unit_patterns_found("USD 38/m per meter")


def test_known_false_positive_is_documented_behaviour():
    # 'gross' also appears as a price qualifier; the guard still fires (deliberate, over-cautious).
    assert "gross" in eo.unit_patterns_found("EUR 100.00 gross incl. VAT")


@pytest.mark.parametrize("path", sorted(SAMPLES.glob("*.txt")), ids=lambda p: p.name)
def test_existing_samples_are_not_affected(path):
    assert eo.unit_patterns_found(path.read_text(encoding="utf-8")) == []
