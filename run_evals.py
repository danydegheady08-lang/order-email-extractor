"""Evaluation harness for extract_order.py.

Runs every evals/cases/NN_name.txt through the real extractor (live API calls) and
compares the accepted fields with evals/cases/NN_name.json.

Usage:
    python run_evals.py                  # all cases, live
    python run_evals.py --check          # validate the case files only, no API calls
    python run_evals.py --only injection # cases whose name or tags contain the text
    python run_evals.py --rescore evals/last_run.json   # re-grade saved results, no API
    python run_evals.py --workers 8 --out evals/last_run.json
    python run_evals.py --models claude-haiku-5-5,claude-sonnet-5-5,claude-opus-5-5
        # model comparison: evals/cases + evals/hard (live, guard on), one table per model,
        # a comparison table and every case where the models disagree

"Today" is pinned (default 2026-11-05) so past/future date checks don't drift.
extract_order.py is imported as-is and is not modified or patched.

How a field is scored (truth is the JSON file, "accepted" is what the extractor kept):
    correct       accepted == truth (both null counts as correct)
    abstained     truth has a value, extractor returned nothing/flagged it  (safe miss)
    wrong         truth has a value, extractor accepted a DIFFERENT value   (silent error if not reviewed)
    hallucinated  truth is null, extractor accepted a value                  (silent error if not reviewed)

Needs-human-review rule: a case goes on the list if the validator flagged anything
(result.issues) OR the model's notes mention doubt (see DOUBT_PATTERN) or an
injection attempt. The list is about routing, separate from correctness, so the report
also shows how many wrong results the rule caught and how many slipped through.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import extract_order as eo

ROOT = Path(__file__).parent
CASES_DIR = ROOT / "evals" / "cases"
SCORED = ("quantity", "unit_price", "currency", "delivery_date", "supplier_name")
DEFAULT_TODAY = "2026-11-05"

# Phrases in the model's notes that signal real doubt. Deliberately NOT included: "inferred" /
# "assumed" on their own (the model says "supplier inferred from signature" on routine emails)
# and "not confirmed" (it usually just describes a missing field the validator already flags).
DOUBT_PATTERN = re.compile(
    r"unclear|ambiguous|ambiguity|uncertain|not sure|unsure|doubt|conflict|contradict|"
    r"discrepan|inconsisten|guess|possibl[ey]|probabl[ey]|might|may be|could be|"
    r"approximate|vague|cannot determine|can't determine|"
    r"typo|mismatch|not explicitly|role as|"
    r"inject|embedded instruction|override|manipulat|suspicious",
    re.IGNORECASE,
)
# "no contradictions", "without conflict", "not ambiguous" are reassurance, not doubt.
_NEGATED = re.compile(r"\b(?:no|not|without|nor|neither)\b(?:\s+\w+){0,3}?\s*$", re.IGNORECASE)


def doubt_hit(notes: str) -> str | None:
    for m in DOUBT_PATTERN.finditer(notes or ""):
        if not _NEGATED.search(notes[: m.start()][-40:]):
            return m.group(0)
    return None


# ---------------------------------------------------------------- loading


def load_cases(only: str | None = None, directory: Path = CASES_DIR) -> list[dict]:
    cases = []
    for txt in sorted(directory.glob("*.txt")):
        js = txt.with_suffix(".json")
        if not js.exists():
            raise SystemExit(f"missing answer file for {txt.name}")
        truth = json.loads(js.read_text(encoding="utf-8"))
        missing = [f for f in SCORED if f not in truth]
        if missing:
            raise SystemExit(f"{js.name}: missing keys {missing}")
        case = {
            "name": txt.stem,
            "tags": truth.get("tags", []),
            "note": truth.get("note", ""),
            "truth": {f: truth[f] for f in SCORED},
            "text": txt.read_text(encoding="utf-8"),
        }
        if only and only.lower() not in (case["name"] + " " + " ".join(case["tags"])).lower():
            continue
        cases.append(case)
    if not cases:
        raise SystemExit("no cases found")
    return cases


# ---------------------------------------------------------------- scoring


def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^\w\s&]", " ", s.casefold())
    return re.sub(r"\s+", " ", s).strip()


def same(field: str, truth, got) -> bool:
    if truth is None or got is None:
        return truth is None and got is None
    if field in ("quantity", "unit_price"):
        return abs(float(truth) - float(got)) < 1e-9
    if field == "supplier_name":
        return _norm_name(truth) == _norm_name(got)
    return str(truth).strip().upper() == str(got).strip().upper()


def grade_field(field: str, truth, got) -> str:
    if same(field, truth, got):
        return "correct"
    if truth is None:
        return "hallucinated"
    return "abstained" if got is None else "wrong"


def review_reasons(result: eo.Result | None, error: str | None) -> list[str]:
    if error:
        return [f"extractor error: {error}"]
    reasons = [f"validator flagged {i['field']} ({i['status']})" for i in result.issues]
    hit = doubt_hit(result.model_notes)
    if hit:
        reasons.append(f"model notes mention doubt ({hit!r})")
    return reasons


def run_case(case: dict, client, today: date) -> dict:
    out = {"name": case["name"], "tags": case["tags"], "truth": case["truth"], "note": case["note"]}
    try:
        result = eo.extract(case["text"], client=client, today=today)
    except eo.ExtractionError as e:
        out.update(error=str(e), accepted={}, issues=[], model_notes="")
    else:
        out.update(error=None, accepted=result.accepted, issues=result.issues, model_notes=result.model_notes)
    out["grades"] = {f: grade_field(f, case["truth"][f], out["accepted"].get(f)) for f in SCORED}
    out["review"] = review_reasons(None if out["error"] else result, out["error"])
    return out


# ---------------------------------------------------------------- report


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:5.1f}%" if d else "  n/a"


def fmt(v) -> str:
    return "null" if v is None else str(v)


def print_report(rows: list[dict], label: str = "") -> int:
    n = len(rows)
    print(f"\n=== Accuracy per field{label} ({n} emails) ===")
    head = f"{'field':<15}{'correct':>9}{'accuracy':>10}{'abstained':>11}{'wrong':>7}{'halluc.':>9}"
    print(head)
    print("-" * len(head))
    tot = {k: 0 for k in ("correct", "abstained", "wrong", "hallucinated")}
    for f in SCORED:
        c = {k: sum(r["grades"][f] == k for r in rows) for k in tot}
        for k in tot:
            tot[k] += c[k]
        print(f"{f:<15}{c['correct']:>6}/{n:<3}{pct(c['correct'], n):>10}"
              f"{c['abstained']:>11}{c['wrong']:>7}{c['hallucinated']:>9}")
    cells = n * len(SCORED)
    print("-" * len(head))
    print(f"{'ALL FIELDS':<15}{tot['correct']:>6}/{cells:<3}{pct(tot['correct'], cells):>10}"
          f"{tot['abstained']:>11}{tot['wrong']:>7}{tot['hallucinated']:>9}")
    perfect = sum(all(g == "correct" for g in r["grades"].values()) for r in rows)
    print(f"\nEmails with every field correct: {perfect}/{n} ({pct(perfect, n).strip()})")
    print("abstained = real value exists but extractor returned null/flagged (safe miss)")
    print("wrong / halluc. = extractor accepted an incorrect value (dangerous)")

    # per-tag view
    tags = sorted({t for r in rows for t in r["tags"]})
    print("\n=== Accuracy by case type (all fields) ===")
    for t in tags:
        sub = [r for r in rows if t in r["tags"]]
        ok = sum(g == "correct" for r in sub for g in r["grades"].values())
        print(f"{t:<20}{len(sub):>3} emails  {ok:>3}/{len(sub) * len(SCORED):<3} {pct(ok, len(sub) * len(SCORED))}")

    # every wrong or flagged result
    print("\n=== Every wrong or flagged result ===")
    shown = 0
    for r in rows:
        bad = {f: g for f, g in r["grades"].items() if g != "correct"}
        flagged = r["review"] or r["issues"]
        if not bad and not flagged:
            continue
        shown += 1
        print(f"\n{r['name']}  [{', '.join(r['tags'])}]" + (f"  -- {r['note']}" if r["note"] else ""))
        if r["error"]:
            print(f"  ERROR: {r['error']}")
        for f, g in bad.items():
            print(f"  {g.upper():<12} {f}: expected {fmt(r['truth'][f])}, got {fmt(r['accepted'].get(f))}")
        for i in r["issues"]:
            print(f"  flagged      {i['field']} [{i['status']}]: {i['reason'][:90]}")
        if r["model_notes"]:
            print(f"  model notes: {r['model_notes'][:300]}")
    if not shown:
        print("(none)")

    # needs human review
    review = [r for r in rows if r["review"]]
    wrong_rows = [r for r in rows if any(g in ("wrong", "hallucinated") for g in r["grades"].values())]
    silent = [r for r in wrong_rows if not r["review"]]
    print(f"\n=== Needs human review ({len(review)}/{n} emails) ===")
    for r in review:
        status = "has wrong value" if r in wrong_rows else "all accepted values correct"
        print(f"- {r['name']}: {'; '.join(r['review'])}  [{status}]")
    print("\nReview rule: validator flagged any field, OR model notes mention doubt/injection.")
    print(f"Emails with a wrong/hallucinated value: {len(wrong_rows)}; caught by the review rule: "
          f"{len(wrong_rows) - len(silent)}; SILENT (wrong and not routed to review): {len(silent)}")
    for r in silent:
        bad = [f"{f}: expected {fmt(r['truth'][f])}, got {fmt(r['accepted'].get(f))}"
               for f, g in r["grades"].items() if g in ("wrong", "hallucinated")]
        print(f"  SILENT  {r['name']}: " + "; ".join(bad))
    clean_review = [r for r in review if r not in wrong_rows and not r["issues"]]
    print(f"Review load from notes-only triggers (nothing flagged by validator): {len(clean_review)}")
    return 1 if silent else 0


# ---------------------------------------------------------------- model comparison

# USD per million tokens (input, output), from the Anthropic price list. Approximate: ignores
# caching (not used here), batch discounts and Haiku's higher rate above 100K-token prompts.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
COMPARE_DIRS = (CASES_DIR, ROOT / "evals" / "hard")


class _TimedMessages:
    def __init__(self, inner, model: str, stats: dict):
        self._inner, self._model, self._stats = inner, model, stats

    def create(self, **kwargs):
        kwargs["model"] = self._model  # extract_order.py hardcodes its model; override per call here
        t0 = time.perf_counter()
        try:
            resp = self._inner.create(**kwargs)
        finally:
            self._stats["latency"] += time.perf_counter() - t0
        usage = getattr(resp, "usage", None)
        self._stats["in"] += getattr(usage, "input_tokens", 0) or 0
        self._stats["out"] += getattr(usage, "output_tokens", 0) or 0
        return resp


class TimedClient:
    """Wraps a real client: forces a model, times calls and sums token usage. Nothing else changes."""

    def __init__(self, client, model: str):
        self.stats = {"latency": 0.0, "in": 0, "out": 0}
        self.messages = _TimedMessages(client.messages, model, self.stats)


def run_case_timed(case: dict, client, model: str, today: date) -> dict:
    tc = TimedClient(client, model)
    row = run_case(case, tc, today)
    row["latency_s"] = tc.stats["latency"]
    row["input_tokens"], row["output_tokens"] = tc.stats["in"], tc.stats["out"]
    return row


def row_cost(model: str, row: dict) -> float | None:
    if model not in PRICES:
        return None
    pin, pout = PRICES[model]
    return (row["input_tokens"] * pin + row["output_tokens"] * pout) / 1e6


def model_summary(model: str, rows: list[dict]) -> dict:
    n = len(rows)
    wrong_rows = [r for r in rows if any(g in ("wrong", "hallucinated") for g in r["grades"].values())]
    costs = [row_cost(model, r) for r in rows]
    return {
        "model": model,
        "emails": n,
        "correct": sum(g == "correct" for r in rows for g in r["grades"].values()),
        "cells": n * len(SCORED),
        "silent": sum(1 for r in wrong_rows if not r["review"]),
        "review": sum(1 for r in rows if r["review"]),
        "errors": sum(1 for r in rows if r["error"]),
        "latency": sum(r["latency_s"] for r in rows) / n,
        "tokens_in": sum(r["input_tokens"] for r in rows),
        "tokens_out": sum(r["output_tokens"] for r in rows),
        "cost": None if None in costs else sum(costs),
    }


def print_model_table(model: str, rows: list[dict]) -> None:
    n = len(rows)
    print(f"\n=== {model}: accuracy per field ({n} emails, guard on) ===")
    head = f"{'field':<15}{'correct':>9}{'accuracy':>10}{'abstained':>11}{'wrong':>7}{'halluc.':>9}"
    print(head)
    print("-" * len(head))
    tot = {k: 0 for k in ("correct", "abstained", "wrong", "hallucinated")}
    for f in SCORED:
        c = {k: sum(r["grades"][f] == k for r in rows) for k in tot}
        for k in tot:
            tot[k] += c[k]
        print(f"{f:<15}{c['correct']:>6}/{n:<3}{pct(c['correct'], n):>10}"
              f"{c['abstained']:>11}{c['wrong']:>7}{c['hallucinated']:>9}")
    print("-" * len(head))
    cells = n * len(SCORED)
    print(f"{'ALL FIELDS':<15}{tot['correct']:>6}/{cells:<3}{pct(tot['correct'], cells):>10}"
          f"{tot['abstained']:>11}{tot['wrong']:>7}{tot['hallucinated']:>9}")
    errs = [r for r in rows if r["error"]]
    if errs:
        print(f"Extractor errors: {len(errs)} (first: {errs[0]['name']}: {errs[0]['error'][:120]})")
    for r in rows:
        if any(g in ("wrong", "hallucinated") for g in r["grades"].values()) and not r["review"]:
            bad = [f"{f}: expected {fmt(r['truth'][f])}, got {fmt(r['accepted'].get(f))}"
                   for f, g in r["grades"].items() if g in ("wrong", "hallucinated")]
            print(f"  SILENT  {r['name']}: " + "; ".join(bad))


def print_comparison(summaries: list[dict]) -> None:
    print("\n=== Model comparison ===")
    head = (f"{'model':<20}{'fields correct':>17}{'silent errors':>15}{'review list':>13}"
            f"{'avg latency':>13}{'tokens in/out':>16}{'approx cost':>13}")
    print(head)
    print("-" * len(head))
    for s in summaries:
        cost = "n/a" if s["cost"] is None else f"${s['cost']:.4f}"
        print(f"{s['model']:<20}{s['correct']:>8}/{s['cells']:<4}{pct(s['correct'], s['cells']):>5}"
              f"{s['silent']:>15}{s['review']:>9}/{s['emails']:<3}{s['latency']:>11.2f}s"
              f"{s['tokens_in']:>9}/{s['tokens_out']:<6}{cost:>13}")
    print("silent errors = emails with a wrong/hallucinated value that are NOT on the review list")
    print("cost = tokens x list price (haiku $0.10/$0.50, sonnet $2/$10, opus $4/$20 per 1M in/out); approximate")
    for s in summaries:
        if s["errors"]:
            print(f"WARNING: {s['model']} had {s['errors']} extractor errors (counted as abstained/review)")


def print_disagreements(models: list[str], by_model: dict[str, dict[str, dict]]) -> None:
    names = list(by_model[models[0]])
    out = []
    for name in names:
        rows = {m: by_model[m][name] for m in models}
        diffs = []
        for f in SCORED:
            vals = {m: rows[m]["accepted"].get(f) for m in models}
            first = vals[models[0]]
            if any(not same(f, first, v) for v in vals.values()):
                diffs.append((f, vals))
        review = {m: bool(rows[m]["review"]) for m in models}
        if diffs or len(set(review.values())) > 1:
            out.append((name, rows, diffs, review))
    print(f"\n=== Cases where models disagree ({len(out)}/{len(names)}) ===")
    print("(accepted value differs on any field, or review-list membership differs)")
    for name, rows, diffs, review in out:
        r0 = rows[models[0]]
        print(f"\n{name}  [{', '.join(r0['tags'])}]")
        for f, vals in diffs:
            parts = []
            for m in models:
                g = rows[m]["grades"][f]
                tag = {"correct": "ok", "abstained": "abstained"}.get(g, "WRONG")
                parts.append(f"{m.replace('claude-', '')}={fmt(vals[m])} [{tag}]")
            print(f"  {f}: expected {fmt(r0['truth'][f])} | " + " | ".join(parts))
        if len(set(review.values())) > 1:
            print("  review list: " + ", ".join(
                f"{m.replace('claude-', '')}={'yes' if v else 'no'}" for m, v in review.items()))
    if not out:
        print("(none)")


def compare_models(models: list[str], args, today: date) -> int:
    cases = []
    for d in COMPARE_DIRS:
        for c in load_cases(args.only, d):
            c["name"] = f"{d.name}/{c['name']}"
            cases.append(c)
    try:
        client = eo.make_client()
    except eo.ExtractionError as e:
        print(f"cannot start: {e}", file=sys.stderr)
        return 2
    print(f"Comparing {len(models)} models on {len(cases)} emails each "
          f"({', '.join(d.name for d in COMPARE_DIRS)}), live, unit/partial guard on, today pinned to {today}")
    summaries, by_model = [], {}
    for m in models:
        print(f"\nRunning {m} ...", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            rows = list(pool.map(lambda c: run_case_timed(c, client, m, today), cases))
        by_model[m] = {r["name"]: r for r in rows}
        out = ROOT / "evals" / f"compare_{m}.json"
        out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print_model_table(m, rows)
        print(f"Raw results saved to {out}")
        summaries.append(model_summary(m, rows))
    print_comparison(summaries)
    print_disagreements(models, by_model)
    return 0


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the order-extraction evals.")
    ap.add_argument("--dir", default=str(CASES_DIR), help="case directory (default evals/cases; try evals/hard)")
    ap.add_argument("--only", help="run only cases whose name or tags contain this text")
    ap.add_argument("--today", default=DEFAULT_TODAY, help="pinned 'today' for date checks (YYYY-MM-DD)")
    ap.add_argument("--workers", type=int, default=5, help="parallel API calls (default 5)")
    ap.add_argument("--out", help="where to save raw results (default evals/last_run[_<dir>].json)")
    ap.add_argument("--rescore", metavar="JSON", help="re-grade a saved run (no API calls), e.g. after editing the review rule")
    ap.add_argument("--models", help="comma-separated model ids; runs evals/cases + evals/hard on each and compares")
    ap.add_argument("--check", action="store_true", help="validate case files only; no API calls")
    args = ap.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        if not models:
            raise SystemExit("--models needs at least one model id")
        return compare_models(models, args, date.fromisoformat(args.today))
    case_dir = Path(args.dir)
    label = "" if case_dir.name == CASES_DIR.name else f" - {case_dir.name} set"
    cases = load_cases(args.only, case_dir)
    if args.check:
        print(f"OK: {len(cases)} cases, all with answer files and all {len(SCORED)} fields present.")
        return 0

    if args.rescore:
        rows = json.loads(Path(args.rescore).read_text(encoding="utf-8"))
        truths = {c["name"]: c["truth"] for c in cases}
        rows = [r for r in rows if r["name"] in truths]
        for r in rows:
            r["truth"] = truths[r["name"]]  # re-read answers, so label fixes apply without re-running
            r["grades"] = {f: grade_field(f, r["truth"][f], r["accepted"].get(f)) for f in SCORED}
            res =None if r["error"] else eo.Result(accepted=r["accepted"], issues=r["issues"], model_notes=r["model_notes"])
            r["review"] = review_reasons(res, r["error"])
        return print_report(rows, label)

    today = date.fromisoformat(args.today)
    try:
        client = eo.make_client()
    except eo.ExtractionError as e:
        print(f"cannot start: {e}", file=sys.stderr)
        return 2

    print(f"Running {len(cases)} emails live on {eo.MODEL} (today pinned to {today}) ...")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(lambda c: run_case(c, client, today), cases))

    out = Path(args.out) if args.out else ROOT / "evals" / f"last_run{'_' + case_dir.name if label else ''}.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    code = print_report(rows, label)
    print(f"\nRaw results saved to {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
