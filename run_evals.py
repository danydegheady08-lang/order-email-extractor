"""Evaluation harness for extract_order.py.

Runs every evals/cases/NN_name.txt through the real extractor (live API calls) and
compares the accepted fields with evals/cases/NN_name.json.

Usage:
    python run_evals.py                  # all cases, live
    python run_evals.py --check          # validate the case files only, no API calls
    python run_evals.py --only injection # cases whose name or tags contain the text
    python run_evals.py --rescore evals/last_run.json   # re-grade saved results, no API
    python run_evals.py --workers 8 --out evals/last_run.json

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


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the order-extraction evals.")
    ap.add_argument("--dir", default=str(CASES_DIR), help="case directory (default evals/cases; try evals/hard)")
    ap.add_argument("--only", help="run only cases whose name or tags contain this text")
    ap.add_argument("--today", default=DEFAULT_TODAY, help="pinned 'today' for date checks (YYYY-MM-DD)")
    ap.add_argument("--workers", type=int, default=5, help="parallel API calls (default 5)")
    ap.add_argument("--out", help="where to save raw results (default evals/last_run[_<dir>].json)")
    ap.add_argument("--rescore", metavar="JSON", help="re-grade a saved run (no API calls), e.g. after editing the review rule")
    ap.add_argument("--check", action="store_true", help="validate case files only; no API calls")
    args = ap.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
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
