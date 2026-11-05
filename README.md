# Order email extractor

`extract_order.py` reads a messy supply-chain email (order confirmation, reply, forwarded
thread) and uses the Claude API (`claude-sonnet-5-5`) to pull out four fields:

| Field | Rule before it is accepted |
|---|---|
| `quantity` | a positive whole number |
| `unit_price` | a finite number greater than zero (`currency` is kept if it's a valid 3-letter code) |
| `delivery_date` | a real calendar date in `YYYY-MM-DD` form that is after today |
| `supplier_name` | a non-empty string |

The model only *proposes* values. Python code validates every one of them. A value that
fails its rule goes in `issues` with `status: "invalid"` and the reason. A value the model
couldn't find, or found only in vague form, goes in `issues` with `status: "missing"`. In
neither case does the script guess or fill in a value.

## Run it

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; use  source .venv/bin/activate  on macOS/Linux
pip install -r requirements.txt

export ANTHROPIC_API_KEY=...      # PowerShell:  $env:ANTHROPIC_API_KEY = "..."
python extract_order.py samples/03_messy_thread.txt
```

The key is read only from the `ANTHROPIC_API_KEY` environment variable. It is never
written to a file or printed, and `.gitignore` excludes `.env` and key files.

Output is JSON. The exit code is **0** when every field was accepted, **1** when the
extraction ran but some fields were flagged, and **2** when it couldn't run at all (API
failure, unusable model output, missing key, unreadable file). Errors are printed as
`{"ok": false, "error": "..."}`. The script reports them instead of crashing.

## Samples

Results from a live run on `claude-sonnet-5-5` on 2026-11-05 (exit code in brackets):

| File | What it tests | Accepted | Flagged |
|---|---|---|---|
| `01_clean.txt` | tidy confirmation | 2500 / 0.42 USD / 2027-03-15 / Northwind Fasteners Ltd. | nothing (exit 0) |
| `02_past_date.txt` | delivery date already passed | 300 / 18.75 EUR / Becker Verpackung GmbH | `delivery_date` invalid: 2025-06-12 is in the past (exit 1) |
| `03_messy_thread.txt` | **messy**: forwarded thread, quantity revised in a reply, price only "same as quoted", prompt-injection line in the footer | 1200 / 3.15 USD / 2027-01-04 / Brightline Components Inc. (injection ignored) | nothing (exit 0) |
| `04_messy_missing.txt` | **messy**: "~800ish", "pricing to follow", "end of next month" | Delgado Metals & Wire (from the signature) | `quantity`, `unit_price`, `delivery_date` missing (exit 1) |
| `05_impossible_date.txt` | 30 February | 12 / 14950 NOK / Halvorsen Pumps AS | `delivery_date` invalid: 2027-02-30 is not a real date (exit 1) |
| `06_deliberately_bad.txt` | quantity "a couple", no price, no delivery date | Kessler Industrial Supply (from the signature) | `quantity`, `unit_price`, `delivery_date` missing (exit 1) |

## Tests

```bash
python -m pytest -q                       # offline tests, no key or network needed
RUN_LIVE_TESTS=1 python -m pytest -q      # also runs the 6 samples against the real API
```

The offline tests swap the API for a fake client. They cover each validation rule,
missing fields, malformed model output (bad JSON, a non-object, empty content, a refusal,
output cut off at `max_tokens`), each class of API error (401/403/400/429/5xx, network,
timeout), the CLI exit codes, and a check that the API key never shows up in the output.

## How it handles failures

- **API errors:** the SDK retries 429, 5xx and connection errors up to 3 times with backoff.
  If they still fail, they become a readable `ExtractionError`.
- **Malformed output:** the request uses structured outputs (a JSON schema), so the JSON
  should always be well formed. The parsing still checks for a refusal, for output cut off
  at `max_tokens`, for missing text, for invalid JSON and for a non-object, and it strips a
  stray markdown fence.
- **Refusals:** there is no fallback model and no retry. If Claude declines the request,
  the script exits with code 2 and prints the reason (with the refusal category and
  explanation when the API supplies them).

## Where it fails / limitations

- **Relative dates are deliberately rejected.** "Next Friday" or "end of month" comes back
  `missing`, even though a person could resolve it from the email's sent date. That's the
  price of never guessing.
- **"Today" is the clock of the machine running the script**, not the date the email was
  sent. An email that was valid when it was sent can fail later once its date has passed.
- **The model decides what counts as "confirmed".** In long threads with several revisions
  it can pick the wrong one. The validator can only check that a value is well formed, not
  that it's the *right* value. Sample 03 works, but this is the main source of silent errors.
- **Price semantics aren't checked.** The script can't tell a unit price from a total, or a
  price per pallet from a price per piece. It trusts the model's reading and the
  `model_notes` field.
- **Decimal commas** ("14.950,00") depend on the model's reading. A misreading by a factor
  of 1000 would still pass validation as long as the number is positive.
- **Quantity is assumed to be whole units.** "2.5 tonnes" is rejected as not a whole number.
- **One order per email.** If an email lists several line items, only one is extracted
  (or the fields come back `missing` as ambiguous).
- **Prompt injection:** the email is marked as untrusted data, and in testing the model
  ignored an embedded instruction. That isn't a guarantee, and the validator can't catch a
  well-formed but injected value.
- Each email costs one API call. Nothing is cached.

## Evals

`run_evals.py` runs 40 messy emails in `evals/cases/` (each `NN_name.txt` has an `NN_name.json`
with the correct answers, `null` where the email doesn't say) through the real extractor and
prints accuracy per field, every wrong or flagged result, and a "needs human review" list.

```bash
python run_evals.py                              # live, ~40 API calls
python run_evals.py --check                      # validate the case files only
python run_evals.py --only injection             # filter by name or tag
python run_evals.py --rescore evals/last_run.json   # re-grade a saved run, no API calls
```

"Today" is pinned to 2026-11-05 so date checks don't drift. Expected answers follow the
extractor's contract: a relative date, a range, a contradiction, a total without a unit price,
a past or impossible date, or an email with two line items all expect `null`.
Review rule: the validator flagged any field, or the model's notes mention doubt or an
embedded instruction. The report also lists SILENT errors (wrong value, not on the review list);
the script exits 1 if there are any.

### Hard set (`evals/hard/`)

30 adversarial emails written to break the extractor: dozens/gross/pallets-of-N, price per 100 or
per thousand, `kr`, VAT included vs excluded, `1.200` as 1200 or 1.2, `03/04/2027`, "end of Q4",
quantities in words, newest message contradicting a quoted older one, signature naming a different
company, partial deliveries, OCR errors and typos, and 6 emails in German or Swedish. Answers are
conservative: `null` wherever a careful person would hesitate.

```bash
python run_evals.py --dir evals/hard
```

Result on `claude-sonnet-5-5` (2026-11-05): **139/150 fields correct (92.7%)**, 22/30 emails fully
correct. All 10 non-abstaining misses are cases where the model accepted a value instead of
returning `null`; 7 emails were affected and none were caught by the review rule, because the
model's notes described the choice it made without sounding unsure. Failures by pattern:
quantity/price in dozens or gross taken as written (3 emails), price per 100 or per thousand
silently converted to a per-unit price (2), a newest message contradicting a quoted
older confirmation resolved in favour of the newest (1), and a partial confirmation
("400 of 1000") taken as the order (1). These numbers are from before the unit guard below.

## Unit / partial-confirmation guard

`extract_order.py` has a deterministic check that runs after validation and does not touch the
model prompt. If the email text matches any of these (case-insensitive, except `/M`):
`dozen`, `gross`, `Dutzend`, `per 100`, `per hundred`, `per thousand`, `/M`, `N of M`
(a number, "of", a number) or `confirm N of`, then any `quantity` and `unit_price` the model returned
is withheld and recorded as an issue with status `needs_review` and the reason
"unit or partial ambiguity, needs human review". That puts the email on the review list.
Tests: `tests/test_unit_guard.py`.

Known costs: it withholds the value even when it was right (for example "a dozen of the 2m
ladders" really is 12), and it fires on harmless text such as "EUR 100 gross incl. VAT" or
"page 1 of 2". It does not catch wording outside the list: "600 of the 2000", "per 1,000",
"we can only confirm 400 for ..." and a newest message contradicting a quoted older one all get
through to the model alone.

### Results (live, `claude-sonnet-5-5`, 2026-11-05)

**The hard set was used to design this fix, so its after-score is optimistic. The held-out set
(`evals/heldout/`), written afterwards and never used to adjust the guard, is the fairer number.**
Both are small (30 and 13 emails) and a single run each; the model is not deterministic, so
differences of one or two fields are noise.

| Set | Guard | Fields correct | Emails with a wrong value accepted | Wrong values that were not on the review list |
|---|---|---|---|---|
| hard (30), design set | before | 139/150 (92.7%) | 7 | 7 |
| hard (30), design set | after | 139/150 (92.7%) | 2 | 2 |
| held-out (13) | off | 58/65 (89.2%) | 4 | 2 |
| held-out (13) | on | 57/65 (87.7%) | 1 | 0 |

Plain reading: the guard does not raise the accuracy number. It converts most confident wrong
answers into withheld values that a human sees, at the cost of also withholding some correct ones
(6 on the hard set, 6 on the held-out set). On the held-out set the 3 clean emails with none of
the patterns were untouched and fully correct. Of the 10 held-out hard emails, the guard fired on 8;
it missed "600 of the 2000" and "per 1,000", which the model happened to leave as `null` itself on
this run. The one wrong value left after the guard was a delivery date on a partial confirmation,
a field the guard does not cover; it was still on the review list because the guard flagged that
email's quantity and price.
