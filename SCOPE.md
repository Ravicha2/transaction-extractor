# Bank Screenshot → Categorized Transactions (a needle3 test-drive)

## What this is

A weekend, fully-local, **measured test of needle3** (Cactus-Compute, 121M, Apache-2.0) on the
task I actually do by hand: turning bank-app screenshots into categorized transaction records.

This is a model trial, not a product. The deliverable is numbers, not a tool.
Original "receipt photo" framing was dropped during scoping: I don't keep paper receipts;
the corpus I have and the pain I feel are bank screenshots.

## Corpus & labels (freeze before any pipeline code)

- ~30 bank-app screenshots (one bank, same phone). List views; multiple transactions per screenshot.
- Capture timestamp per screenshot (file creation time / sidecar) — needed to resolve sticky
  date headers ("Today", "Yesterday", "Tue 15 Sep").
- Labels: hand-labeled by me, cross-checked against my bank transaction history.
- Category enum — **frozen 2026-09-20** in `categories.json` (source of truth), never changed after:
  Misc and souvenir, health & fitness, grocery&eat out, deposit, tuition fee, transport.
  Amended same day to match the categories actually used in `ground_truth.csv` (verbatim strings).

## Schema

A screenshot yields a **list of records**:

- `vendor`  — passthrough, as printed (bank descriptor). Extracted for my records, **never scored**.
- `date`    — nullable. Resolved from sticky headers + capture timestamp. If unresolvable,
  the correct answer is `null` — never a guess.
- `amount`  — exact match.
- `category`— one of my frozen enum. This is the decision needle3 must make; it's the field
  only a model can win (regex can't do it).

## Pipeline (all local, no cloud, ever)

1. **OCR** — Tesseract on the screenshot (clean UI text; also logs word-error contribution).
2. **needle3 extraction** — `pip install cactus-needle`; grammar-constrained decode over the
   schema above (array of records); OCR text + capture timestamp as context.
3. **Confidence** — record needle3's native calibrated confidence per field; no gate logic in v1,
   measurement only.
4. **Control arm** — 5-line regex baseline for date/amount, so "needle3 vs regex" is an honest headline.

## Eval & pass bars

- Record matching: extracted records matched to labels by (amount, date); extra/missing records count as errors.
- Zero-shot field accuracy: **date ≥90%, amount ≥90%, category ≥85%**.
- **Calibration check**: does needle3's confidence separate correct from incorrect fields?
  If yes, a future confidence gate is real; if no, the score is decoration. This is the most
  interesting possible finding.
- Stretch (optional, only if time left): `open-jev` / `jev-schema-scorer` as a category-only
  comparison arm (choice-over-enum is exactly Jev's primitive; both repro models are OOD for
  this — treat numbers accordingly).
- Output: `REPORT.md` with per-field accuracy, per-cohort notes, calibration table, and the
  needle3-vs-regex verdict.

## Explicitly out of scope v1

- LoRA fine-tuning (`needle build`) — the natural *second* weekend, decided by this one's numbers
- Paper receipts, email/PDF receipts (Docling), canonical vendor renaming
- `tax`/GST, line items, spreadsheet/CSV integration, confidence-gate UX, any auto-action

## Hygiene

- Screenshots contain my account activity: `data/` stays gitignored and local. Never pushed.

## Success check

30 labeled screenshots, frozen before code. If bars are met → needle3 stays, schedule the
fine-tune weekend. If not → the failure breakdown (OCR, count, field attach, category confusion)
is the result, and it decides whether fine-tuning is worth a weekend at all.
