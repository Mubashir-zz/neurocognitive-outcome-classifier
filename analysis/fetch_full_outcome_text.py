#!/usr/bin/env python3
"""
Fetch the complete outcome-measure text for every ClinicalTrials.gov trial in the
training population.

The `Outcome text (for cognition check)` column in the labelled dataset is
truncated at a median of 400 characters, against a median of ~1,850 in the
registry, and frequently cuts off the outcome that names the cognitive
instrument. Any comparison between a text model and a keyword rule computed on
that column is measuring truncation as much as it is measuring the models.

This pulls the untruncated text so the comparison can be redone honestly.
Trials registered only in international registries (ChiCTR, EU-CTR, JPRN, CTRI,
ANZCTR) have no ClinicalTrials.gov record and are necessarily excluded.

    python analysis/fetch_full_outcome_text.py

Writes data/derived/ctgov_full_outcome_text.csv and caches the raw responses so
re-runs need no network.
"""

import csv
import json
import os
import sys
import time
from urllib import error, parse, request

SRC = "data/CLASSIFIER_training_population.csv"
CACHE = "data/derived/ctgov_cache.json"
OUT = "data/derived/ctgov_full_outcome_text.csv"
API = "https://clinicaltrials.gov/api/v2/studies"


def outcome_text(nct):
    url = f"{API}/{parse.quote(nct)}?fields=protocolSection.outcomesModule"
    for attempt in range(3):
        try:
            with request.urlopen(url, timeout=60) as r:
                data = json.load(r)
            break
        except error.HTTPError as exc:
            if exc.code == 404:
                return ""
            if attempt == 2:
                return None
            time.sleep(3 * (attempt + 1))
        except Exception:
            if attempt == 2:
                return None
            time.sleep(3 * (attempt + 1))
    module = data.get("protocolSection", {}).get("outcomesModule", {})
    parts = []
    for key in ("primaryOutcomes", "secondaryOutcomes", "otherOutcomes"):
        for o in module.get(key, []):
            parts.append(" ".join(filter(None, [o.get("measure"), o.get("description")])))
    return " | ".join(p for p in parts if p)


def main():
    with open(SRC, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))

    ncts = sorted({(r["NCT/TrialID"] or "").strip() for r in rows
                   if (r["NCT/TrialID"] or "").strip().upper().startswith("NCT")})
    print(f"{len(rows)} training trials; {len(ncts)} unique ClinicalTrials.gov IDs "
          f"({len(rows) - sum(1 for r in rows if (r['NCT/TrialID'] or '').strip().upper().startswith('NCT'))} "
          f"international-only, not retrievable)")

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

    todo = [n for n in ncts if n not in cache]
    print(f"fetching {len(todo)} ({len(ncts) - len(todo)} cached)")
    for i, nct in enumerate(todo, 1):
        cache[nct] = outcome_text(nct)
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}", end="\r", flush=True)
            json.dump(cache, open(CACHE, "w"))
        time.sleep(0.12)
    json.dump(cache, open(CACHE, "w"))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial_id", "full_outcome_text", "n_chars"])
        n_ok = 0
        for nct in ncts:
            text = cache.get(nct)
            if text:
                w.writerow([nct, text, len(text)])
                n_ok += 1
    print(f"\nwrote {OUT}: {n_ok} trials with retrievable outcome text")


if __name__ == "__main__":
    main()
