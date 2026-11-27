#!/usr/bin/env python3
"""
PlanetariUM-ML: Horizons ephemeris downloader and dataset builder
=================================================================
Downloads heliocentric positions (ICRF frame, AU and AU/day) from JPL
Horizons: x, y, z, vx, vy, vz for the 16 bodies of the experiment:

  8 planets + Pluto + Eris + Haumea + Makemake + Ceres + Vesta + Pallas + Meliboea

Windows (identical for all 16 bodies):
  TRAIN: 1975-01-01 -> 2024-12-31  (exactly 50 years; TNOs still cover <25% of one
         orbit, but it is the uniform maximum window, declared in the paper)
  TEST:  2025-01-01 -> 2026-11-30  (temporal hold-out, NEVER used in training)

Cadence: 1h for ALL bodies (uniform sampling: the data must reflect what
actually happened over those years, slow movers included).

Note on volume: ~438k rows per body in train + ~17k in test, ~16 bodies
=> ~7.3M rows total, ~870MB merged CSV. Assumed and accepted.

Usage:
  python planetarium_fetch.py download                 # everyone (slow)
  python planetarium_fetch.py download --bodies mercury,ceres,eris
  python planetarium_fetch.py merge                    # build the big CSV
  python planetarium_fetch.py status                   # what is downloaded so far

Output:
  data/raw/<body>/<phase>_<chunk>.csv   (cached chunks, resumable)
  data/planetarium_ml_dataset.csv       (merged dataset, ~7M rows, ~800MB)

Declared reference for the paper: JPL Horizons (ssd.jpl.nasa.gov), ICRF
frame, heliocentric center 500@10, units AU and AU/day. Always write
"validated against JPL Horizons ephemerides", never "observed real position".
"""

import argparse
import io
import os
import sys
import time

import pandas as pd
import requests

API = "https://ssd.jpl.nasa.gov/api/horizons.api"

TRAIN_DEFAULT = ("1975-01-01 00:00", "2024-12-31 23:00")
TEST_WINDOW = ("2025-01-01 00:00", "2026-11-30 00:00")

# Step size is uniform (1h) for the whole set; all bodies share the exact
# same training window (2000-2024) by design decision.
# ALL planets are taken at their SYSTEM BARYCENTER (major bodies 1-8):
# 1=Mercury, 2=Venus, 3=Earth-Moon, 4=Mars, 5=Jupiter, 6=Saturn, 7=Uranus,
# 8=Neptune, 9=Pluto system. Consistent dynamical objects across the whole set.
BODIES = {
    "mercury":   dict(hid="1",      step="1h"),
    "venus":     dict(hid="2",      step="1h"),
    "earth":     dict(hid="3",      step="1h"),   # Earth-Moon barycenter
    "mars":      dict(hid="4",      step="1h"),
    "jupiter":   dict(hid="5",      step="1h"),   # Jupiter system barycenter
    "saturn":    dict(hid="6",      step="1h"),   # Saturn system barycenter
    "uranus":    dict(hid="7",      step="1h"),
    "neptune":   dict(hid="8",      step="1h"),
    # 9 = Pluto system barycenter, same convention as the planets above
    "pluto":     dict(hid="9",      step="1h"),
    "eris":      dict(hid="136199;", step="1h"),
    "haumea":    dict(hid="136108;", step="1h"),
    "makemake":  dict(hid="136472;", step="1h"),
    # Asteroids: the ';' disambiguates from major bodies in the COMMAND field
    "ceres":     dict(hid="1;",      step="1h"),
    "vesta":     dict(hid="4;",      step="1h"),
    "pallas":    dict(hid="2;",      step="1h"),
    "meliboea":  dict(hid="137;",    step="1h"),
}

RAW = "data/raw"
MERGED = "data/planetarium_ml_dataset.csv"
HEADERS = {"User-Agent": "PlanetariUM-ML/0.1 (academic research; mfrutosp)"}


def year_chunks(start, stop):
    """Split [start, stop] into one-year pieces: small, resumable,
    server-friendly requests."""
    y0 = int(start[:4])
    y1 = int(stop[:4])
    out = []
    for y in range(y0, y1 + 1):
        s = start if y == y0 else f"{y}-01-01 00:00"
        e = stop if y == y1 else f"{y}-12-31 23:59"
        out.append((f"{y}", s, e))
    return out


def fetch_chunk(hid, start, stop, step, max_retries=5):
    """One Horizons API request: heliocentric state vectors in CSV format."""
    params = {
        "format": "text",
        "COMMAND": f"'{hid}'",
        "EPHEM_TYPE": "VECTORS",
        "CENTER": "'500@10'",        # Sun (heliocentric)
        "START_TIME": f"'{start}'",
        "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": f"'{step}'",
        "REF_PLANE": "'FRAME'",      # ICRF
        "REF_SYSTEM": "'ICRF'",
        "OUT_UNITS": "'AU-D'",       # positions in AU, velocities in AU/day
        "CSV_FORMAT": "'YES'",
        "VEC_LABELS": "'YES'",
    }
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=300)
            if r.status_code == 429:  # rate limited: wait and retry
                wait = 60 * attempt
                print(f"    429 rate-limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            if "$$SOE" not in r.text or "$$EOE" not in r.text:
                # Horizons returns errors as HTTP 200 with plain text: show and retry
                msg = [l for l in r.text.splitlines() if "ERROR" in l.upper()]
                raise RuntimeError(f"response without data: {msg[:3]}")
            block = r.text.split("$$SOE")[1].split("$$EOE")[0].strip()
            lines = [l for l in block.splitlines() if l.strip()]
            # Keep only data rows: a Julian Day number in the first field.
            # (Horizons CSV header wording varies, so we parse by POSITION:
            # col0=JD, col1=date, col2..7 = x,y,z,vx,vy,vz, then LT,RG,RR)
            data_lines = [l for l in lines
                          if l.split(",")[0].strip().replace(".", "").isdigit()]
            if not data_lines:
                raise RuntimeError(f"no data rows; first lines: {lines[:3]}")
            df = pd.read_csv(io.StringIO("\n".join(data_lines)), header=None)
            df = df.iloc[:, :11]
            df.columns = ["jd", "date", "x", "y", "z", "vx", "vy", "vz",
                          "lt", "rg", "rr"]
            df = df[["jd", "date", "x", "y", "z", "vx", "vy", "vz"]]
            return df
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = 15 * attempt
            print(f"    failed ({e}); retry {attempt+1}/{max_retries} in {wait}s")
            time.sleep(wait)


def body_plan(name):
    """List of chunks (phase, label, start, stop, step) for one body."""
    spec = BODIES[name]
    plan = []
    tr_start = spec.get("train_start", TRAIN_DEFAULT[0])
    for label, s, e in year_chunks(tr_start, TRAIN_DEFAULT[1]):
        plan.append(("train", label, s, e, spec["step"]))
    te_start, te_stop = TEST_WINDOW
    for label, s, e in year_chunks(te_start, te_stop):
        plan.append(("test", label, s, e, spec["step"]))
    return plan


def chunk_path(name, phase, label):
    return os.path.join(RAW, name, f"{phase}_{label}.csv")


def download(names, sleep_between=2.0):
    """sleep_between: pause between requests, to be polite with JPL's server
    (they ask for moderate use; 2s by default). Raise it if you ever see 429s."""
    for name in names:
        os.makedirs(os.path.join(RAW, name), exist_ok=True)
        print(f"[{name}] hid={BODIES[name]['hid']}")
        for phase, label, s, e, step in body_plan(name):
            out = chunk_path(name, phase, label)
            if os.path.exists(out):
                print(f"  {phase}/{label}: cached, skipping")
                continue
            print(f"  {phase}/{label}: {s} -> {e} step {step} ...")
            df = fetch_chunk(BODIES[name]["hid"], s, e, step)
            df.to_csv(out, index=False)
            print(f"    {len(df)} rows -> {out}")
            time.sleep(sleep_between)  # be polite to NASA's server


def merge():
    frames = []
    for name in BODIES:
        frames_body = []
        for phase, label, _, _, _ in body_plan(name):
            p = chunk_path(name, phase, label)
            if os.path.exists(p):
                df = pd.read_csv(p)
                # Normalize Horizons' date string ("A.D. 1975-Jan-01 00:00:00.0000")
                # into a proper datetime64 column; keep raw chunks untouched.
                df["date"] = pd.to_datetime(
                    df["date"].str.replace("A.D. ", "", regex=False).str.strip(),
                    format="%Y-%b-%d %H:%M:%S.%f", errors="coerce")
                df["body"] = name
                df["phase"] = phase
                frames_body.append(df)
        if frames_body:
            got = pd.concat(frames_body, ignore_index=True)
            print(f"[{name}] {len(got)} rows "
                  f"(train {(got.phase=='train').sum()}, test {(got.phase=='test').sum()})")
            frames.append(got)
        else:
            print(f"[{name}] NO DATA: run 'download' first")
    if not frames:
        sys.exit("Nothing to merge.")
    full = pd.concat(frames, ignore_index=True)[
        ["body", "phase", "jd", "date", "x", "y", "z", "vx", "vy", "vz"]
    ]
    full.to_csv(MERGED, index=False)
    print(f"\nMerged dataset: {len(full):,} rows -> {MERGED} "
          f"({os.path.getsize(MERGED)/1e6:.0f} MB)")
    # Per-body summary: the sanity check to run before training anything
    print("\nSummary per body (train/test row counts):")
    print(full.groupby(["body", "phase"]).size().unstack(fill_value=0))


def status():
    for name in BODIES:
        done = [p for p in body_plan(name) if os.path.exists(chunk_path(name, *p[:2]))]
        print(f"[{name}] {len(done)}/{len(body_plan(name))} chunks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PlanetariUM-ML Horizons fetcher")
    ap.add_argument("command", choices=["download", "merge", "status"])
    ap.add_argument("--bodies", default=None,
                    help="subset, e.g. --bodies mercury,ceres,eris")
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="seconds between requests (courtesy pause for JPL)")
    args = ap.parse_args()

    names = list(BODIES) if args.bodies is None else args.bodies.split(",")
    for n in names:
        if n not in BODIES:
            sys.exit(f"Unknown body: {n}. Valid: {', '.join(BODIES)}")

    if args.command == "download":
        download(names, sleep_between=args.sleep)
    elif args.command == "merge":
        merge()
    else:
        status()