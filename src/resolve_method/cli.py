from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

from resolve_method.baselines import (
    BASELINES,
    FEATURE_METHODS,
    consensus_reference,
    masked_marginal_scores,
    run_baseline,
)
from resolve_method.campaign import SCHEDULES, VARIANTS, canonical_variant, parse_schedule, run_campaign
from resolve_method.data import load_csv
from resolve_method.scores import elite_from_labels, elite_k

def _csv_id(path: Path) -> str:
    given = Path(path).expanduser()
    try:
        return given.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        home = Path.home().resolve()
        try:
            return given.resolve().relative_to(home).as_posix()
        except ValueError:
            return given.as_posix()

def _elites(table: dict) -> set[int]:
    if table["elite"] is None:
        ids, _k = elite_from_labels(table["y"])
        return ids
    ids = {int(i) for i in _marked(table["elite"])}
    k = len(ids)
    n = table["y"].size
    if not 0 < k < n:
        raise ValueError(f"elite column marks {k} of {n} rows; need a proper subset")
    if k != elite_k(n):
        print(f"elite column has {k} rows; the default rule would use {elite_k(n)}", file=sys.stderr)
    return ids

def _marked(mask):
    return np.flatnonzero(np.asarray(mask, dtype=bool))

def _methods(spec: str) -> list[str]:
    names = []
    known = ("ours",) + VARIANTS + BASELINES
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part == "all":
            names.append("full")
            names.extend(BASELINES)
        elif part == "baselines":
            names.extend(BASELINES)
        elif part in known:
            names.append(part)
        else:
            raise ValueError(
                f"unknown method {part!r}; choose a variant, baselines, all, or one of {', '.join(VARIANTS + BASELINES)}"
            )
    out = []
    for name in names:
        if name not in out:
            out.append(name)
    return out

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run our ESM-2 keep/tune campaign, the ten baselines, or both."
    )
    parser.add_argument("--csv", required=True, help="path to the input CSV; see README and examples/test_input.csv")
    parser.add_argument("--schedule", required=True,
                        help="comma-separated: 2x24,3x16,4x12,6x8,8x6,12x4")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--method", default="ours",
                        help="ours, a Figure 3/4 variant, baselines, all, or a comma-separated list")
    parser.add_argument("--reference", default=None,
                        help="wild-type sequence for masked marginal. Default: per-column consensus")
    parser.add_argument("--gpu", type=int, default=None,
                        help="CUDA device index. Required unless every method is uniform_random.")
    parser.add_argument("--out", required=True, help="directory for campaign.json and summary.csv")
    args = parser.parse_args(argv)
    try:
        methods = _methods(args.method)
    except ValueError as exc:
        parser.error(str(exc))

    schedules = [s.strip() for s in args.schedule.split(",") if s.strip()]
    unknown = [s for s in schedules if s.replace("×", "x").replace("X", "x") not in SCHEDULES]
    if unknown:
        parser.error(f"unknown schedule(s) {unknown}; choose from {', '.join(SCHEDULES)}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        parser.error("pass at least one seed")

    needs_model = any(name != "uniform_random" for name in methods)
    if needs_model and args.gpu is None:
        parser.error("--gpu is required unless every requested method is uniform_random")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    encoder = None
    if needs_model:
        from resolve_method.lora import Esm2Encoder
        encoder = Esm2Encoder(device="cuda:0")

    given = Path(args.csv).expanduser()
    table = load_csv(given)
    csv_path = _csv_id(given)
    elite = _elites(table)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = encoder.embed_mean(table["sequences"]) if any(name in FEATURE_METHODS for name in methods) else None
    reference = None
    marginal = None
    if "frozen_esm2_masked_marginal_greedy" in methods:
        reference = args.reference.strip().upper() if args.reference else consensus_reference(table["sequences"])
        marginal = masked_marginal_scores(
            encoder.model, encoder.tok, table["sequences"], reference, encoder.device)
    summaries = []
    campaigns = []
    fields = ["method", "csv", "schedule", "seed", "n", "k", "hits_at_64", "recall",
              "n_frozen", "n_head", "n_repr", "n_repeat"]
    for schedule in schedules:
        key = schedule.replace("×", "x").replace("X", "x")
        rounds, take = parse_schedule(key)
        for seed in seeds:
            for name in methods:
                print(f"begin {name} {csv_path} {key} seed {seed} n={table['y'].size} k={len(elite)}", flush=True)
                if name in VARIANTS:
                    result = run_campaign(
                        source=csv_path, seed=seed, sequences=table["sequences"], y=table["y"],
                        elite=elite, encoder=encoder, schedule=key,
                        variant=canonical_variant(name),
                    )
                    result["method"] = canonical_variant(name)
                    modes = [row["mode"] for row in result["rows"]]
                    extra = {
                        "n_frozen": modes.count("frozen"),
                        "n_head": modes.count("head"),
                        "n_repr": modes.count("repr"),
                        "n_repeat": modes.count("repeat"),
                    }
                    detail = ",".join(modes)
                else:
                    result = run_baseline(
                        name=name, source=csv_path, seed=seed, sequences=table["sequences"],
                        y=table["y"], elite=elite, schedule=key, rounds=rounds, take=take,
                        features=raw, marginal=marginal, reference=reference,
                    )
                    extra = {"n_frozen": "", "n_head": "", "n_repr": "", "n_repeat": ""}
                    detail = ""
                print(
                    f"done {name} {csv_path} {key} seed {seed} hits {result['hits_at_64']}/{result['k']}"
                    + (f" modes {detail}" if detail else ""),
                    flush=True,
                )
                campaigns.append(result)
                summaries.append({
                    "method": name,
                    "csv": csv_path,
                    "schedule": key,
                    "seed": seed,
                    "n": result["n"],
                    "k": result["k"],
                    "hits_at_64": result["hits_at_64"],
                    "recall": result["recall"],
                    **extra,
                })
    payload = {
        "csv": csv_path,
        "reference": reference,
        "campaigns": campaigns,
    }
    text = json.dumps(payload, indent=2)
    tmp = out / "campaign.json.tmp"
    tmp.write_text(text)
    tmp.replace(out / "campaign.json")
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote {out / 'campaign.json'}", flush=True)
    print(f"wrote {out / 'summary.csv'}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
