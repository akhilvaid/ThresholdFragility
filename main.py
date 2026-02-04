#!/bin/python3

import os
import collections

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import tqdm
import polars as pl

from threshold_fragility import DoTheDew


# Config
@dataclass(frozen=True)
class RunnerConfig:
    out_dir: str = "AnalysisOutput"
    n_workers: int = 20
    shuffle: bool = True
    seed: int = 0

    # Filenames
    run_summary_name: str = "run_summary.parquet"
    per_year_name: str = "per_year_long.parquet"
    decomp_name: str = "decomposition_siteyear.parquet"

    # If True, write empty parquet (schema-less) on no rows
    write_empty: bool = True


# Flatten helpers
def _training_params_to_dict(tp) -> dict:
    return {
        "model": tp.model,
        "cutoff_year": tp.cutoff,
        "external_site": tp.external_site,
        "delta_days": tp.delta,
        "pretrained": tp.pretrained,
    }


def flatten_run_summary(result: dict, tp) -> dict:
    base = _training_params_to_dict(tp)
    row = {
        **base,
        "qualifier": result.get("qualifier"),
        "threshold_frozen": result.get("threshold_frozen"),
    }

    # Pooled split metrics: split__metric columns
    for split_name, m in (result.get("metrics") or {}).items():
        if not isinstance(m, dict):
            continue
        for k, v in m.items():
            row[f"{split_name}__{k}"] = v

    return row


def flatten_per_year_long(result: dict, tp) -> List[dict]:
    base = _training_params_to_dict(tp)
    qualifier = result.get("qualifier")
    threshold_frozen = result.get("threshold_frozen")

    rows: List[dict] = []

    per_year_all = result.get("per_year") or {}
    for dataset_name, per_year in per_year_all.items():
        if not isinstance(per_year, dict):
            continue

        for year, year_dict in per_year.items():
            if not isinstance(year_dict, dict):
                continue

            # Frozen sentinel (-1)
            if -1 in year_dict and isinstance(year_dict[-1], dict):
                m = year_dict[-1]
                rows.append(
                    {
                        **base,
                        "qualifier": qualifier,
                        "threshold_frozen": threshold_frozen,
                        "dataset": str(dataset_name),
                        "year": int(year),
                        "k_calibration": -1,
                        "resample_id": -1,
                        **m,
                    }
                )

            for k_calib, nested in year_dict.items():
                if int(k_calib) == -1:
                    continue

                if isinstance(nested, dict) and nested:
                    sample_val = next(iter(nested.values()))
                    if isinstance(sample_val, dict):
                        for resample_id, m in nested.items():
                            if not isinstance(m, dict):
                                continue
                            rows.append(
                                {
                                    **base,
                                    "qualifier": qualifier,
                                    "threshold_frozen": threshold_frozen,
                                    "dataset": str(dataset_name),
                                    "year": int(year),
                                    "k_calibration": int(k_calib),
                                    "resample_id": int(resample_id),
                                    **m,
                                }
                            )
                        continue

                if isinstance(nested, dict):
                    rows.append(
                        {
                            **base,
                            "qualifier": qualifier,
                            "threshold_frozen": threshold_frozen,
                            "dataset": str(dataset_name),
                            "year": int(year),
                            "k_calibration": int(k_calib),
                            "resample_id": -1,
                            **nested,
                        }
                    )

    return rows


def _pl_to_rows(df: pl.DataFrame) -> List[dict]:
    if df is None or df.is_empty():
        return []
    return df.to_dicts()


def flatten_decomposition_siteyear(result: dict, tp) -> List[dict]:
    base = _training_params_to_dict(tp)
    qualifier = result.get("qualifier")
    threshold_frozen = result.get("threshold_frozen")

    rows: List[dict] = []

    decomp = result.get("decomposition") or {}
    if not isinstance(decomp, dict):
        return rows

    def consume(df_any: Any, dataset_name: str) -> None:
        if df_any is None:
            return

        # We expect a polars DataFrame; if something else sneaks in, skip cleanly.
        if not isinstance(df_any, pl.DataFrame):
            return

        d = df_any

        # Normalize names (support both upstream naming styles)
        if "SITE" in d.columns and "external_site" not in d.columns:
            d = d.rename({"SITE": "external_site"})
        if "ECG_YEAR" in d.columns and "year" not in d.columns:
            d = d.rename({"ECG_YEAR": "year"})

        # Basic requirements
        if "external_site" not in d.columns or "year" not in d.columns:
            return

        d = d.with_columns(
            pl.lit(str(dataset_name)).alias("dataset"),
            pl.lit(str(qualifier)).alias("qualifier"),
            pl.lit(float(threshold_frozen)).alias("threshold_frozen"),
            pl.col("external_site").cast(pl.Utf8),
            pl.col("year").cast(pl.Int64),
        )

        for r in _pl_to_rows(d):
            rows.append({**base, **r})

    consume(
        decomp.get("internal_prospective_siteyear"),
        "internal_prospective_decomposition",
    )
    consume(
        decomp.get("external_prospective_siteyear"),
        "external_prospective_decomposition",
    )

    return rows


def run_analysis(cfg: RunnerConfig = RunnerConfig()) -> None:
    # One experiment for demonstration purposes
    Experiment = collections.namedtuple(
        "Experiment", ["cutoff", "external_site", "delta", "model", "pretrained"]
    )

    combinations = [Experiment(
        cutoff=2016,
        external_site="MSB",
        delta=14,
        model="resnet50",
        pretrained=True,
    )]

    os.makedirs(cfg.out_dir, exist_ok=True)

    run_rows: List[dict] = []
    per_year_rows: List[dict] = []
    decomp_rows: List[dict] = []

    skipped = 0

    for tp in tqdm.tqdm(combinations, total=len(combinations), desc="Runs", unit="run"):
        result = DoTheDew(tp).result_collation()

        if result is None:
            skipped += 1
            continue

        run_rows.append(flatten_run_summary(result, tp))
        per_year_rows.extend(flatten_per_year_long(result, tp))
        decomp_rows.extend(flatten_decomposition_siteyear(result, tp))

    df_runs = pl.DataFrame(run_rows) if run_rows else pl.DataFrame()
    df_per_year = pl.DataFrame(per_year_rows) if per_year_rows else pl.DataFrame()
    df_decomp = pl.DataFrame(decomp_rows) if decomp_rows else pl.DataFrame()

    # Write outputs
    run_path = os.path.join(cfg.out_dir, cfg.run_summary_name)
    per_year_path = os.path.join(cfg.out_dir, cfg.per_year_name)
    decomp_path = os.path.join(cfg.out_dir, cfg.decomp_name)

    if df_runs.height > 0 or cfg.write_empty:
        df_runs.write_parquet(run_path)
    if df_per_year.height > 0 or cfg.write_empty:
        df_per_year.write_parquet(per_year_path)
    if df_decomp.height > 0 or cfg.write_empty:
        df_decomp.write_parquet(decomp_path)

    print("Finished")

if __name__ == "__main__":
    run_analysis()
