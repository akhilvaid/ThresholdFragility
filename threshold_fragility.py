import polars as pl
import numpy as np

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# Config globals
TARGET_SENSITIVITY = 0.90
CALIBRATION_N = [10, 50, 100, 200]
MIN_EVAL_N = 10  # Minimum evaluation set size after recalibration
EUIA_NAME = "Echocardiography Utilization Impact Analysis (EUIA)"

# Defaults
SENS_TOL = 0.05  # for volatility summaries
N_RESAMPLES_PER_K = 200  # resampling count PER k
NEAR_THR_EPS = 0.02  # threshold fragility diagnostics
MIN_POS_IN_CAL = 5  # recalibration guardrails
MIN_EVAL_N_LOCAL = 25  # recalibration guardrails
STRATIFY_CAL = True

# Follow-up windows (days after ECG) to quantify echo utilization
ECHO_FU_WINDOWS_DAYS = [30, 90]

# Minimum gap to avoid counting labeling / near-term echo as downstream
ECHO_FU_MIN_GAP_DAYS = 14


class Windowing:
    @staticmethod
    def add_year(df: pl.DataFrame) -> pl.DataFrame:
        if "ECG_YEAR" in df.columns:
            return df
        return df.with_columns(pl.col("ECGDATETIME").dt.year().alias("ECG_YEAR"))

    @staticmethod
    def pre(df: pl.DataFrame, cutoff_year: int) -> pl.DataFrame:
        df = Windowing.add_year(df)
        return df.filter(pl.col("ECG_YEAR") <= int(cutoff_year))

    @staticmethod
    def post(df: pl.DataFrame, cutoff_year: int) -> pl.DataFrame:
        df = Windowing.add_year(df)
        return df.filter(pl.col("ECG_YEAR") >= int(cutoff_year))

    @staticmethod
    def all(df: pl.DataFrame) -> pl.DataFrame:
        return Windowing.add_year(df)


# Echocardiography Utilization Impact Analysis (EUIA)
class EUIA:
    @staticmethod
    def annotate_with_followup_echo(
        df_ecg: pl.DataFrame, df_echo: pl.DataFrame
    ) -> pl.DataFrame:
        req_ecg = {"MRN", "ECGDATETIME"}
        req_echo = {"MRN", "ECHODATETIME"}

        missing_ecg = req_ecg - set(df_ecg.columns)
        missing_echo = req_echo - set(df_echo.columns)
        if missing_ecg:
            raise ValueError(f"EUIA missing ECG columns: {sorted(missing_ecg)}")
        if missing_echo:
            raise ValueError(f"EUIA missing ECHO columns: {sorted(missing_echo)}")

        # Normalize types: join_asof works best with Datetime.
        df_ecg = df_ecg.with_columns(
            pl.col("MRN").cast(pl.Utf8),
            pl.col("ECGDATETIME").cast(pl.Datetime),
        )

        # ECHODATETIME may be Date (from EF table); cast to Datetime at midnight for asof join.
        df_echo = (
            df_echo.with_columns(
                pl.col("MRN").cast(pl.Utf8),
                pl.col("ECHODATETIME").cast(pl.Datetime),
            )
            .drop_nulls()
            .unique(subset=["MRN", "ECHODATETIME"])
            .sort(["MRN", "ECHODATETIME"])
        )

        df_ecg_sorted = df_ecg.sort(["MRN", "ECGDATETIME"])

        df_joined = df_ecg_sorted.join_asof(
            df_echo,
            left_on="ECGDATETIME",
            right_on="ECHODATETIME",
            by="MRN",
            strategy="forward",
        ).rename({"ECHODATETIME": "FIRST_ECHO_AFTER_ECG"})

        # Days to first echo (null if no echo after ECG)
        df_joined = df_joined.with_columns(
            (pl.col("FIRST_ECHO_AFTER_ECG") - pl.col("ECGDATETIME"))
            .dt.total_days()
            .alias("ECHO_FU_DAYS")
        )

        # Window indicators with min-gap exclusion
        for w in ECHO_FU_WINDOWS_DAYS:
            df_joined = df_joined.with_columns(
                (
                    (pl.col("ECHO_FU_DAYS").is_not_null())
                    & (pl.col("ECHO_FU_DAYS") > float(ECHO_FU_MIN_GAP_DAYS))
                    & (pl.col("ECHO_FU_DAYS") <= float(w))
                ).alias(f"ECHO_FU_{int(w)}D")
            )

        return df_joined


class Summaries:
    @staticmethod
    def _nan_vol() -> dict:
        return {
            "vol_n_years": 0,
            "vol_frac_years_outside_tol": np.nan,
            "vol_sens_sd": np.nan,
            "vol_sens_iqr": np.nan,
            "vol_sens_range": np.nan,
            "vol_spec_sd": np.nan,
            "vol_spec_iqr": np.nan,
            "vol_spec_range": np.nan,
            "vol_fp_per_1000_sd": np.nan,
            "vol_fp_per_1000_iqr": np.nan,
            "vol_fp_per_1000_range": np.nan,
            "vol_fn_per_1000_sd": np.nan,
            "vol_fn_per_1000_iqr": np.nan,
            "vol_fn_per_1000_range": np.nan,
            "vol_alerts_per_1000_sd": np.nan,
            "vol_alerts_per_1000_iqr": np.nan,
            "vol_alerts_per_1000_range": np.nan,
            "vol_sens_yoy_mae": np.nan,
            "vol_sens_yoy_mse": np.nan,
            "vol_fp_per_1000_yoy_mae": np.nan,
            "vol_fp_per_1000_yoy_mse": np.nan,
        }

    @staticmethod
    def _vol_stats(x: np.ndarray) -> dict:
        x = x[~np.isnan(x)]
        if x.size == 0:
            return {"sd": np.nan, "iqr": np.nan, "range": np.nan}
        q1, q3 = np.quantile(x, 0.25), np.quantile(x, 0.75)
        sd = float(np.std(x, ddof=1)) if x.size >= 2 else 0.0
        return {"sd": sd, "iqr": float(q3 - q1), "range": float(np.max(x) - np.min(x))}

    @staticmethod
    def _yoy_mae_mse(x: np.ndarray) -> tuple[float, float]:
        x = x[~np.isnan(x)]
        if x.size <= 1:
            return (np.nan, np.nan)
        d = np.diff(x)
        return (float(np.mean(np.abs(d))), float(np.mean(d**2)))

    @staticmethod
    def volatility_from_yearly_rows(
        yearly_rows: list[dict], *, sens_target: float, sens_tol: float
    ) -> dict:
        if not yearly_rows:
            return Summaries._nan_vol()

        yearly_rows = sorted(yearly_rows, key=lambda r: int(r.get("year", -1)))

        sens = np.array(
            [r.get("sensitivity", np.nan) for r in yearly_rows], dtype=float
        )
        spec = np.array(
            [r.get("specificity", np.nan) for r in yearly_rows], dtype=float
        )
        fp1000 = np.array(
            [r.get("fp_per_1000", np.nan) for r in yearly_rows], dtype=float
        )
        fn1000 = np.array(
            [r.get("fn_per_1000", np.nan) for r in yearly_rows], dtype=float
        )
        alerts = np.array(
            [r.get("alerts_per_1000", np.nan) for r in yearly_rows], dtype=float
        )

        clean_s = sens[~np.isnan(sens)]
        frac_out = (
            float(np.mean(np.abs(clean_s - float(sens_target)) > float(sens_tol)))
            if clean_s.size
            else np.nan
        )

        vs = Summaries._vol_stats(sens)
        vsp = Summaries._vol_stats(spec)
        vfp = Summaries._vol_stats(fp1000)
        vfn = Summaries._vol_stats(fn1000)
        va = Summaries._vol_stats(alerts)

        sens_mae, sens_mse = Summaries._yoy_mae_mse(sens)
        fp_mae, fp_mse = Summaries._yoy_mae_mse(fp1000)

        return {
            "vol_n_years": int(len(yearly_rows)),
            "vol_frac_years_outside_tol": float(frac_out),
            "vol_sens_sd": float(vs["sd"]),
            "vol_sens_iqr": float(vs["iqr"]),
            "vol_sens_range": float(vs["range"]),
            "vol_spec_sd": float(vsp["sd"]),
            "vol_spec_iqr": float(vsp["iqr"]),
            "vol_spec_range": float(vsp["range"]),
            "vol_fp_per_1000_sd": float(vfp["sd"]),
            "vol_fp_per_1000_iqr": float(vfp["iqr"]),
            "vol_fp_per_1000_range": float(vfp["range"]),
            "vol_fn_per_1000_sd": float(vfn["sd"]),
            "vol_fn_per_1000_iqr": float(vfn["iqr"]),
            "vol_fn_per_1000_range": float(vfn["range"]),
            "vol_alerts_per_1000_sd": float(va["sd"]),
            "vol_alerts_per_1000_iqr": float(va["iqr"]),
            "vol_alerts_per_1000_range": float(va["range"]),
            "vol_sens_yoy_mae": float(sens_mae) if sens_mae == sens_mae else np.nan,
            "vol_sens_yoy_mse": float(sens_mse) if sens_mse == sens_mse else np.nan,
            "vol_fp_per_1000_yoy_mae": float(fp_mae) if fp_mae == fp_mae else np.nan,
            "vol_fp_per_1000_yoy_mse": float(fp_mse) if fp_mse == fp_mse else np.nan,
        }


class AnalysisCalcs:
    @staticmethod
    def _to_numpy_binary_outcomes(df: pl.DataFrame) -> np.ndarray:
        return df["OUTCOME"].to_numpy().astype(int)

    @staticmethod
    def _to_numpy_preds(df: pl.DataFrame) -> np.ndarray:
        return df["PRED"].to_numpy().astype(float)

    @staticmethod
    def compute_threshold(df_pred: pl.DataFrame) -> float:
        preds = AnalysisCalcs._to_numpy_preds(df_pred)
        outcomes = AnalysisCalcs._to_numpy_binary_outcomes(df_pred)

        if outcomes.size == 0 or np.unique(outcomes).size < 2:
            return 1.0

        fpr, tpr, thresholds = metrics.roc_curve(outcomes, preds)

        idxs = np.where(tpr >= TARGET_SENSITIVITY)[0]
        if idxs.size > 0:
            optimal_threshold = float(thresholds[idxs[0]])
        else:
            optimal_threshold = 1.0

        if not np.isfinite(optimal_threshold):
            optimal_threshold = 1.0
        return float(np.clip(optimal_threshold, 0.0, 1.0))

    @staticmethod
    def _confusion_counts(outcomes: np.ndarray, preds_binary: np.ndarray) -> dict:
        tp = int(np.sum((outcomes == 1) & (preds_binary == 1)))
        fp = int(np.sum((outcomes == 0) & (preds_binary == 1)))
        tn = int(np.sum((outcomes == 0) & (preds_binary == 0)))
        fn = int(np.sum((outcomes == 1) & (preds_binary == 0)))
        n = int(outcomes.size)
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": n}

    @staticmethod
    def _per_1000(count: int, n: int) -> float:
        if n <= 0:
            return float("nan")
        return float(count) * 1000.0 / float(n)

    @staticmethod
    def _near_thr_mass(preds: np.ndarray, thr: float, eps: float) -> float:
        if preds.size == 0:
            return float("nan")
        return float(np.mean(np.abs(preds - float(thr)) <= float(eps)))

    @staticmethod
    def clf_metrics(df: pl.DataFrame, threshold: float) -> dict:
        outcomes = AnalysisCalcs._to_numpy_binary_outcomes(df)
        preds = AnalysisCalcs._to_numpy_preds(df)

        n = int(outcomes.size)
        if n == 0:
            return {
                "auroc": np.nan,
                "auprc": np.nan,
                "sensitivity": np.nan,
                "specificity": np.nan,
                "ppv": np.nan,
                "npv": np.nan,
                "threshold": float(threshold),
                "target_sensitivity": float(TARGET_SENSITIVITY),
                "delta_sens_vs_target": np.nan,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "n": 0,
                "prevalence": np.nan,
                "alerts_per_1000": np.nan,
                "fp_per_1000": np.nan,
                "fn_per_1000": np.nan,
                "tp_per_1000": np.nan,
                "tn_per_1000": np.nan,
                "fpr": np.nan,
                "fnr": np.nan,
                "balanced_accuracy": np.nan,
                "youden_j": np.nan,
                "brier": np.nan,
                "mean_pred": np.nan,
                "near_thr_eps": float(NEAR_THR_EPS),
                "near_thr_mass": np.nan,
                "has_both_classes": False,
            }

        has_both = np.unique(outcomes).size == 2

        auroc = metrics.roc_auc_score(outcomes, preds) if has_both else np.nan
        auprc = metrics.average_precision_score(outcomes, preds) if has_both else np.nan

        preds_binary = (preds >= float(threshold)).astype(int)

        sensitivity = metrics.recall_score(
            outcomes, preds_binary, average="binary", zero_division=0
        )
        specificity = metrics.recall_score(
            outcomes, preds_binary, pos_label=0, average="binary", zero_division=0
        )
        ppv = metrics.precision_score(
            outcomes, preds_binary, average="binary", zero_division=0
        )
        npv = metrics.precision_score(
            outcomes, preds_binary, pos_label=0, average="binary", zero_division=0
        )

        counts = AnalysisCalcs._confusion_counts(outcomes, preds_binary)
        prevalence = float(np.mean(outcomes)) if n > 0 else float("nan")

        alerts = counts["tp"] + counts["fp"]

        alerts_per_1000 = AnalysisCalcs._per_1000(alerts, n)
        fp_per_1000 = AnalysisCalcs._per_1000(counts["fp"], n)
        fn_per_1000 = AnalysisCalcs._per_1000(counts["fn"], n)
        tp_per_1000 = AnalysisCalcs._per_1000(counts["tp"], n)
        tn_per_1000 = AnalysisCalcs._per_1000(counts["tn"], n)

        delta_sens_vs_target = float(sensitivity) - float(TARGET_SENSITIVITY)

        fpr = 1.0 - float(specificity)
        fnr = 1.0 - float(sensitivity)
        balanced_accuracy = 0.5 * (float(sensitivity) + float(specificity))
        youden_j = float(sensitivity) + float(specificity) - 1.0
        brier = float(np.mean((preds - outcomes) ** 2)) if n > 0 else float("nan")

        mean_pred = float(np.mean(preds)) if preds.size else float("nan")
        near_thr_mass = AnalysisCalcs._near_thr_mass(
            preds, float(threshold), NEAR_THR_EPS
        )

        # EUIA echo follow-up metrics
        eui_out = {
            "eui_name": EUIA_NAME,
            "echo_fu_min_gap_days": int(ECHO_FU_MIN_GAP_DAYS),
        }
        for w in ECHO_FU_WINDOWS_DAYS:
            col = f"ECHO_FU_{int(w)}D"
            if col not in df.columns:
                continue

            echo_any = df[col].to_numpy().astype(bool)
            alerts_mask = preds_binary.astype(bool)

            n_alerts = int(np.sum(alerts_mask))
            echo_in_alerts = int(np.sum(echo_any & alerts_mask))
            echo_in_nonalerts = int(np.sum(echo_any & (~alerts_mask)))

            echo_any_per_1000 = AnalysisCalcs._per_1000(int(np.sum(echo_any)), n)
            echo_in_alerts_per_1000 = AnalysisCalcs._per_1000(echo_in_alerts, n)
            echo_in_nonalerts_per_1000 = AnalysisCalcs._per_1000(echo_in_nonalerts, n)

            # If policy requires echo for every alert, incremental echo volume is:
            incremental = n_alerts - echo_in_alerts
            incremental_per_1000 = AnalysisCalcs._per_1000(int(incremental), n)

            echo_any_rate = float(np.mean(echo_any)) if n > 0 else np.nan
            echo_alerts_rate = (
                float(echo_in_alerts / n_alerts) if n_alerts > 0 else np.nan
            )

            eui_out.update(
                {
                    f"echo_fu_{int(w)}d_any_per_1000": float(echo_any_per_1000),
                    f"echo_fu_{int(w)}d_among_alerts_per_1000": float(
                        echo_in_alerts_per_1000
                    ),
                    f"echo_fu_{int(w)}d_among_nonalerts_per_1000": float(
                        echo_in_nonalerts_per_1000
                    ),
                    f"incremental_echos_{int(w)}d_per_1000": float(
                        incremental_per_1000
                    ),
                    f"echo_fu_{int(w)}d_any_rate": float(echo_any_rate),
                    f"echo_fu_{int(w)}d_among_alerts_rate": float(echo_alerts_rate),
                }
            )

        return {
            "auroc": float(auroc) if np.isfinite(auroc) else np.nan,
            "auprc": float(auprc) if np.isfinite(auprc) else np.nan,
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "ppv": float(ppv),
            "npv": float(npv),
            "threshold": float(threshold),
            "target_sensitivity": float(TARGET_SENSITIVITY),
            "delta_sens_vs_target": float(delta_sens_vs_target),
            **counts,
            "prevalence": float(prevalence),
            "alerts_per_1000": float(alerts_per_1000),
            "fp_per_1000": float(fp_per_1000),
            "fn_per_1000": float(fn_per_1000),
            "tp_per_1000": float(tp_per_1000),
            "tn_per_1000": float(tn_per_1000),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "balanced_accuracy": float(balanced_accuracy),
            "youden_j": float(youden_j),
            "brier": float(brier),
            "mean_pred": float(mean_pred),
            "near_thr_eps": float(NEAR_THR_EPS),
            "near_thr_mass": float(near_thr_mass),
            "has_both_classes": bool(has_both),
            **eui_out,
        }

    # Recalibration sampling policies
    @staticmethod
    def _deterministic_seed(year: int, n_sample: int, resample_id: int) -> int:
        seed = (year * 1_000_003 + n_sample * 10_009 + resample_id * 101) & 0xFFFFFFFF
        return int(seed)

    @staticmethod
    def _sample_indices_stratified(
        y: np.ndarray, k: int, rng: np.random.Generator
    ) -> np.ndarray | None:
        n = y.size
        if k >= n:
            return np.arange(n, dtype=int)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        if pos_idx.size == 0 or neg_idx.size == 0:
            return None

        prev = pos_idx.size / n
        k_pos = int(round(prev * k))
        k_pos = min(max(k_pos, 1), pos_idx.size)
        k_neg = k - k_pos
        if k_neg <= 0:
            k_neg = 1
            k_pos = k - 1
        if k_neg > neg_idx.size:
            return None

        take_pos = rng.choice(pos_idx, size=k_pos, replace=False)
        take_neg = rng.choice(neg_idx, size=k_neg, replace=False)
        idx = np.concatenate([take_pos, take_neg])
        rng.shuffle(idx)
        return idx.astype(int)

    @staticmethod
    def _sample_calibration_indices(
        y: np.ndarray, k: int, rng: np.random.Generator
    ) -> np.ndarray:
        n = y.size
        if k >= n:
            return np.arange(n, dtype=int)

        if STRATIFY_CAL:
            idx = AnalysisCalcs._sample_indices_stratified(y, k, rng)
            if idx is not None:
                return idx

        return rng.choice(np.arange(n), size=k, replace=False).astype(int)

    @staticmethod
    def per_year_metrics(df: pl.DataFrame, threshold: float, recalibrate: bool) -> dict:
        df = Windowing.add_year(df).sort("ECGDATETIME")
        years = sorted(df["ECG_YEAR"].unique().to_list())

        results: dict[int, dict] = {}

        for year in years:
            df_year = df.filter(pl.col("ECG_YEAR") == int(year)).sort("ECGDATETIME")
            results[int(year)] = {}

            if not recalibrate:
                metrics_year = AnalysisCalcs.clf_metrics(df_year, threshold)
                results[int(year)][-1] = {
                    **metrics_year,
                    "calibration_n_requested": -1,
                    "calibration_n": -1,
                    "resample_id": -1,
                    "n_year": int(df_year.height),
                    "eval_n": int(df_year.height),
                    "threshold_frozen": float(threshold),
                    "threshold_recalibrated": float(threshold),
                    "threshold_delta": 0.0,
                    "abs_threshold_delta": 0.0,
                    "calibration_policy": "none_frozen_threshold",
                    "evaluation_policy": "all_year",
                    "calibration_valid": True,
                    "calib_pos": np.nan,
                    "calib_prev": np.nan,
                    "eval_pos": (
                        int(df_year.select(pl.col("OUTCOME").sum()).item())
                        if df_year.height
                        else 0
                    ),
                    "eval_prev": (
                        float(df_year.select(pl.col("OUTCOME").mean()).item())
                        if df_year.height
                        else np.nan
                    ),
                    "min_pos_in_cal": MIN_POS_IN_CAL,
                    "min_eval_n": MIN_EVAL_N_LOCAL,
                }
                continue

            n_year = int(df_year.height)
            if n_year == 0:
                continue

            y_all = AnalysisCalcs._to_numpy_binary_outcomes(df_year)

            for k_requested in CALIBRATION_N:
                k_requested = int(k_requested)

                max_k_eff = (
                    max(1, n_year - int(MIN_EVAL_N_LOCAL))
                    if MIN_EVAL_N_LOCAL > 0
                    else n_year
                )
                k_eff = min(k_requested, max_k_eff)

                results[int(year)][k_requested] = {}

                for resample_id in range(N_RESAMPLES_PER_K):
                    seed = AnalysisCalcs._deterministic_seed(
                        int(year), int(k_requested), int(resample_id)
                    )
                    rng = np.random.default_rng(seed)

                    idx_cal = AnalysisCalcs._sample_calibration_indices(
                        y_all, k_eff, rng
                    )

                    mask = np.ones(n_year, dtype=bool)
                    mask[idx_cal] = False

                    df_cal = df_year.filter(pl.Series("m", ~mask))
                    df_eval = df_year.filter(pl.Series("m", mask))

                    calib_pos = (
                        int(df_cal.select(pl.col("OUTCOME").sum()).item())
                        if df_cal.height
                        else 0
                    )
                    calib_prev = (
                        float(df_cal.select(pl.col("OUTCOME").mean()).item())
                        if df_cal.height
                        else np.nan
                    )

                    eval_pos = (
                        int(df_eval.select(pl.col("OUTCOME").sum()).item())
                        if df_eval.height
                        else 0
                    )
                    eval_prev = (
                        float(df_eval.select(pl.col("OUTCOME").mean()).item())
                        if df_eval.height
                        else np.nan
                    )

                    calibration_valid = bool(
                        (calib_pos >= MIN_POS_IN_CAL) and (df_eval.height > 0)
                    )

                    common_audit = {
                        "calibration_n_requested": int(k_requested),
                        "calibration_n": int(k_eff),
                        "resample_id": int(resample_id),
                        "n_year": int(n_year),
                        "eval_n": int(df_eval.height),
                        "threshold_frozen": float(threshold),
                        "calibration_policy": (
                            "stratified_without_replacement_within_year"
                            if STRATIFY_CAL
                            else "random_without_replacement_within_year"
                        ),
                        "evaluation_policy": "out_of_sample_remaining_cases",
                        "calibration_valid": bool(calibration_valid),
                        "calib_pos": int(calib_pos),
                        "calib_prev": (
                            float(calib_prev) if calib_prev == calib_prev else np.nan
                        ),
                        "eval_pos": int(eval_pos),
                        "eval_prev": (
                            float(eval_prev) if eval_prev == eval_prev else np.nan
                        ),
                        "min_pos_in_cal": int(MIN_POS_IN_CAL),
                        "min_eval_n": int(MIN_EVAL_N_LOCAL),
                    }

                    if not calibration_valid:
                        results[int(year)][k_requested][int(resample_id)] = {
                            **AnalysisCalcs.clf_metrics(df_eval, float("nan")),
                            "threshold_recalibrated": np.nan,
                            "threshold_delta": np.nan,
                            "abs_threshold_delta": np.nan,
                            **common_audit,
                        }
                        continue

                    thr_recal = AnalysisCalcs.compute_threshold(df_cal)
                    threshold_delta = float(thr_recal - float(threshold))
                    abs_threshold_delta = float(abs(threshold_delta))

                    metrics_eval = AnalysisCalcs.clf_metrics(df_eval, thr_recal)

                    results[int(year)][k_requested][int(resample_id)] = {
                        **metrics_eval,
                        "threshold_recalibrated": float(thr_recal),
                        "threshold_delta": float(threshold_delta),
                        "abs_threshold_delta": float(abs_threshold_delta),
                        **common_audit,
                    }

        return results


class ThresholdFragilityDecomposition:
    @staticmethod
    def _clip01(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return np.clip(p, eps, 1.0 - eps)

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = ThresholdFragilityDecomposition._clip01(p)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def fit_train_calibrator(df_train: pl.DataFrame) -> dict:
        # Returns dict with a,b and basic fit diagnostics.
        y = AnalysisCalcs._to_numpy_binary_outcomes(df_train)
        p = AnalysisCalcs._to_numpy_preds(df_train)

        if y.size == 0 or np.unique(y).size < 2:
            return {"valid": False, "a": np.nan, "b": np.nan, "n": int(y.size)}

        x = ThresholdFragilityDecomposition._logit(p).reshape(-1, 1)

        # Very light regularization for numerical stability.
        lr = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
        )
        lr.fit(x, y)

        a = float(lr.intercept_[0])
        b = float(lr.coef_[0][0])

        return {"valid": True, "a": a, "b": b, "n": int(y.size)}

    @staticmethod
    def apply_calibrator(p: np.ndarray, calib: dict) -> np.ndarray:
        """
        Apply training-fit calibrator to new probabilities.
        """
        if not calib.get("valid", False):
            return p.astype(float)
        a = float(calib["a"])
        b = float(calib["b"])
        z = a + b * ThresholdFragilityDecomposition._logit(p)
        return ThresholdFragilityDecomposition._sigmoid(z).astype(float)

    @staticmethod
    def _tpr_fpr_at_threshold(
        y: np.ndarray, p: np.ndarray, thr: float
    ) -> tuple[float, float]:
        if y.size == 0 or np.unique(y).size < 2:
            return (np.nan, np.nan)

        yhat = (p >= float(thr)).astype(int)

        tp = np.sum((y == 1) & (yhat == 1))
        fn = np.sum((y == 1) & (yhat == 0))
        fp = np.sum((y == 0) & (yhat == 1))
        tn = np.sum((y == 0) & (yhat == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        return (float(tpr), float(fpr))

    @staticmethod
    def _alerts_per_1000(pi: float, tpr: float, fpr: float) -> float:
        """
        Expected alert rate per 1,000 given prevalence, TPR, FPR:
        """
        if not np.isfinite(pi) or not np.isfinite(tpr) or not np.isfinite(fpr):
            return np.nan
        return 1000.0 * (pi * tpr + (1.0 - pi) * fpr)

    @staticmethod
    def _local_roc_slope(y: np.ndarray, p: np.ndarray, thr: float, eps: float) -> float:
        """
        Estimate local ROC slope dTPR/dFPR around the operating threshold by
        computing TPR/FPR at thr-eps and thr+eps and taking a finite-diff ratio.

        High magnitude suggests small threshold movement can strongly change TPR vs FPR.
        """
        if y.size == 0 or np.unique(y).size < 2:
            return np.nan

        tpr_lo, fpr_lo = ThresholdFragilityDecomposition._tpr_fpr_at_threshold(
            y, p, thr - eps
        )
        tpr_hi, fpr_hi = ThresholdFragilityDecomposition._tpr_fpr_at_threshold(
            y, p, thr + eps
        )

        if not (
            np.isfinite(tpr_lo)
            and np.isfinite(tpr_hi)
            and np.isfinite(fpr_lo)
            and np.isfinite(fpr_hi)
        ):
            return np.nan

        dfpr = fpr_hi - fpr_lo
        dtpr = tpr_hi - tpr_lo
        if abs(dfpr) < 1e-12:
            return np.nan
        return float(dtpr / dfpr)

    @staticmethod
    def decompose_site_year(
        df_train: pl.DataFrame,
        df_siteyear: pl.DataFrame,
        *,
        thr_frozen: float,
        near_eps: float,
    ) -> dict:
        """
        Decompose delta alerts/1000 and delta sensitivity relative to the training period.
        """
        # --- training baseline ---
        y_tr = AnalysisCalcs._to_numpy_binary_outcomes(df_train)
        p_tr = AnalysisCalcs._to_numpy_preds(df_train)

        pi_tr = float(np.mean(y_tr)) if y_tr.size else np.nan
        tpr_tr, fpr_tr = ThresholdFragilityDecomposition._tpr_fpr_at_threshold(
            y_tr, p_tr, thr_frozen
        )

        alerts_tr = ThresholdFragilityDecomposition._alerts_per_1000(
            pi_tr, tpr_tr, fpr_tr
        )

        # Sensitivity baseline (should be near target by construction)
        sens_tr = tpr_tr

        # Observed site year
        y_sy = AnalysisCalcs._to_numpy_binary_outcomes(df_siteyear)
        p_sy = AnalysisCalcs._to_numpy_preds(df_siteyear)

        pi_sy = float(np.mean(y_sy)) if y_sy.size else np.nan
        tpr_sy, fpr_sy = ThresholdFragilityDecomposition._tpr_fpr_at_threshold(
            y_sy, p_sy, thr_frozen
        )
        alerts_sy = ThresholdFragilityDecomposition._alerts_per_1000(
            pi_sy, tpr_sy, fpr_sy
        )
        sens_sy = tpr_sy

        # Counterfactuals: 1
        alerts_prevfixed = ThresholdFragilityDecomposition._alerts_per_1000(
            pi_tr, tpr_sy, fpr_sy
        )

        # Counterfactuals: 2
        calib = ThresholdFragilityDecomposition.fit_train_calibrator(df_train)
        p_sy_cal = ThresholdFragilityDecomposition.apply_calibrator(p_sy, calib)
        tpr_sy_cal, fpr_sy_cal = ThresholdFragilityDecomposition._tpr_fpr_at_threshold(
            y_sy, p_sy_cal, thr_frozen
        )

        alerts_cal_prevfixed = ThresholdFragilityDecomposition._alerts_per_1000(
            pi_tr, tpr_sy_cal, fpr_sy_cal
        )
        sens_cal = tpr_sy_cal  # prevalence does not affect TPR

        # components for ALERTS drift (most important)
        delta_alerts_total = (
            alerts_sy - alerts_tr
            if (alerts_sy == alerts_sy and alerts_tr == alerts_tr)
            else np.nan
        )

        # sequential components (order: prevalence -> calibration -> residual)
        comp_prev = (
            (alerts_sy - alerts_prevfixed)
            if (alerts_sy == alerts_sy and alerts_prevfixed == alerts_prevfixed)
            else np.nan
        )
        comp_cal = (
            (alerts_prevfixed - alerts_cal_prevfixed)
            if (
                alerts_prevfixed == alerts_prevfixed
                and alerts_cal_prevfixed == alerts_cal_prevfixed
            )
            else np.nan
        )
        comp_resid = (
            (alerts_cal_prevfixed - alerts_tr)
            if (alerts_cal_prevfixed == alerts_cal_prevfixed and alerts_tr == alerts_tr)
            else np.nan
        )

        # components for SENS drift (prevalence component is 0 by construction)
        delta_sens_total = (
            sens_sy - sens_tr if (sens_sy == sens_sy and sens_tr == sens_tr) else np.nan
        )
        comp_sens_cal = (
            (sens_sy - sens_cal)
            if (sens_sy == sens_sy and sens_cal == sens_cal)
            else np.nan
        )
        comp_sens_resid = (
            (sens_cal - sens_tr)
            if (sens_cal == sens_cal and sens_tr == sens_tr)
            else np.nan
        )

        local_slope_raw = ThresholdFragilityDecomposition._local_roc_slope(
            y_sy, p_sy, thr_frozen, float(near_eps)
        )
        local_slope_cal = ThresholdFragilityDecomposition._local_roc_slope(
            y_sy, p_sy_cal, thr_frozen, float(near_eps)
        )

        return {
            # Training baseline
            "train_prev": float(pi_tr) if pi_tr == pi_tr else np.nan,
            "train_tpr": float(tpr_tr) if tpr_tr == tpr_tr else np.nan,
            "train_fpr": float(fpr_tr) if fpr_tr == fpr_tr else np.nan,
            "train_alerts_per_1000": (
                float(alerts_tr) if alerts_tr == alerts_tr else np.nan
            ),
            "train_sensitivity": float(sens_tr) if sens_tr == sens_tr else np.nan,
            # Observed site-year
            "siteyear_prev": float(pi_sy) if pi_sy == pi_sy else np.nan,
            "siteyear_tpr": float(tpr_sy) if tpr_sy == tpr_sy else np.nan,
            "siteyear_fpr": float(fpr_sy) if fpr_sy == fpr_sy else np.nan,
            "siteyear_alerts_per_1000": (
                float(alerts_sy) if alerts_sy == alerts_sy else np.nan
            ),
            "siteyear_sensitivity": float(sens_sy) if sens_sy == sens_sy else np.nan,
            # Counterfactuals
            "cf_prevfixed_alerts_per_1000": (
                float(alerts_prevfixed)
                if alerts_prevfixed == alerts_prevfixed
                else np.nan
            ),
            "cf_cal_prevfixed_alerts_per_1000": (
                float(alerts_cal_prevfixed)
                if alerts_cal_prevfixed == alerts_cal_prevfixed
                else np.nan
            ),
            "cf_cal_sensitivity": float(sens_cal) if sens_cal == sens_cal else np.nan,
            "cf_cal_tpr": float(tpr_sy_cal) if tpr_sy_cal == tpr_sy_cal else np.nan,
            "cf_cal_fpr": float(fpr_sy_cal) if fpr_sy_cal == fpr_sy_cal else np.nan,
            # Decomposition: ALERTS drift
            "delta_alerts_total_per_1000": (
                float(delta_alerts_total)
                if delta_alerts_total == delta_alerts_total
                else np.nan
            ),
            "delta_alerts_prev_component_per_1000": (
                float(comp_prev) if comp_prev == comp_prev else np.nan
            ),
            "delta_alerts_cal_component_per_1000": (
                float(comp_cal) if comp_cal == comp_cal else np.nan
            ),
            "delta_alerts_residual_per_1000": (
                float(comp_resid) if comp_resid == comp_resid else np.nan
            ),
            # Decomposition: SENS drift
            "delta_sens_total": (
                float(delta_sens_total)
                if delta_sens_total == delta_sens_total
                else np.nan
            ),
            "delta_sens_cal_component": (
                float(comp_sens_cal) if comp_sens_cal == comp_sens_cal else np.nan
            ),
            "delta_sens_residual": (
                float(comp_sens_resid) if comp_sens_resid == comp_sens_resid else np.nan
            ),
            # Diagnostics
            "thr_frozen": float(thr_frozen),
            "near_eps": float(near_eps),
            "local_roc_slope_raw": (
                float(local_slope_raw) if local_slope_raw == local_slope_raw else np.nan
            ),
            "local_roc_slope_cal": (
                float(local_slope_cal) if local_slope_cal == local_slope_cal else np.nan
            ),
            "cal_valid": bool(calib.get("valid", False)),
            "cal_a": (
                float(calib.get("a", np.nan))
                if calib.get("a", np.nan) == calib.get("a", np.nan)
                else np.nan
            ),
            "cal_b": (
                float(calib.get("b", np.nan))
                if calib.get("b", np.nan) == calib.get("b", np.nan)
                else np.nan
            ),
            "cal_n": int(calib.get("n", 0)),
            "n_siteyear": int(y_sy.size),
            "has_both_classes_siteyear": (
                bool(np.unique(y_sy).size == 2) if y_sy.size else False
            ),
        }

    @staticmethod
    def decompose_by_site_year(
        df_train: pl.DataFrame,
        df_eval: pl.DataFrame,
        *,
        thr_frozen: float,
        near_eps: float,
    ) -> pl.DataFrame:
        df_eval = Windowing.add_year(df_eval)

        req = {"SITE", "ECG_YEAR", "PRED", "OUTCOME"}
        missing = req - set(df_eval.columns)
        if missing:
            raise ValueError(f"Decomposition missing columns: {sorted(missing)}")

        sites = df_eval["SITE"].unique().to_list()
        rows: list[dict] = []

        for site in sites:
            df_s = df_eval.filter(pl.col("SITE") == site)
            years = sorted(df_s["ECG_YEAR"].unique().to_list())

            for year in years:
                df_sy = df_s.filter(pl.col("ECG_YEAR") == int(year))
                d = ThresholdFragilityDecomposition.decompose_site_year(
                    df_train,
                    df_sy,
                    thr_frozen=float(thr_frozen),
                    near_eps=float(near_eps),
                )
                rows.append(
                    {
                        "SITE": str(site),
                        "ECG_YEAR": int(year),
                        **d,
                    }
                )

        return pl.DataFrame(rows) if rows else pl.DataFrame()


class DoTheDew:
    def __init__(self, experiment_params):
        self.experiment_params = experiment_params

        # EUIA: echo events (MRN + ECHODATETIME) derived from EF dataset
        # Running this is optional, and requires an identified dataset to pull from.
        self.df_echos = None

    @staticmethod
    def _pooled_triplet(df: pl.DataFrame, cutoff_year: int, thr: float) -> dict:
        df_pre = Windowing.pre(df, cutoff_year)
        df_post = Windowing.post(df, cutoff_year)
        df_all = Windowing.all(df)

        m_pre = AnalysisCalcs.clf_metrics(df_pre, thr)
        m_post = AnalysisCalcs.clf_metrics(df_post, thr)
        m_all = AnalysisCalcs.clf_metrics(df_all, thr)

        out = {}
        for k, v in m_pre.items():
            out[f"pre__{k}"] = v
        for k, v in m_post.items():
            out[f"post__{k}"] = v
        for k, v in m_all.items():
            out[f"all__{k}"] = v
        return out

    @staticmethod
    def _pooled_deltas(ext_triplet: dict, int_triplet: dict) -> dict:
        keys = [
            "sensitivity",
            "specificity",
            "auroc",
            "brier",
            "alerts_per_1000",
            "fp_per_1000",
            "fn_per_1000",
            "near_thr_mass",
            "mean_pred",
            # EUIA headline fields
            "incremental_echos_90d_per_1000",
            "echo_fu_90d_any_per_1000",
            "echo_fu_90d_among_alerts_per_1000",
        ]
        out = {}
        for window in ("pre", "post", "all"):
            for k in keys:
                ek = f"{window}__{k}"
                ik = f"{window}__{k}"
                ev = ext_triplet.get(ek, np.nan)
                iv = int_triplet.get(ik, np.nan)
                out[f"delta_{window}__{k}"] = (
                    float(ev - iv) if (ev == ev and iv == iv) else np.nan
                )
        return out

    @staticmethod
    def _extract_yearly_post_rows(
        per_year_frozen: dict, cutoff_year: int
    ) -> list[dict]:
        rows = []
        for year, year_dict in per_year_frozen.items():
            if int(year) < int(cutoff_year):
                continue
            if -1 not in year_dict:
                continue
            m = year_dict[-1]
            rows.append({"year": int(year), **m})
        return rows

    def result_collation(self):
        cutoff_year = int(
            self.experiment_params.cutoff
        )  # NOTE Dates for the demo have been post-shifted to preserve patient privacy

        # Frozen threshold: internal test
        df_test = pl.read_parquet("TestData/df_test.parquet")
        df_test_prospective = pl.read_parquet("TestData/df_test_prospective.parquet")
        df_ext = pl.read_parquet("TestData/df_ext.parquet")
        df_ext_prospective = pl.read_parquet("TestData/df_ext_prospective.parquet")

        threshold = AnalysisCalcs.compute_threshold(Windowing.pre(df_test, cutoff_year))
        df_train_pre = Windowing.pre(df_test, cutoff_year)

        # Threshold fragility decomposition (SITE x YEAR)
        try:
            decomp_internal = ThresholdFragilityDecomposition.decompose_by_site_year(
                df_train_pre,
                df_test_prospective,
                thr_frozen=threshold,
                near_eps=float(NEAR_THR_EPS),
            )
        except Exception:
            decomp_internal = pl.DataFrame()

        try:
            decomp_external = ThresholdFragilityDecomposition.decompose_by_site_year(
                df_train_pre,
                df_ext_prospective,
                thr_frozen=threshold,
                near_eps=float(NEAR_THR_EPS),
            )
        except Exception:
            decomp_external = pl.DataFrame()

        # Pooled metrics @ frozen threshold (whole split)
        m_test = AnalysisCalcs.clf_metrics(df_test, threshold)
        m_test_pr = AnalysisCalcs.clf_metrics(df_test_prospective, threshold)
        m_ext = AnalysisCalcs.clf_metrics(df_ext, threshold)
        m_ext_pr = AnalysisCalcs.clf_metrics(df_ext_prospective, threshold)

        pooled_windows = {
            "internal_test": self._pooled_triplet(df_test, cutoff_year, threshold),
            "internal_prospective": self._pooled_triplet(
                df_test_prospective, cutoff_year, threshold
            ),
            "external": self._pooled_triplet(df_ext, cutoff_year, threshold),
            "external_prospective": self._pooled_triplet(
                df_ext_prospective, cutoff_year, threshold
            ),
        }

        pooled_deltas = {
            "external_minus_internal": self._pooled_deltas(
                pooled_windows["external"], pooled_windows["internal_test"]
            ),
            "external_prospective_minus_internal": self._pooled_deltas(
                pooled_windows["external_prospective"], pooled_windows["internal_test"]
            ),
        }

        # Yearly drift @ frozen threshold
        m_peryear_test = AnalysisCalcs.per_year_metrics(
            df_test_prospective, threshold, recalibrate=False
        )
        m_peryear_ext = AnalysisCalcs.per_year_metrics(
            df_ext_prospective, threshold, recalibrate=False
        )

        yearly_test_post_rows = self._extract_yearly_post_rows(
            m_peryear_test, cutoff_year
        )
        yearly_ext_post_rows = self._extract_yearly_post_rows(
            m_peryear_ext, cutoff_year
        )

        vol_test_post = Summaries.volatility_from_yearly_rows(
            yearly_test_post_rows, sens_target=TARGET_SENSITIVITY, sens_tol=SENS_TOL
        )
        vol_ext_post = Summaries.volatility_from_yearly_rows(
            yearly_ext_post_rows, sens_target=TARGET_SENSITIVITY, sens_tol=SENS_TOL
        )

        # Recalibration
        m_peryear_test_recal = AnalysisCalcs.per_year_metrics(
            df_test_prospective, threshold, recalibrate=True
        )
        m_peryear_ext_recal = AnalysisCalcs.per_year_metrics(
            df_ext_prospective, threshold, recalibrate=True
        )

        return {
            "cutoff_year": int(cutoff_year),
            "threshold_frozen": float(threshold),
            "config": {
                "target_sensitivity": float(TARGET_SENSITIVITY),
                "sens_tol": float(SENS_TOL),
                "near_thr_eps": float(NEAR_THR_EPS),
                "calibration_n_values": list(CALIBRATION_N),
                "n_resamples_per_k": int(N_RESAMPLES_PER_K),
                "min_pos_in_cal": int(MIN_POS_IN_CAL),
                "stratify_cal": bool(STRATIFY_CAL),
                # EUIA config
                "eui_name": EUIA_NAME,
                "echo_fu_windows_days": list(ECHO_FU_WINDOWS_DAYS),
                "echo_fu_min_gap_days": int(ECHO_FU_MIN_GAP_DAYS),
            },
            "metrics": {
                "test": m_test,
                "test_prospective": m_test_pr,
                "external": m_ext,
                "external_prospective": m_ext_pr,
            },
            "pooled_windows": pooled_windows,
            "pooled_deltas": pooled_deltas,
            "volatility": {
                "internal_prospective_post": vol_test_post,
                "external_prospective_post": vol_ext_post,
            },
            "per_year": {
                "test_prospective_frozen": m_peryear_test,
                "external_prospective_frozen": m_peryear_ext,
                "test_prospective_recalibrated": m_peryear_test_recal,
                "external_prospective_recalibrated": m_peryear_ext_recal,
            },
            "decomposition": {
                "internal_prospective_siteyear": decomp_internal,
                "external_prospective_siteyear": decomp_external,
            },
        }
