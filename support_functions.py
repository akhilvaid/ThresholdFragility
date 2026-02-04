import multiprocessing.dummy as multiprocessing

import numpy as np
import pandas as pd
import polars as pl

from sklearn import metrics

from config import Config, Experiment

TARGET_SENSITIVITY = 0.90


class PerfCalculation:
    @classmethod
    def calculate_metrics_single_sample(cls, df: pl.DataFrame) -> dict:

        s_true = df["TRUE"].to_numpy().astype(np.int32)
        s_pred = df["PRED"].to_numpy().astype(np.float32)

        auroc = metrics.roc_auc_score(s_true, s_pred)
        auprc = metrics.average_precision_score(s_true, s_pred)
        fpr, tpr, thresholds = metrics.roc_curve(s_true, s_pred)

        # Threshold for target sensitivity
        optimal_idx = np.where(tpr >= TARGET_SENSITIVITY)[0][0]
        optimal_threshold = thresholds[optimal_idx]

        s_pred_binary = (s_pred >= optimal_threshold).astype(int)
        specificity = metrics.recall_score(
            s_true, s_pred_binary, pos_label=0, average="binary", zero_division=0
        )
        ppv = metrics.precision_score(
            s_true, s_pred_binary, average="binary", zero_division=0
        )
        npv = metrics.precision_score(
            s_true, s_pred_binary, pos_label=0, average="binary", zero_division=0
        )

        return {
            "auroc": auroc,
            "auprc": auprc,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
            "optimal_threshold": optimal_threshold,
        }

    @classmethod
    def mean_ci(cls, array: pd.Series) -> tuple:
        array = np.array(array)
        mean = float(np.mean(array))
        lower = float(np.percentile(array, 2.5))
        upper = float(np.percentile(array, 97.5))
        return mean, lower, upper

    @classmethod
    def perf_calculation(cls, df_pred: pl.DataFrame) -> dict:
        # Calculate performance metrics from predictions
        # AUROC, AUPRC, Sensitivity, Specificity, PPV, NPV at the optimal threshold
        # Bootstrapping
        np.random.seed(Config.RANDOM_STATE)

        # Multiprocessed bootstrapping
        with multiprocessing.Pool(processes=1) as pool:
            # Create a list of bootstrap samples
            bootstrap_samples = [
                df_pred.sample(n=df_pred.height, with_replacement=True)
                for _ in range(Config.n_bootstrap)
            ]
            # Map the performance calculation function to each bootstrap sample
            bootstrapped_results = pool.map(
                PerfCalculation.calculate_metrics_single_sample, bootstrap_samples
            )

        # Finalize results
        df_results = pd.DataFrame(bootstrapped_results)

        # Return a tuple containing the mean and 95% CI for each metric
        results = {
            "n_samples": df_pred.height,
            "prevalence": float(df_pred["TRUE"].mean()),
            "auroc": cls.mean_ci(df_results["auroc"]),
            "auprc": cls.mean_ci(df_results["auprc"]),
            "specificity": cls.mean_ci(df_results["specificity"]),
            "ppv": cls.mean_ci(df_results["ppv"]),
            "npv": cls.mean_ci(df_results["npv"]),
            "optimal_threshold": cls.mean_ci(df_results["optimal_threshold"]),
        }

        return results
