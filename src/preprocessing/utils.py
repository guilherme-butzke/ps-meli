import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer


def diagnose_bin_feasibility(
    df: pd.DataFrame,
    cols: list,
    candidate_n_bins: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Identifies the maximum feasible number of quantile bins for each column.

    Fits a KBinsDiscretizer per column and detects when sklearn silently reduces
    the bin count due to insufficient unique quantile values. Returns a dict
    mapping each column to its actual (feasible) bin count, suitable for passing
    directly to BinnedFeaturesTransformer(n_bins=...).

    Args:
        df (pd.DataFrame): Input DataFrame containing the columns to diagnose.
        cols (list): Column names to evaluate.
        candidate_n_bins (int): Requested number of bins to test. Default is 10.
        verbose (bool): If True, prints a per-column diagnostic report.

    Returns:
        dict: {col: actual_n_bins} for every column in cols.
    """
    X_imputed = SimpleImputer(strategy="median").fit_transform(df[cols])

    if verbose:
        print(f"Diagnosing bin feasibility ({candidate_n_bins} requested bins):\n")

    per_feature_n_bins = {}
    for i, col in enumerate(cols):
        series = X_imputed[:, i]
        n_unique = len(np.unique(series))

        # Iteratively reduce n_bins until sklearn no longer warns.
        # One pass is not enough: the reduced count may itself still trigger warnings.
        n = candidate_n_bins
        had_warning = True
        while had_warning and n >= 2:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                binner = KBinsDiscretizer(
                    n_bins=n,
                    encode="ordinal",
                    strategy="quantile",
                    quantile_method="averaged_inverted_cdf",
                )
                binner.fit(series.reshape(-1, 1))
                actual = int(binner.n_bins_[0])
            had_warning = any("too small" in str(w.message) for w in caught)
            if had_warning:
                n = actual if actual < n else n - 1

        per_feature_n_bins[col] = n
        if verbose:
            status = (
                f"reduced to {n} bins (from {candidate_n_bins})"
                if n < candidate_n_bins
                else f"OK       {n} bins"
            )
            print(f"  [{i}] {col:15s}  unique={n_unique:6d}  {status}")

    if verbose:
        print(f"\nSuggested n_bins dict:\n{per_feature_n_bins}")

    return per_feature_n_bins


def build_feature_info(
    original_features_config: dict,
    feature_engineering_config: dict,
    df_raw: pd.DataFrame,
    df_features: pd.DataFrame,
) -> tuple:
    """
    Builds a feature metadata DataFrame and resolves continuous/categorical column lists.

    Args:
        original_features_config: Dict mapping original column names to their
            {'dtype': ..., 'action': ...} config. Only 'keep' columns are included.
        feature_engineering_config: Dict mapping engineered feature keys to their
            {'dtype': ..., 'config': {'new_col': ..., ...}} config.
        df_raw: The raw input DataFrame (before feature engineering).
        df_features: The engineered features DataFrame.

    Returns:
        df_complete_data (pd.DataFrame): Concatenation of df_raw and df_features.
        df_info_variables (pd.DataFrame): Columns ['feature', 'type', 'source'].
        continuous_cols (list): Features with type=='continuous' present in df_complete_data.
        categorical_cols (list): Features with type=='categorical' present in df_complete_data.
    """
    df_complete_data = pd.concat([df_raw, df_features], axis=1)

    prefix_to_dtype = {
        v["config"]["new_col"]: v["dtype"]
        for v in feature_engineering_config.values()
    }

    def _resolve_dtype(col: str) -> str:
        if col in prefix_to_dtype:
            return prefix_to_dtype[col]
        for prefix, dtype in prefix_to_dtype.items():
            if col.startswith(prefix):
                return dtype
        return "unknown"

    original_rows = [
        {"feature": col, "type": cfg["dtype"], "source": "original"}
        for col, cfg in original_features_config.items()
        if col in df_raw.columns and cfg["action"] == "keep"
    ]

    new_cols = [c for c in df_features.columns if c not in df_raw.columns]
    engineered_rows = [
        {"feature": col, "type": _resolve_dtype(col), "source": "engineered"}
        for col in new_cols
    ]

    df_info_variables = pd.DataFrame(original_rows + engineered_rows)

    continuous_cols = [
        c for c in df_info_variables.loc[df_info_variables.type == "continuous", "feature"]
        if c in df_complete_data.columns
    ]
    categorical_cols = [
        c for c in df_info_variables.loc[df_info_variables.type == "categorical", "feature"]
        if c in df_complete_data.columns
    ]

    return df_complete_data, df_info_variables, continuous_cols, categorical_cols