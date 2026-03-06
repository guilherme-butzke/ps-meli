import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


def create_missing_flag(df: pd.DataFrame, col: str, new_col: str) -> pd.Series:
    """
    Create binary missing indicator.

    Args:
        df (pd.DataFrame): Input dataframe.
        col (str): Original column name.
        new_col (str): Name of new feature.

    Returns:
        pd.Series: Binary indicator (1 if missing, 0 otherwise).
    """
    return df[col].isna().astype(int).rename(new_col)


def create_zero_flag(df: pd.DataFrame, col: str, new_col: str) -> pd.Series:
    """
    Create binary zero indicator.

    Args:
        df (pd.DataFrame): Input dataframe.
        col (str): Original column name.
        new_col (str): Name of new feature.

    Returns:
        pd.Series: Binary indicator (1 if value == 0).
    """
    return (df[col] == 0).astype(int).rename(new_col)


def create_log_feature(df: pd.DataFrame, col: str, new_col: str) -> pd.Series:
    """
    Apply log1p transformation.
    If zeros or negatives are present, this will return NaN for those rows (x>-1 required). 
    Consider using log1p(x - min + shift) if you want to handle negatives.

    Args:
        df (pd.DataFrame): Input dataframe.
        col (str): Original column.
        new_col (str): Name of new feature.

    Returns:
        pd.Series: Log-transformed feature.
    """
    x = pd.to_numeric(df[col], errors="coerce").apply(lambda v: v if v > -1 else np.nan)  # log1p requires x > -1
    result = np.log1p(x)
    return result.rename(new_col)

def create_ratio_feature(
    df: pd.DataFrame, cols: List[str] | Tuple[str],  new_col: str
) -> pd.Series:
    """
    Create ratio feature with safe division.

    Args:
        df (pd.DataFrame): Input dataframe.
        cols (List[str] | Tuple[str]): List or tuple of two column names (numerator, denominator).
        new_col (str): Name of new feature.

    Returns:
        pd.Series: Ratio feature.
    """
    col_num, col_den = cols
    ratio = df[col_num] / df[col_den].replace(0, np.nan)
    return ratio.rename(new_col)


def create_top_percentile_flag(
    df: pd.DataFrame, col: str, percentile: float, new_col: str
) -> pd.Series:
    """
    Flag top percentile values.

    Args:
        df (pd.DataFrame): Input dataframe.
        col (str): Column name.
        percentile (float): Percentile threshold (e.g., 0.99).
        new_col (str): Name of new feature.

    Returns:
        pd.Series: Binary flag.
    """
    threshold = df[col].quantile(percentile)
    return (df[col] >= threshold).astype(int).rename(new_col)


def create_quantile_bins(
    df: pd.DataFrame, col: str, n_bins: int, new_col: str
) -> pd.Series:
    """
    Create quantile-based bins.

    Args:
        df (pd.DataFrame): Input dataframe.
        col (str): Column name.
        n_bins (int): Number of bins.
        new_col (str): Name of new feature.

    Returns:
        pd.Series: Categorical bins.
    """
    return pd.qcut(df[col], q=n_bins, duplicates="drop").rename(new_col)

# ----------------------------
# Datetime features
# ----------------------------

def create_hour(df: pd.DataFrame, col: str, new_col: str) -> pd.Series:
    """
    Extract hour of day from a datetime-like column.

    Args:
        df: Input dataframe.
        col: Datetime column.
        new_col: Output feature name.

    Returns:
        Hour of day (0..23) as int.
    """
    dt = pd.to_datetime(df[col], errors="coerce")
    return dt.dt.hour.astype("Int64").rename(new_col)


def create_weekday(df: pd.DataFrame, col: str, new_col: str) -> pd.Series:
    """
    Extract weekday (Mon=0..Sun=6) from a datetime-like column.

    Args:
        df: Input dataframe.
        col: Datetime column.
        new_col: Output feature name.

    Returns:
        Weekday as int.
    """
    dt = pd.to_datetime(df[col], errors="coerce")
    return dt.dt.weekday.astype("Int64").rename(new_col)


def create_is_weekend(df: pd.DataFrame, col: str, new_col: str) -> pd.Series:
    """
    Weekend indicator from a datetime-like column.

    Args:
        df: Input dataframe.
        col: Datetime column.
        new_col: Output feature name.

    Returns:
        Binary indicator (1 if Sat/Sun).
    """
    dt = pd.to_datetime(df[col], errors="coerce")
    is_weekend = dt.dt.weekday.isin([5, 6]).astype(int)
    return pd.Series(is_weekend, index=df.index, name=new_col)


def create_hour_cyclic(
    df: pd.DataFrame,
    col: str,
    new_col: str,
    component: str = "sin",
    period: float = 24.0,
) -> pd.Series:
    """
    Cyclic encoding for hour of day, returning a single component (sin or cos).

    Args:
        df: Input dataframe.
        col: Datetime column.
        new_col: Output feature name.
        component: Which component to return: "sin" (default) or "cos".
        period: Period of the cycle (24 hours by default).

    Returns:
        pd.Series: Sin or cos cyclic encoding of hour.
    """
    dt = pd.to_datetime(df[col], errors="coerce")
    hour = dt.dt.hour.astype(float)  # NaN if dt is NaT
    angle = 2.0 * np.pi * hour / period

    component = component.lower().strip()
    if component == "sin":
        out = np.sin(angle)
    elif component == "cos":
        out = np.cos(angle)
    else:
        raise ValueError("component must be 'sin' or 'cos'.")

    return pd.Series(out, index=df.index, name=new_col)


# ----------------------------
# Categorical encodings
# ----------------------------

def create_frequency_encoding(df: pd.DataFrame, col: str, new_col: str, normalize: bool = True) -> pd.Series:
    """
    Frequency encoding for high-cardinality categoricals.

    Args:
        df: Input dataframe.
        col: Categorical column.
        new_col: Output feature name.
        normalize: If True, use relative frequency; else counts.

    Returns:
        Frequency-encoded numeric Series.
    """
    counts = df[col].value_counts(dropna=False)
    if normalize:
        counts = counts / len(df)
    return df[col].map(counts).astype(float).rename(new_col)


def create_rare_grouping(
    df: pd.DataFrame,
    col: str,
    new_col: str,
    min_freq: float = 0.01,
    other_label: str = "__OTHER__",
) -> pd.Series:
    """
    Group rare categories into a single label.

    Args:
        df: Input dataframe.
        col: Categorical column.
        new_col: Output feature name.
        min_freq: Minimum relative frequency to keep a category.
        other_label: Label for grouped rare categories.

    Returns:
        Grouped categorical Series.
    """
    vc = df[col].value_counts(dropna=False, normalize=True)
    keep = set(vc[vc >= min_freq].index)
    grouped = df[col].where(df[col].isin(keep), other_label)
    return grouped.astype("object").rename(new_col)


def create_target_encoding_cv(
    df: pd.DataFrame,
    col: str,
    target_col: str,
    new_col: str,
    n_splits: int = 5,
    smoothing: float = 10.0,
    random_state: int = 42,
) -> pd.Series:
    """
    Cross-validated target encoding (leakage-aware).
    Uses KFold; if you have time ordering, pass folds externally instead.

    Args:
        df: Input dataframe.
        col: Categorical column to encode.
        target_col: Binary target column (0/1).
        new_col: Output feature name.
        n_splits: Number of folds.
        smoothing: Larger -> more shrinkage toward global mean.
        random_state: Seed.

    Returns:
        Target-encoded Series aligned to df index.
    """
    # local import to avoid hard dependency if user doesn't want sklearn
    from sklearn.model_selection import KFold

    y = df[target_col].astype(float)
    x = df[col].astype("object")

    global_mean = y.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    out = pd.Series(index=df.index, dtype=float)

    for tr_idx, va_idx in kf.split(df):
        x_tr, y_tr = x.iloc[tr_idx], y.iloc[tr_idx]
        x_va = x.iloc[va_idx]

        stats = y_tr.groupby(x_tr).agg(["mean", "count"])
        # smoothing: (count * mean + smoothing*global) / (count + smoothing)
        smooth_mean = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)

        out.iloc[va_idx] = x_va.map(smooth_mean).fillna(global_mean).astype(float)

    return out.rename(new_col)


# ----------------------------
# Skew transformations
# ----------------------------

def create_yeojohnson(df: pd.DataFrame, col: str, new_col: str) -> pd.Series:
    """
    Yeo-Johnson transform (handles zeros/negatives).

    Args:
        df: Input dataframe.
        col: Numeric column.
        new_col: Output name.

    Returns:
        Transformed Series.
    """
    from sklearn.preprocessing import PowerTransformer

    x = pd.to_numeric(df[col], errors="coerce").to_frame()
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    z = pt.fit_transform(x)[:, 0]
    return pd.Series(z, index=df.index, name=new_col)


def create_boxcox(df: pd.DataFrame, col: str, new_col: str, shift: float = 1e-6) -> pd.Series:
    """
    Box-Cox transform (requires positive values). If values are non-positive,
    applies a shift based on min value + shift.

    Args:
        df: Input dataframe.
        col: Numeric column.
        new_col: Output name.
        shift: Small constant to ensure positivity if shifting is needed.

    Returns:
        Transformed Series.
    """
    from sklearn.preprocessing import PowerTransformer

    x = pd.to_numeric(df[col], errors="coerce")
    min_val = np.nanmin(x.values)
    if not np.isfinite(min_val):
        return pd.Series(np.nan, index=df.index, name=new_col)

    x_pos = x.copy()
    if min_val <= 0:
        x_pos = x_pos - min_val + shift

    pt = PowerTransformer(method="box-cox", standardize=False)
    z = pt.fit_transform(x_pos.to_frame())[:, 0]
    return pd.Series(z, index=df.index, name=new_col)


# ----------------------------
# Binning (IV-ready)
# ----------------------------

def create_quantile_bins_dropna(
    df: pd.DataFrame,
    col: str,
    n_bins: int,
    new_col: str,
    add_missing_bin: bool = True,
) -> pd.Series:
    """
    Quantile bins returning label IDs instead of interval ranges.
    """

    x = pd.to_numeric(df[col], errors="coerce")
    valid = x.notna()

    out = pd.Series(pd.NA, index=df.index, dtype="object")

    if valid.sum() == 0:
        if add_missing_bin:
            return out.where(valid, "__MISSING__").rename(new_col)
        return out.rename(new_col)

    try:
        b = pd.qcut(
            x[valid],
            q=n_bins,
            labels=False,
            duplicates="drop"
        )

        if b.nunique() < 2:
            if add_missing_bin:
                return out.where(valid, "__MISSING__").rename(new_col)
            return out.rename(new_col)

        out.loc[valid] = b.astype("Int64").astype(str)

    except ValueError:
        # fallback using percentile bins
        qs = np.unique(
            np.nanpercentile(x[valid], np.linspace(0, 100, n_bins + 1))
        )

        if qs.size < 3:
            if add_missing_bin:
                return out.where(valid, "__MISSING__").rename(new_col)
            return out.rename(new_col)

        qs[0], qs[-1] = -np.inf, np.inf
        b = pd.cut(x[valid], bins=qs, labels=False, include_lowest=True)
        out.loc[valid] = b.astype("Int64").astype(str)

    if add_missing_bin:
        out = out.where(valid, "__MISSING__")

    return out.rename(new_col)


# ----------------------------
# Interactions
# ----------------------------

def create_interaction_mul(
    df: pd.DataFrame,
    col: Tuple[str, str] | List[str],
    new_col: str,
) -> pd.Series:
    """
    Multiply two numeric columns to create an interaction feature.

    Args:
        df: Input dataframe.
        col: Pair of columns (col_a, col_b).
        new_col: Output feature name.

    Returns:
        Interaction Series (col_a * col_b).
    """
    if not isinstance(col, (tuple, list)) or len(col) != 2:
        raise ValueError("col must be a tuple/list with exactly 2 column names, e.g. ('monto','e').")

    col_a, col_b = col[0], col[1]
    x = pd.to_numeric(df[col_a], errors="coerce")
    y = pd.to_numeric(df[col_b], errors="coerce")
    return (x * y).rename(new_col)

# ----------------------------
# PCA features (optional)
# ----------------------------

def create_pca_components(
    df: pd.DataFrame,
    cols: List[str] | Tuple[str, ...],
    n_components: int,
    new_col: str,
) -> pd.DataFrame:
    """
    Fit PCA on selected columns and return component features.

    Note: Fit PCA on TRAIN only in real pipelines to avoid leakage.

    Args:
        df: Input dataframe.
        cols: Columns to include (must be numeric).
        n_components: Number of components.
        prefix: Feature prefix, e.g. 'pca_beh'.

    Returns:
        DataFrame with columns {prefix}_1..{prefix}_n.
    """
    from sklearn.decomposition import PCA

    if type(cols) == tuple:
        cols = list(cols)

    X = df[cols].apply(pd.to_numeric, errors="coerce")
    # basic imputation for PCA only (better to do this in a proper pipeline)
    X = X.fillna(X.median())

    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X)

    out = pd.DataFrame(Z, index=df.index, columns=[f"{new_col}_{i+1}" for i in range(n_components)])
    return out

class FeatureEngineer:
    """
    Feature engineering wrapper for modular transformations.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def missing_flag(self, col: str, new_col: str) -> pd.Series:
        return create_missing_flag(self.df, col, new_col)

    def zero_flag(self, col: str, new_col: str) -> pd.Series:
        return create_zero_flag(self.df, col, new_col)

    def log_feature(self, col: str, new_col: str) -> pd.Series:
        return create_log_feature(self.df, col, new_col)

    def ratio(self, cols: Tuple[str, str] | List[str], new_col: str) -> pd.Series:
        return create_ratio_feature(self.df, cols, new_col)

    def top_percentile_flag(self, col: str, percentile: float, new_col: str) -> pd.Series:
        return create_top_percentile_flag(self.df, col, percentile, new_col)

    def quantile_bins(self, col: str, n_bins: int, new_col: str) -> pd.Series:
        return create_quantile_bins(self.df, col, n_bins, new_col)

    def hour(self, col: str, new_col: str) -> pd.Series:
        return create_hour(self.df, col, new_col)

    def weekday(self, col: str, new_col: str) -> pd.Series:
        return create_weekday(self.df, col, new_col)

    def is_weekend(self, col: str, new_col: str) -> pd.Series:
        return create_is_weekend(self.df, col, new_col)

    def hour_cyclic(self,
        col: str, new_col: str, component: str = "sin", period: float = 24.0,) -> pd.Series:
        return create_hour_cyclic(self.df, col, new_col, component=component, period=period)

    def frequency_encoding(self, col: str, new_col: str, normalize: bool = True) -> pd.Series:
        return create_frequency_encoding(self.df, col, new_col, normalize)

    def rare_grouping(self, col: str, new_col: str, min_freq: float = 0.01, other_label: str = "__OTHER__") -> pd.Series:
        return create_rare_grouping(self.df, col, new_col, min_freq, other_label)

    def target_encoding_cv(self, col: str, target_col: str, new_col: str, n_splits: int = 5, smoothing: float = 10.0, random_state: int = 42) -> pd.Series:
        return create_target_encoding_cv(self.df, col, target_col, new_col, n_splits, smoothing, random_state)

    def yeojohnson(self, col: str, new_col: str) -> pd.Series:
        return create_yeojohnson(self.df, col, new_col)

    def boxcox(self, col: str, new_col: str, shift: float = 1e-6) -> pd.Series:
        return create_boxcox(self.df, col, new_col, shift)

    def quantile_bins_dropna(self, col: str, n_bins: int, new_col: str, add_missing_bin: bool = True) -> pd.Series:
        return create_quantile_bins_dropna(self.df, col, n_bins, new_col, add_missing_bin)
    
    def interaction_mul(
        self,
        col: Tuple[str, str] | List[str],
        new_col: str,
    ) -> pd.Series:
        return create_interaction_mul(self.df, col, new_col)

    def pca_components(self, cols: List[str], n_components: int, new_col: str) -> pd.DataFrame:
        return create_pca_components(self.df, cols, n_components, new_col)
    
    def apply_transformations(self, config_list: List[dict]) -> pd.DataFrame:
        """
        Apply multiple feature transformations based on configuration list.

        Args:
            config_list: List of dictionaries defining transformations.

        Returns:
            DataFrame with all generated features.
        """
        features = []

        for cfg in config_list:
            process = cfg["process"]
            col = cfg["col"]
            new_col = cfg["new_col"]
            params = cfg.get("params", {})

            try:
                method = getattr(self, process)

                # Handle tuple columns (like ratio)
                result = method(col, new_col=new_col, **params)
                # if isinstance(col, tuple):
                #     result = method(*col, new_col=new_col, **params)
                # else:
                #     result = method(col, new_col=new_col, **params)

                # Some methods return DataFrame (PCA)
                if isinstance(result, pd.DataFrame):
                    features.append(result)
                else:
                    features.append(result)

            except Exception as e:
                print(f"[WARNING] Failed {process} on {col}: {e}")

        return pd.concat(features, axis=1)