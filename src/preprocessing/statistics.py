from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple, Union

import numpy as np
import pandas as pd


FeatureType = Literal["continuous", "categorical"]
ReturnType = Literal["dict", "series"]


# ----------------------------
# Helpers
# ----------------------------

def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _to_binary_target(s: pd.Series) -> pd.Series:
    """
    Coerce target to {0,1}. Keeps NaN as NaN.
    """
    y = pd.to_numeric(s, errors="coerce")
    # if user already has 0/1, keep; else map truthy to 1.
    # safest: keep only 0 and 1, set others to NaN
    y = y.where(y.isin([0, 1]))
    return y


def _drop_pairwise_nan(x: pd.Series, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    m = x.notna() & y.notna()
    return x.loc[m].to_numpy(), y.loc[m].to_numpy()


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, (float, int, np.floating, np.integer)) and np.isfinite(v):
            return float(v)
        if isinstance(v, (float, int, np.floating, np.integer)) and not np.isfinite(v):
            return None
        return float(v)
    except Exception:
        return None


def _interpret_pvalue(p: Optional[float], alpha: float = 0.05) -> str:
    if p is None:
        return "p-value not available."
    return "Significant" if p < alpha else "Not significant"


def _interpret_auc(auc: Optional[float]) -> str:
    if auc is None:
        return "AUC not available."
    # typical heuristic bands
    if 0.7 <= auc < 0.8:
        return "Moderate separation"
    if auc >= 0.8:
        return "Strong separation"
    if 0.6 <= auc < 0.7:
        return "Weak separation"
    return "Close to random"


def _interpret_cohens_d(d: Optional[float]) -> str:
    if d is None:
        return "Cohen's d not available."
    ad = abs(d)
    if ad < 0.2:
        return "Negligible"
    if ad < 0.5:
        return "Small"
    if ad < 0.8:
        return "Medium"
    return "Large"


def _interpret_iv(iv: Optional[float]) -> str:
    if iv is None:
        return "IV not available."
    if iv < 0.02:
        return "Useless"
    if iv < 0.10:
        return "Weak"
    if iv < 0.30:
        return "Medium"
    if iv < 0.50:
        return "Strong"
    return "Suspicious (check leakage)"


def _interpret_cramers_v(v: Optional[float]) -> str:
    if v is None:
        return "not available"
    if v < 0.1:
        return "Negligible"
    if v < 0.3:
        return "Weak"
    if v < 0.5:
        return "Moderate"
    return "Strong"


def _fmt(val, decimals: int = 3) -> str:
    """Format a numeric value to fixed decimal places, or return 'N/A'."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


# ----------------------------
# Continuous vs Binary metrics
# ----------------------------

def mann_whitney_u_test(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    """
    Mann–Whitney U test for continuous feature vs binary target (0/1).
    """
    from scipy.stats import mannwhitneyu

    x_num = _to_numeric_series(x)
    y_bin = _to_binary_target(y)
    xv, yv = _drop_pairwise_nan(x_num, y_bin)

    x0 = xv[yv == 0]
    x1 = xv[yv == 1]

    if len(x0) < 2 or len(x1) < 2:
        return {"stat": None, "p_value": None, "n0": int(len(x0)), "n1": int(len(x1))}

    # two-sided by default
    stat, p = mannwhitneyu(x0, x1, alternative="two-sided")
    return {"stat": _safe_float(stat), "p_value": _safe_float(p), "n0": int(len(x0)), "n1": int(len(x1))}


def ks_statistic(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    """
    KS statistic for continuous feature vs binary target (0/1).
    """
    from scipy.stats import ks_2samp

    x_num = _to_numeric_series(x)
    y_bin = _to_binary_target(y)
    xv, yv = _drop_pairwise_nan(x_num, y_bin)

    x0 = xv[yv == 0]
    x1 = xv[yv == 1]

    if len(x0) < 2 or len(x1) < 2:
        return {"ks": None, "p_value": None, "n0": int(len(x0)), "n1": int(len(x1))}

    res = ks_2samp(x0, x1, alternative="two-sided", mode="auto")
    return {"ks": _safe_float(res.statistic), "p_value": _safe_float(res.pvalue), "n0": int(len(x0)), "n1": int(len(x1))}


def point_biserial_corr(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    """
    Point-biserial correlation between continuous x and binary y.
    """
    from scipy.stats import pointbiserialr

    x_num = _to_numeric_series(x)
    y_bin = _to_binary_target(y)
    xv, yv = _drop_pairwise_nan(x_num, y_bin)

    if len(xv) < 3:
        return {"r_pb": None, "p_value": None, "n": int(len(xv))}

    r, p = pointbiserialr(yv, xv)  # scipy expects (binary, continuous)
    return {"r_pb": _safe_float(r), "p_value": _safe_float(p), "n": int(len(xv))}


def single_feature_auc(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    """
    ROC AUC using a single feature as score. Returns both raw AUC and 'auc_abs' = max(auc, 1-auc).
    """
    from sklearn.metrics import roc_auc_score

    x_num = _to_numeric_series(x)
    y_bin = _to_binary_target(y)
    xv, yv = _drop_pairwise_nan(x_num, y_bin)

    # Need both classes present
    if len(np.unique(yv)) < 2 or len(yv) < 10:
        return {"auc": None, "auc_abs": None, "n": int(len(yv))}

    try:
        auc = float(roc_auc_score(yv, xv))
        return {"auc": auc, "auc_abs": max(auc, 1.0 - auc), "n": int(len(yv))}
    except Exception:
        return {"auc": None, "auc_abs": None, "n": int(len(yv))}


def cohens_d(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    """
    Cohen's d for difference in means between y=1 and y=0.
    """
    x_num = _to_numeric_series(x)
    y_bin = _to_binary_target(y)
    xv, yv = _drop_pairwise_nan(x_num, y_bin)

    x0 = xv[yv == 0]
    x1 = xv[yv == 1]

    if len(x0) < 2 or len(x1) < 2:
        return {"d": None, "mean0": _safe_float(np.nanmean(x0)) if len(x0) else None,
                "mean1": _safe_float(np.nanmean(x1)) if len(x1) else None,
                "n0": int(len(x0)), "n1": int(len(x1))}

    m0, m1 = float(np.mean(x0)), float(np.mean(x1))
    s0, s1 = float(np.std(x0, ddof=1)), float(np.std(x1, ddof=1))
    n0, n1 = len(x0), len(x1)

    # pooled sd
    denom = math.sqrt(((n0 - 1) * s0**2 + (n1 - 1) * s1**2) / (n0 + n1 - 2)) if (n0 + n1 - 2) > 0 else np.nan
    d = (m1 - m0) / denom if denom and np.isfinite(denom) and denom > 0 else np.nan

    return {"d": _safe_float(d), "mean0": m0, "mean1": m1, "n0": int(n0), "n1": int(n1)}


def logistic_wald_test_continuous(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    """
    Logistic regression Wald test for a single continuous predictor.
    Catches perfect separation and numerical instability.
    """

    try:
        import statsmodels.api as sm
        from statsmodels.tools.sm_exceptions import PerfectSeparationWarning, PerfectSeparationError
    except Exception:
        return {"beta": None, "odds_ratio": None, "p_value": None, "status": "statsmodels_not_installed"}

    import warnings
    import numpy as np

    x_num = pd.to_numeric(x, errors="coerce")
    y_bin = pd.to_numeric(y, errors="coerce").where(y.isin([0, 1]))

    mask = x_num.notna() & y_bin.notna()
    xv = x_num.loc[mask].values
    yv = y_bin.loc[mask].values

    if len(np.unique(yv)) < 2:
        return {"beta": None, "odds_ratio": None, "p_value": None, "status": "single_class"}

    X = sm.add_constant(xv)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            model = sm.Logit(yv, X).fit(disp=False)

            # Check for perfect separation warning
            for warn in w:
                if isinstance(warn.message, PerfectSeparationWarning):
                    return {
                        "beta": None,
                        "odds_ratio": None,
                        "p_value": None,
                        "status": "perfect_separation"
                    }

            beta = float(model.params[1])
            pval = float(model.pvalues[1])
            odds_ratio = float(np.exp(beta))

            return {
                "beta": beta,
                "odds_ratio": odds_ratio,
                "p_value": pval,
                "status": "ok"
            }

        except (PerfectSeparationError, np.linalg.LinAlgError, OverflowError) as e:
            return {
                "beta": None,
                "odds_ratio": None,
                "p_value": None,
                "status": f"failed: {str(e)}"
            }
# ----------------------------
# Categorical vs Binary metrics
# ----------------------------

def chi_squared_test(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    """
    Chi-squared test of independence for categorical feature vs binary target.
    """
    from scipy.stats import chi2_contingency

    y_bin = _to_binary_target(y)
    m = x.notna() & y_bin.notna()
    if m.sum() < 10:
        return {"chi2": None, "p_value": None, "dof": None, "n": int(m.sum())}

    ct = pd.crosstab(x.loc[m].astype("object"), y_bin.loc[m].astype(int))
    # Ensure both target columns exist
    for col in [0, 1]:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[[0, 1]]

    if ct.shape[0] < 2:
        return {"chi2": None, "p_value": None, "dof": None, "n": int(m.sum())}

    chi2, p, dof, _ = chi2_contingency(ct.values, correction=False)
    return {"chi2": _safe_float(chi2), "p_value": _safe_float(p), "dof": int(dof), "n": int(m.sum())}


def cramers_v(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    """
    Cramér’s V (effect size) for categorical feature vs binary target.
    """
    from scipy.stats import chi2_contingency

    y_bin = _to_binary_target(y)
    m = x.notna() & y_bin.notna()
    if m.sum() < 10:
        return {"cramers_v": None, "n": int(m.sum())}

    ct = pd.crosstab(x.loc[m].astype("object"), y_bin.loc[m].astype(int))
    for col in [0, 1]:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[[0, 1]]

    if ct.shape[0] < 2:
        return {"cramers_v": None, "n": int(m.sum())}

    chi2, _, _, _ = chi2_contingency(ct.values, correction=False)
    n = ct.values.sum()
    r, k = ct.shape
    denom = n * (min(k - 1, r - 1))
    v = math.sqrt(chi2 / denom) if denom > 0 else np.nan
    return {"cramers_v": _safe_float(v), "n": int(n)}


def information_value_woe(
    x: pd.Series,
    y: pd.Series,
    eps: float = 1e-9,
    max_bins: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Information Value (IV) and WoE table for categorical x vs binary y.

    Args:
        eps: smoothing to avoid division by zero
        max_bins: optional cap on number of categories by grouping rare ones into '__OTHER__'
    """
    y_bin = _to_binary_target(y)
    m = x.notna() & y_bin.notna()
    if m.sum() < 10:
        return {"iv": None, "woe_table": None, "n": int(m.sum())}

    xc = x.loc[m].astype("object").copy()
    yc = y_bin.loc[m].astype(int)

    if max_bins is not None and xc.nunique(dropna=False) > max_bins:
        vc = xc.value_counts(dropna=False)
        keep = set(vc.head(max_bins - 1).index)
        xc = xc.where(xc.isin(keep), "__OTHER__")

    df = pd.DataFrame({"x": xc, "y": yc})
    gb = df.groupby("x")["y"].agg(["count", "sum"])
    gb = gb.rename(columns={"sum": "bad"})
    gb["good"] = gb["count"] - gb["bad"]

    total_good = gb["good"].sum()
    total_bad = gb["bad"].sum()

    if total_good == 0 or total_bad == 0:
        return {"iv": None, "woe_table": None, "n": int(m.sum())}

    gb["dist_good"] = gb["good"] / (total_good + eps)
    gb["dist_bad"] = gb["bad"] / (total_bad + eps)

    gb["woe"] = np.log((gb["dist_good"] + eps) / (gb["dist_bad"] + eps))
    gb["iv_component"] = (gb["dist_good"] - gb["dist_bad"]) * gb["woe"]

    iv = float(gb["iv_component"].sum())
    woe_table = gb.reset_index().sort_values("woe")

    return {
        "iv": _safe_float(iv),
        "woe_table": woe_table,
        "n": int(m.sum()),
    }


def logistic_dummies_test_categorical(
    x: pd.Series,
    y: pd.Series,
    max_categories: int = 20,
) -> Dict[str, Any]:
    """
    Logistic regression for categorical feature (one-hot encoded).
    Detects separation and instability.

    Args:
        max_categories: Maximum number of dummy columns allowed. If the feature
            has more distinct values, rare categories are collapsed into
            '__OTHER__' before dummification to avoid memory blowup.
    """

    try:
        import statsmodels.api as sm
        from statsmodels.tools.sm_exceptions import PerfectSeparationWarning, PerfectSeparationError
    except Exception:
        return {"lr_p_value": None, "status": "statsmodels_not_installed"}

    import warnings
    import numpy as np
    from scipy.stats import chi2

    y_bin = pd.to_numeric(y, errors="coerce").where(y.isin([0, 1]))
    mask = x.notna() & y_bin.notna()

    if mask.sum() < 30:
        return {"lr_p_value": None, "status": "insufficient_data"}

    xc = x.loc[mask].astype(str)   # cast to str: handles intervals, mixed types
    yc = y_bin.loc[mask].astype(float)

    # Guard: collapse rare categories so dummies matrix never exceeds max_categories columns
    if xc.nunique() > max_categories:
        vc = xc.value_counts()
        keep = set(vc.head(max_categories - 1).index)
        xc = xc.where(xc.isin(keep), "__OTHER__")

    # Build dummies from filtered xc (aligned with yc) and force float dtype
    X = pd.get_dummies(xc, drop_first=True).astype(float)

    # if X is empty after dummies (single category), return early
    if X.shape[1] == 0:
        return {"lr_p_value": None, "status": "single_category"}

    X = sm.add_constant(X)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            full_model = sm.Logit(yc, X).fit(disp=False)
            null_model = sm.Logit(yc, np.ones((len(yc), 1))).fit(disp=False)

            for warn in w:
                if isinstance(warn.message, PerfectSeparationWarning):
                    return {"lr_p_value": None, "status": "perfect_separation"}

            lr_stat = 2 * (full_model.llf - null_model.llf)
            df = full_model.df_model
            pval = chi2.sf(lr_stat, df)

            return {"lr_p_value": float(pval), "status": "ok"}

        except (PerfectSeparationError, np.linalg.LinAlgError, OverflowError) as e:
            return {"lr_p_value": None, "status": f"failed: {str(e)}"}
# ----------------------------
# Class API (pattern similar to FeatureEngineer)
# ----------------------------

@dataclass
class FeatureTargetStatistics:
    """
    Compute feature-vs-target statistics for:
      - continuous vs binary target
      - categorical vs binary target

    Returns a dict or a pandas Series with metrics + interpretations.
    """

    df: pd.DataFrame
    target_col: str
    alpha: float = 0.05

    def _y(self) -> pd.Series:
        if self.target_col not in self.df.columns:
            raise KeyError(f"target_col '{self.target_col}' not found in dataframe.")
        return self.df[self.target_col]

    def compute(
        self,
        feature_col: str,
        feature_type: FeatureType,
        return_type: ReturnType = "dict",
        include_woe_table: bool = False,
        iv_max_bins: Optional[int] = None,
        lg_max_categories: int = 20,
    ) -> Union[Dict[str, Any], pd.Series]:
        """
        Compute main metrics for one feature.

        Args:
            feature_col: feature column name
            feature_type: 'continuous' or 'categorical'
            return_type: 'dict' or 'series'
            include_woe_table: if True, include woe_table DataFrame (dict output only)
            iv_max_bins: optional cap for categories in IV/WOE (rare -> '__OTHER__')
            lg_max_categories: max distinct categories before collapsing rares in the
                logistic-dummies test (prevents memory blowup on high-cardinality features)

        Returns:
            dict or pd.Series with metrics and interpretations.
        """
        if feature_col not in self.df.columns:
            raise KeyError(f"feature_col '{feature_col}' not found in dataframe.")

        x = self.df[feature_col]
        y = self._y()

        out: Dict[str, Any] = {
            "feature": feature_col,
            "feature_type": feature_type,
            "target": self.target_col,
            "target_type": "binary",
            "alpha": self.alpha,
            "metrics": {},
            "interpretation": {},
        }

        if feature_type == "continuous":
            mw = mann_whitney_u_test(x, y)
            ks = ks_statistic(x, y)
            pb = point_biserial_corr(x, y)
            auc = single_feature_auc(x, y)
            cd = cohens_d(x, y)
            lw = logistic_wald_test_continuous(x, y)

            out["metrics"].update(
                {
                    "mann_whitney_u": mw,
                    "ks": ks,
                    "point_biserial": pb,
                    "single_feature_auc": auc,
                    "cohens_d": cd,
                    "logit_wald": lw,
                }
            )

            out["interpretation"].update(
                {
                    "mann_w_u": f"{_fmt(mw.get('p_value'))} ({_interpret_pvalue(mw.get('p_value'), self.alpha)})",
                    "ks": f"KS={_fmt(ks.get('ks'))} (higher means better separation); p={ks.get('p_value'):.3e}" if ks.get('p_value') is not None else f"KS={_fmt(ks.get('ks'))} (higher means better separation); p=N/A",
                    "point_biserial": f"r_pb={_fmt(pb.get('r_pb'))} (further from 0 is stronger); {_interpret_pvalue(pb.get('p_value'), self.alpha)}",
                    "single_feature_auc": f"AUC_abs={_fmt(auc.get('auc_abs'))} ({_interpret_auc(auc.get('auc_abs'))})",
                    "cohens_d": f"d={_fmt(cd.get('d'))} ({_interpret_cohens_d(cd.get('d'))})",
                    "logit_wald": _interpret_pvalue(lw.get("p_value"), self.alpha) if lw.get("p_value") is not None else lw.get("note", ""),
                }
            )

        elif feature_type == "categorical":
            chi = chi_squared_test(x, y)
            cv = cramers_v(x, y)
            ivr = information_value_woe(x, y, max_bins=iv_max_bins)
            lg = logistic_dummies_test_categorical(x, y, max_categories=lg_max_categories)

            out["metrics"].update(
                {
                    "chi2": chi,
                    "cramers_v": cv,
                    "information_value": {
                        "iv": ivr.get("iv"),
                        "n": ivr.get("n"),
                    },
                    "logit_dummies_lr": lg,
                }
            )

            if include_woe_table:
                # keep as DataFrame in dict output
                out["metrics"]["information_value"]["woe_table"] = ivr.get("woe_table")

            out["interpretation"].update(
                {
                    "chi2": f"{_fmt(chi.get('p_value'))} ({_interpret_pvalue(chi.get('p_value'), self.alpha)})",
                    "Cramers_v": f"V={_fmt(cv.get('cramers_v'))} ({_interpret_cramers_v(cv.get('cramers_v'))})",
                    "IV": f"{_fmt(ivr.get('iv'))} ({_interpret_iv(ivr.get('iv'))})",
                    "logit_dummies_lr": f"{_fmt(lg.get('lr_p_value'))} ({_interpret_pvalue(lg.get('lr_p_value'), self.alpha)})" if lg.get("lr_p_value") is not None else lg.get("status", ""),
                }
            )
        else:
            raise ValueError("feature_type must be 'continuous' or 'categorical'.")

        if return_type == "series":
            # Flatten (no DataFrames inside Series)
            flat: Dict[str, Any] = {
                "feature": out["feature"],
                "feature_type": out["feature_type"],
                "target": out["target"],
                "alpha": out["alpha"],
            }

            # continuous
            if feature_type == "continuous":
                flat.update(
                    {
                        "mw_p": out["metrics"]["mann_whitney_u"].get("p_value"),
                        "ks": out["metrics"]["ks"].get("ks"),
                        "ks_p": out["metrics"]["ks"].get("p_value"),
                        "r_pb": out["metrics"]["point_biserial"].get("r_pb"),
                        "r_pb_p": out["metrics"]["point_biserial"].get("p_value"),
                        "auc": out["metrics"]["single_feature_auc"].get("auc"),
                        "auc_abs": out["metrics"]["single_feature_auc"].get("auc_abs"),
                        "cohens_d": out["metrics"]["cohens_d"].get("d"),
                        "logit_beta": out["metrics"]["logit_wald"].get("beta"),
                        "logit_or": out["metrics"]["logit_wald"].get("odds_ratio"),
                        "logit_p": out["metrics"]["logit_wald"].get("p_value"),
                        "interpretation": " | ".join([f"{k}: {v}" for k, v in out["interpretation"].items()]),
                    }
                )
            else:
                flat.update(
                    {
                        "chi2_p": out["metrics"]["chi2"].get("p_value"),
                        "cramers_v": out["metrics"]["cramers_v"].get("cramers_v"),
                        "iv": out["metrics"]["information_value"].get("iv"),
                        "logit_lr_p": out["metrics"]["logit_dummies_lr"].get("lr_p_value"),
                        "interpretation": " | ".join([f"{k}: {v}" for k, v in out["interpretation"].items()]),
                    }
                )

            return pd.Series(flat, name=feature_col)

        return out

    def compute_many(
        self,
        feature_cols: List[str],
        feature_type: FeatureType,
        return_type: ReturnType = "series",
        iv_max_bins: Optional[int] = None,
        lg_max_categories: int = 20,
    ) -> Union[List[Dict[str, Any]], pd.DataFrame]:
        """
        Compute metrics for multiple features of the same type.

        Args:
            lg_max_categories: passed to each compute() call; caps the number of
                dummy columns in the logistic test to prevent OOM on high-cardinality
                features (default 20).

        Returns:
          - list of dicts (return_type='dict')
          - DataFrame (return_type='series') with one row per feature
        """
        results = []
        for col in feature_cols:
            res = self.compute(
                feature_col=col,
                feature_type=feature_type,
                return_type="dict" if return_type == "dict" else "series",
                include_woe_table=False,
                iv_max_bins=iv_max_bins,
                lg_max_categories=lg_max_categories,
            )
            results.append(res)

        if return_type == "dict":
            return results  # type: ignore[return-value]

        # series -> DataFrame
        return pd.DataFrame([r.to_dict() for r in results]).set_index("feature")  # type: ignore[arg-type]