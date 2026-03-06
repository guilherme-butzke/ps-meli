import json
from pathlib import Path

from sklearn.compose import ColumnTransformer

from src.preprocessing.custom_transformers import (
    BinaryNaNTransformer,
    BinnedFeaturesTransformer,
    BoxCoxTransformer,
    CategoricalEncoderWithNaN,
    DatetimeFeatureTransformer,
    FrequencyEncodingTransformer,
    InteractionTransformer,
    LogTransformer,
    MissingFlagTransformer,
    NamedPCATransformer,
    RareGroupTransformer,
    RatioTransformer,
    TopPercentileFlagTransformer,
    YeoJohnsonTransformer,
    ZeroFlagTransformer,
)
from src.preprocessing.utils import diagnose_bin_feasibility

_REGISTRY = {
    "BinaryNaNTransformer":          BinaryNaNTransformer,
    "BinnedFeaturesTransformer":     BinnedFeaturesTransformer,
    "BoxCoxTransformer":             BoxCoxTransformer,
    "CategoricalEncoderWithNaN":     CategoricalEncoderWithNaN,
    "DatetimeFeatureTransformer":    DatetimeFeatureTransformer,
    "FrequencyEncodingTransformer":  FrequencyEncodingTransformer,
    "InteractionTransformer":        InteractionTransformer,
    "LogTransformer":                LogTransformer,
    "MissingFlagTransformer":        MissingFlagTransformer,
    "NamedPCATransformer":           NamedPCATransformer,
    "RareGroupTransformer":          RareGroupTransformer,
    "RatioTransformer":              RatioTransformer,
    "TopPercentileFlagTransformer":  TopPercentileFlagTransformer,
    "YeoJohnsonTransformer":         YeoJohnsonTransformer,
    "ZeroFlagTransformer":           ZeroFlagTransformer,
}


def build_preprocessor_from_config(config_path, df_for_diagnosis=None):
    """
    Reads a JSON pipeline config and returns an unfitted ColumnTransformer.

    Each entry in config["steps"] must have:
        name   – step name (string)
        type   – transformer class name from _REGISTRY, or "passthrough"
        cols   – list of column names
        params – (optional) dict of constructor kwargs

    Special behaviour for BinnedFeaturesTransformer:
        If params.n_bins is null/missing and df_for_diagnosis is provided,
        the feasible bin count is diagnosed automatically from the data.
        In production, set n_bins explicitly to a dict to skip diagnosis.

    Args:
        config_path (str | Path): Path to the JSON config file.
        df_for_diagnosis (pd.DataFrame | None): DataFrame used to auto-diagnose
            bin feasibility. Only needed when n_bins is null in the config.

    Returns:
        sklearn.compose.ColumnTransformer (unfitted)
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    steps = []

    for step in config["steps"]:
        name   = step["name"]
        type_  = step["type"]
        cols   = step["cols"]
        params = dict(step.get("params") or {})

        if type_ == "passthrough":
            steps.append((name, "passthrough", cols))
            continue

        # Auto-diagnose bin counts when n_bins is left as null
        if type_ == "BinnedFeaturesTransformer" and params.get("n_bins") is None:
            if df_for_diagnosis is not None:
                params["n_bins"] = diagnose_bin_feasibility(
                    df_for_diagnosis,
                    cols,
                    candidate_n_bins=config.get("binner_candidate_n_bins", 10),
                    verbose=False,
                )
            else:
                del params["n_bins"]   # fall back to BinnedFeaturesTransformer default

        cls = _REGISTRY[type_]
        steps.append((name, cls(**params), cols))

    return ColumnTransformer(
        transformers=steps,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
