# preprocessing/visualization.py

from __future__ import annotations

import os
from typing import Iterable, Optional, Literal, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


FeatureType = Literal["numeric", "categorical"]


def _wrap_annotation(text: Optional[str], max_chars: int = 100) -> str:
    """Insert <br> breaks into annotation text at word boundaries to avoid overflow."""
    if not text:
        return text or ""
    words = text.split(" ")
    lines, current = [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return "<br>".join(lines)


def _ensure_dir(path: str) -> None:
    """Create a folder if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def _infer_feature_type(df: pd.DataFrame, feature: str, max_unique_for_cat: int = 25) -> FeatureType:
    """Infer feature type using dtype and cardinality heuristics."""
    s = df[feature]
    if pd.api.types.is_numeric_dtype(s):
        # Numeric but very low cardinality often behaves like categorical (e.g., 0/1/2)
        nunique = s.nunique(dropna=True)
        return "categorical" if nunique <= max_unique_for_cat else "numeric"
    return "categorical"

def plot_numeric_feature(
    df: pd.DataFrame,
    feature: str,
    target: str = "fraude",
    nbins: int = 30,
    title: Optional[str] = None,
    show_points: bool = True,
    max_points_per_class: int = 2500,
    jitter: float = 0.18,
    seed: int = 42,
    annotations_text: Optional[str] = "---",
    annotation_font_size: int = 11,
) -> go.Figure:
    """Plot numeric feature: left=dual-axis binned counts, right=boxplot + jittered points."""
    d = df[[feature, target]].copy()
    x = pd.to_numeric(d[feature], errors="coerce")
    y = pd.to_numeric(d[target], errors="coerce")

    mask_valid = x.notna() & y.notna()
    x = x[mask_valid]
    y = y[mask_valid]

    # Binary 0/1 check
    y_unique = pd.Series(y).dropna().unique()
    is_binary_01 = False
    if len(y_unique) == 2:
        try:
            vals = set(pd.Series(y_unique).astype(float).tolist())
            is_binary_01 = vals.issubset({0.0, 1.0})
        except Exception:
            is_binary_01 = False

    # NOTE: remove subplot titles to avoid top overlap; we keep a single main title
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"secondary_y": True}, {}]],
        horizontal_spacing=0.20,
    )

    if is_binary_01:
        x0 = x[y == 0]
        x1 = x[y == 1]

        # LEFT: transparent overlays
        fig.add_trace(
            go.Histogram(
                x=x0,
                nbinsx=nbins,
                name=f"{target}=0",
                opacity=0.40,
                marker=dict(line=dict(width=0)),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Histogram(
                x=x1,
                nbinsx=nbins,
                name=f"{target}=1",
                opacity=0.40,
                marker=dict(line=dict(width=0)),
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title_text=feature, row=1, col=1)
        fig.update_yaxes(title_text=f"Count ({target}=0)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text=f"Count ({target}=1)", row=1, col=1, secondary_y=True)

        # RIGHT: boxplots at numeric x = 0 and 1 (so jittered points always align)
        # RIGHT: boxplots WITH real points (no sampling, no manual jitter/scatter)
        fig.add_trace(
            go.Box(
                y=x0,
                name=f"{target}=0",
                boxpoints="outliers", # "all", "outliers", "suspectedoutliers", or False.
                jitter=0.25,
                pointpos=0,
                marker=dict(size=4, opacity=0.55),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Box(
                y=x1,
                name=f"{target}=1",
                boxpoints="outliers",
                jitter=0.25,
                pointpos=0,
                marker=dict(size=4, opacity=0.55),
            ),
            row=1,
            col=2,
        )
        # RIGHT: points on top of boxes
        if show_points:
            rng = np.random.default_rng(seed)

            def _sample(arr: np.ndarray, n: int) -> np.ndarray:
                if len(arr) <= n:
                    return arr
                idx = rng.choice(len(arr), size=n, replace=False)
                return arr[idx]

            # pts0 = _sample(x0.to_numpy(), max_points_per_class)
            # pts1 = _sample(x1.to_numpy(), max_points_per_class)
            pts0 = x0.to_numpy()
            pts1 = x1.to_numpy()

            xj0 = 0 + rng.uniform(-jitter, jitter, size=len(pts0))
            xj1 = 1 + rng.uniform(-jitter, jitter, size=len(pts1))

        fig.update_xaxes(
            title_text=f"{target} (0/1)",
            row=1,
            col=2,
            tickmode="array",
            tickvals=[0, 1],
            ticktext=[f"{target}=0", f"{target}=1"],
            range=[-0.5, 1.5],
            type="linear",  # important: keep numeric axis for jitter
        )
        fig.update_yaxes(title_text=feature, row=1, col=2)

    else:
        # fallback (kept minimal)
        fig.add_trace(go.Histogram(x=x, nbinsx=nbins, opacity=0.4), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Box(y=x, boxpoints=False), row=1, col=2)
        fig.update_xaxes(title_text=feature, row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text=feature, row=1, col=2)

    # ---- Styling (no blue background) ----
    fig.update_layout(
        height=460,
        width = 900,
        margin=dict(l=40, r=0, t=120, b=110),
        paper_bgcolor="white",
        plot_bgcolor="white",
        barmode="overlay",
        # Legend moved to bottom to prevent overlap
        # legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,   # move further down
            xanchor="center",
            x=0.5
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")

    # ---- Centered big title with shadow (no overlap) ----
    main_title = title or f"Numeric Feature: {feature}"

    fig.add_annotation(
        x=0.5,
        y=1.50,
        xref="paper",
        yref="paper",
        text=f"<b>{main_title}</b>",
        showarrow=False,
        font=dict(size=24, color="black"),
        xanchor="center",
        yanchor="top",
    )

    fig.add_annotation(
        x=0.5,
        y=1.30,
        xref="paper",
        yref="paper",
        text=_wrap_annotation(annotations_text),
        showarrow=False,
        font=dict(size=annotation_font_size, color="dimgray"),
        xanchor="center",
        yanchor="top",
        align="center",
    )

    # Optional: small panel titles inside each subplot area (avoids top overlap)
    # fig.add_annotation(x=0.22, y=1.02, xref="paper", yref="paper", text=f"{feature} by {target} — dual y-axes",
    #                    showarrow=False, font=dict(size=14, color="black"))
    # fig.add_annotation(x=0.78, y=1.02, xref="paper", yref="paper", text=f"{feature} vs {target} — boxplot + points",
    #                    showarrow=False, font=dict(size=14, color="black"))

    fig.add_annotation(
        x=0.23,
        y=1.1,
        xref="paper",
        yref="paper",
        text=f"<b>histogram</b>",
        showarrow=False,
        font=dict(size=14),
        xanchor="center",
    )

    fig.add_annotation(
        x=0.77,
        y=1.1,
        xref="paper",
        yref="paper",
        text=f"<b>boxplot + points",
        showarrow=False,
        font=dict(size=14),
        xanchor="center",
    )

    return fig

def plot_categorical_feature(
    df: pd.DataFrame,
    feature: str,
    target: str = "fraude",
    top_k: int = 20,
    n_classes_right: int = 2,
    title: Optional[str] = None,
    annotations_text: Optional[str] = "---",
    annotation_font_size: int = 11,
) -> go.Figure:
    """Categorical plot: left=stacked counts (non-fraud + fraud top segment), right=overall + top-N fraud-rate bars."""
    d = df[[feature, target]].copy()
    # Use .where() to avoid pandas FutureWarning on downcasting during fillna
    s = d[feature].where(d[feature].notna(), other="__MISSING__").astype("object")
    y = pd.to_numeric(d[target], errors="coerce")

    mask_valid = y.notna()
    s = s[mask_valid]
    y = y[mask_valid]

    # Ensure binary 0/1
    y_unique = pd.Series(y).dropna().unique()
    is_binary_01 = False
    if len(y_unique) == 2:
        try:
            vals = set(pd.Series(y_unique).astype(float).tolist())
            is_binary_01 = vals.issubset({0.0, 1.0})
        except Exception:
            is_binary_01 = False
    if not is_binary_01:
        raise ValueError("This layout expects a binary target encoded as 0/1.")

    df_tmp = pd.DataFrame({"cat": s, "y": y.astype(int)})

    # Keep top_k categories (others -> __OTHER__)
    vc = df_tmp["cat"].value_counts(dropna=False)
    keep = set(vc.head(top_k).index.tolist())
    df_tmp["cat"] = df_tmp["cat"].where(df_tmp["cat"].isin(keep), "__OTHER__")

    agg = (
        df_tmp.groupby("cat")
        .agg(total=("y", "size"), fraud=("y", "sum"))
        .assign(non_fraud=lambda t: t["total"] - t["fraud"])
        .assign(rate=lambda t: t["fraud"] / t["total"].replace(0, np.nan))
        .fillna(0.0)
        .sort_values("total", ascending=False)
    )

    cats = agg.index.tolist()
    cats_str = [str(c) for c in cats]  # Convert numeric categories to strings for consistent x-axis
    non_fraud = agg["non_fraud"].to_numpy()
    fraud = agg["fraud"].to_numpy()
    total = agg["total"].to_numpy()

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.20)

    # -------------------------
    # LEFT: stacked count bars
    # -------------------------
    base_color = "#6FA8DC"      # non-fraud (light)
    fraud_color = "#1F4E79"     # fraud (darker)

    fig.add_trace(
        go.Bar(
            x=cats_str,
            y=non_fraud,
            name=f"{target}=0",
            marker_color=base_color,
            opacity=0.85,
            text=[f"Non-Fraud: {int(v)}" if v > 0 else "" for v in non_fraud],
            textposition="auto",
            textfont=dict(color="white", size=11),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=cats_str,
            y=fraud,
            name=f"{target}=1",
            marker_color=fraud_color,
            opacity=0.95,
            text=[f"Fraud: {int(v)}" if v > 0 else "" for v in fraud],
            textposition="outside",
            textfont=dict(color=fraud_color , size=11),
        ),
        row=1,
        col=1,
    )

    fig.update_layout(barmode="stack")

    # Add headroom
    ymax = float(total.max()) if len(total) else 1.0
    fig.update_yaxes(range=[0, ymax * 1.15], title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text=feature, tickangle=35, row=1, col=1)

    # ---------------------------------
    # RIGHT: overall + top-N categories
    # ---------------------------------

    overall_rate = float(df_tmp["y"].mean()) if len(df_tmp) else 0.0

    cats_str = [str(c) for c in cats]  # Convert numeric categories to strings for consistent x-axis
    labels = ["Overall"] + cats_str
    rate_values = [overall_rate] + [float(agg.loc[c, "rate"]) for c in cats]

    colors = ["#8B0000"] + ["#F2A3A3"] * len(cats)  # overall dark, categories lighter

    fig.add_trace(
        go.Bar(
            x=labels,
            y=rate_values,
            marker_color=colors,
            text=[f"{r:.2%}" for r in rate_values],
            textposition="outside",
            cliponaxis=False,
            name="Fraud rate",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    rmax = max(rate_values) if len(rate_values) else 0.0
    fig.update_yaxes(
        title_text="Fraud rate",
        tickformat=".1%",
        range=[0, (rmax * 1.25) if rmax > 0 else 0.05],
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Group", tickangle=35, row=1, col=2)
    # -----------------
    # Styling / layout
    # -----------------
    fig.update_layout(
        height=460,
        width=900,
        margin=dict(l=40, r=0, t=120, b=110),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")

    # Titles / annotations
    main_title = title or f"Categorical Feature: {feature}"

    fig.add_annotation(
        x=0.5, y=1.50, xref="paper", yref="paper",
        text=f"<b>{main_title}</b>",
        showarrow=False, font=dict(size=24, color="black"),
        xanchor="center", yanchor="top",
    )
    fig.add_annotation(
        x=0.5, y=1.30, xref="paper", yref="paper",
        text=_wrap_annotation(annotations_text),
        showarrow=False, font=dict(size=annotation_font_size, color="dimgray"),
        xanchor="center", yanchor="top", align="center",
    )
    fig.add_annotation(
        x=0.23, y=1.1, xref="paper", yref="paper",
        text="<b>counts (stacked: non-fraud + fraud)</b>",
        showarrow=False, font=dict(size=14),
        xanchor="center",
    )
    fig.add_annotation(
        x=0.77, y=1.1, xref="paper", yref="paper",
        text="<b>fraud rate (overall + top categories)</b>",
        showarrow=False, font=dict(size=14),
        xanchor="center",
    )

    return fig

def plot_feature(
    df: pd.DataFrame,
    feature: str,
    target: str = "fraude",
    out_dir: str = "results",
    save_html: bool = True,
    save_png: bool = True,
    annotations_text: Optional[str] = None,
    dtype: str = None,          # <-- new param
    **kwargs,
) -> go.Figure:
    """Route to the correct plot function and save."""

    # resolve dtype
    resolved = dtype if dtype is not None else _infer_feature_type(df, feature)

    try:
        assert resolved in ("categorical", "continuous")
    except AssertionError:
        raise ValueError(
            f"Resolved dtype must be 'categorical' or 'continuous', got: {resolved!r}"
        )

    if resolved == "continuous":
        fig = plot_numeric_feature(
            df=df, feature=feature, target=target,
            annotations_text=annotations_text, **kwargs,
        )
    else:
        fig = plot_categorical_feature(
            df=df, feature=feature, target=target,
            annotations_text=annotations_text, **kwargs,
        )

    # save
    _ensure_dir(out_dir)
    base = os.path.join(out_dir, feature)
    if save_html:
        fig.write_html(f"{base}.html")
    if save_png:
        fig.write_image(f"{base}.png", scale=2)

    return fig

def plot_time_series(
    df: pd.DataFrame,
    x_time: str,
    y_primary: Sequence[str] | str,
    y_secondary: Optional[Sequence[str] | str] = None,
    fraude_col: str = "fraude",
    title: Optional[str] = None,
    annotations_text: Optional[str] = "---",
    annotation_font_size: int = 11,
) -> go.Figure:
    """Time series: primary/secondary lines + points colored by fraude (0/1)."""
    # --- accept single string or list ---
    if isinstance(y_primary, str):
        y_primary = [y_primary]
    if isinstance(y_secondary, str):
        y_secondary = [y_secondary]

    if x_time not in df.columns:
        raise KeyError(f"'{x_time}' not found in df.")
    for c in list(y_primary) + list(y_secondary or []):
        if c not in df.columns:
            raise KeyError(f"'{c}' not found in df.")
    if fraude_col not in df.columns:
        raise KeyError(f"'{fraude_col}' not found in df.")

    d = df[[x_time, fraude_col, *y_primary, *(y_secondary or [])]].copy()
    d[x_time] = pd.to_datetime(d[x_time], errors="coerce")
    d = d.dropna(subset=[x_time]).sort_values(x_time)

    # numeric fraude for coloring
    d["_fraude_num"] = pd.to_numeric(d[fraude_col], errors="coerce")
    d = d.dropna(subset=["_fraude_num"])

    # binary 0/1 check
    u = pd.Series(d["_fraude_num"]).dropna().unique()
    is_binary_01 = False
    if len(u) == 2:
        try:
            is_binary_01 = set(pd.Series(u).astype(float).tolist()).issubset({0.0, 1.0})
        except Exception:
            is_binary_01 = False

    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"secondary_y": True}]],
    )

    # ---- Lines (primary) ----
    for col in y_primary:
        yv = pd.to_numeric(d[col], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=d[x_time],
                y=yv,
                mode="lines",
                name=col,
            ),
            secondary_y=False,
        )

    # ---- Lines (secondary) ----
    if y_secondary:
        for col in y_secondary:
            yv = pd.to_numeric(d[col], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=d[x_time],
                    y=yv,
                    mode="lines",
                    name=f"{col} (sec)",
                    line=dict(dash="dot"),
                ),
                secondary_y=True,
            )

    # ---- Fraud-colored points overlay (use Scattergl for speed) ----
    if is_binary_01 and len(y_primary) > 0:
        primary_col_for_points = y_primary[0]
        y_points = pd.to_numeric(d[primary_col_for_points], errors="coerce")

        mask0 = d["_fraude_num"] == 0
        mask1 = d["_fraude_num"] == 1

        fig.add_trace(
            go.Scattergl(
                x=d.loc[mask0, x_time],
                y=y_points.loc[mask0],
                mode="markers",
                name=f"{fraude_col}=0 points",
                marker=dict(size=6, opacity=0.50),
                showlegend=False,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scattergl(
                x=d.loc[mask1, x_time],
                y=y_points.loc[mask1],
                mode="markers",
                name=f"{fraude_col}=1 points",
                marker=dict(size=7, opacity=0.75),
                showlegend=False,
            ),
            secondary_y=False,
        )

    # ---- Styling (same layout style) ----
    fig.update_layout(
        height=460,
        width=900,
        margin=dict(l=40, r=0, t=120, b=110),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", title_text=x_time)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", title_text="Primary", secondary_y=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", title_text="Secondary", secondary_y=True)

    # ---- Titles / annotations (same placement) ----
    main_title = title or f"Time Series: {x_time}"

    fig.add_annotation(
        x=0.5,
        y=1.50,
        xref="paper",
        yref="paper",
        text=f"<b>{main_title}</b>",
        showarrow=False,
        font=dict(size=24, color="black"),
        xanchor="center",
        yanchor="top",
    )

    fig.add_annotation(
        x=0.5,
        y=1.30,
        xref="paper",
        yref="paper",
        text=_wrap_annotation(annotations_text),
        showarrow=False,
        font=dict(size=annotation_font_size, color="dimgray"),
        xanchor="center",
        yanchor="top",
        align="center",
    )

    fig.add_annotation(
        x=0.5,
        y=1.1,
        xref="paper",
        yref="paper",
        text="<b>lines (primary/secondary) + fraud-colored points</b>",
        showarrow=False,
        font=dict(size=14),
        xanchor="center",
    )

    return fig


def save_all_feature_plots(
    df: pd.DataFrame,
    features: Sequence[str],
    dtype: str = None,
    target: str = "fraude",
    out_dir: str = "results",
    save_html: bool = True,
    save_png: bool = True,
    annotations_text_by_feature: dict = None,
    **kwargs,
) -> None:
    """Loop over features and save plots into results/."""
    _ensure_dir(out_dir)

    if dtype is not None:
        try:
            assert dtype in ("categorical", "continuous")
        except AssertionError:
            raise ValueError(
                f"dtype must be 'categorical' or 'continuous', got: {dtype!r}"
            )

    for feat in features:
        if feat == target:
            continue
        if feat not in df.columns:
            continue

        annotations_text = (
            annotations_text_by_feature.get(feat, "")
            if annotations_text_by_feature is not None
            else " - "
        )

        print(f"Plotting feature: {feat} (dtype={dtype or 'auto'})")

        try:
            _ = plot_feature(
                df=df,
                feature=feat,
                dtype=dtype,
                target=target,
                out_dir=out_dir,
                save_html=save_html,
                save_png=save_png,
                annotations_text=annotations_text,
                **kwargs,
            )
        except Exception as e:
            print(f"  [WARN] Skipping '{feat}': {e}")
            continue