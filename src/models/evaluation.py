import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)


# ─────────────────────────────────────────────────────────────
# 1. SCORING  — extract probabilities from any binary classifier
# ─────────────────────────────────────────────────────────────

def get_scores(model, X_test: pd.DataFrame) -> np.ndarray:
    """
    Returns P(fraud) for every row in X_test.
    Works with any sklearn-compatible binary classifier.

    Args:
        model  : Fitted sklearn model (predict_proba or decision_function).
        X_test : Feature matrix matching what the model was trained on.

    Returns:
        y_scores : 1-D array of shape (n_samples,) with values in [0, 1].
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        raw = model.decision_function(X_test)
        span = raw.max() - raw.min()
        return (raw - raw.min()) / (span + 1e-9)

    raise ValueError("Model has neither predict_proba nor decision_function.")


# ═══════════════════════════════════════════════════════════════
#  COMPUTE LAYER  — pure data, no plots, fully testable
# ═══════════════════════════════════════════════════════════════

def compute_pr_stats(y_true: np.ndarray, y_scores: np.ndarray) -> dict:
    """
    Computes Precision-Recall curve data, AUC-PR, ROC-AUC, and the
    threshold that maximises F1.

    Returns:
        precision, recall, thresholds : sklearn PR curve arrays
        auc_pr       : Average Precision score
        roc_auc      : ROC-AUC score
        f1           : F1 at each threshold point
        best_f1_threshold, best_f1, best_idx
        baseline     : fraud rate (random classifier reference)
    """
    y_true = np.array(y_true)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auc_pr  = average_precision_score(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    best_idx       = int(np.argmax(f1))
    best_threshold = float(thresholds[best_idx])
    best_f1        = float(f1[best_idx])

    return dict(
        precision=precision,
        recall=recall,
        thresholds=thresholds,
        f1=f1,
        auc_pr=auc_pr,
        roc_auc=roc_auc,
        best_f1_threshold=best_threshold,
        best_f1=best_f1,
        best_idx=best_idx,
        baseline=float(y_true.mean()),
    )


def compute_classification_report(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> dict:
    """
    Computes a classification report at a given decision threshold.

    Args:
        y_true     : True binary labels.
        y_scores   : Predicted probabilities.
        threshold  : Decision cutoff (float in [0, 1]).

    Returns:
        dict:
            report_dict  – classification_report as a nested dict (output_dict=True)
            report_str   – formatted string version for printing
            threshold    – echoed back
            y_pred       – binary predictions at the given threshold
    """
    y_true = np.array(y_true)
    y_pred = (y_scores >= threshold).astype(int)

    return dict(
        report_dict=classification_report(y_true, y_pred, digits=4, output_dict=True),
        report_str=classification_report(y_true, y_pred, digits=4),
        threshold=threshold,
        y_pred=y_pred,
    )


def print_classification_report(stats: dict, model_name: str = "Model") -> None:
    """Prints the classification report from compute_classification_report()."""
    print(f"\n{model_name} — Classification Report  (threshold = {stats['threshold']:.4f})")
    print("─" * 62)
    print(stats['report_str'])


def compute_profit_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    monto: np.ndarray | None = None,
    margin: float = 0.25,
    fraud_loss: float = 1.00,
    n_thresholds: int = 300,
) -> dict:
    """
    Sweep thresholds and compute the model's cost function (M) at each threshold.

    Optimizing: M = - (fraud_loss * FN) - (margin * FP)

    Profit components (Model Impact View):
      - Approve good (TN): 0 (baseline revenue, unaffected by model action)
      - Approve fraud (FN): - fraud_loss * amount (loss incurred)
      - Block good (FP): - margin * amount (missed opportunity, lost cash)
      - Block fraud (TP): 0 (avoided loss, no new cash generated)

    FN:FP cost ratio = fraud_loss / margin = 4:1 (default params).
    """
    y_true   = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    if monto is None:
        amounts      = np.ones(len(y_true), dtype=float)
        amount_label = "(count proxy)"
    else:
        amounts      = np.asarray(monto).astype(float)
        amount_label = "($)"

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    profits    = np.empty_like(thresholds, dtype=float)

    for i, t in enumerate(thresholds):
        y_pred = (y_scores >= t).astype(int)   # 1 => block, 0 => approve

        fp = (y_pred == 1) & (y_true == 0)     # blocked legit  (-margin)
        fn = (y_pred == 0) & (y_true == 1)     # approved fraud (-loss)

        profits[i] = (
            - margin     * amounts[fp].sum()
            - fraud_loss * amounts[fn].sum()
        )

    best_idx = int(np.argmax(profits))
    return dict(
        thresholds=thresholds,
        profits=profits,
        best_profit_threshold=float(thresholds[best_idx]),
        best_profit=float(profits[best_idx]),
        best_idx=best_idx,
        amount_label=amount_label,
        margin=margin,
        fraud_loss=fraud_loss,
    )


def compute_confusion_matrix_data(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> dict:
    """
    Computes confusion matrix and cell annotations at a given threshold.

    Returns:
        cm          : 2×2 numpy array [[TN, FP], [FN, TP]]
        z_text      : cell labels with count + row-% (for heatmap annotation)
        labels      : axis labels
        threshold   : echoed back
        tn, fp, fn, tp, precision, recall : scalar metrics
    """
    y_true = np.array(y_true)
    y_pred = (y_scores >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)

    labels = ['Legit (0)', 'Fraud (1)']
    z_text = [
        [f"{cm[i,j]}<br>({cm[i,j] / cm[i].sum():.1%})" for j in range(2)]
        for i in range(2)
    ]

    tn, fp, fn, tp = cm.ravel()
    return dict(
        cm=cm,
        z_text=z_text,
        labels=labels,
        threshold=threshold,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        precision=float(tp / (tp + fp + 1e-9)),
        recall=float(tp / (tp + fn + 1e-9)),
    )


def compute_fbeta_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    beta: float = 2.0,
    n_thresholds: int = 300,
) -> dict:
    """
    Sweeps thresholds and computes F-beta at each point.
    β=2 weights recall 4× more than precision, matching the 4:1 cost structure.

    Returns:
        thresholds, fbeta_vals : sweep arrays
        beta                   : echoed back
        best_fbeta_threshold, best_fbeta
    """
    y_true_arr = np.array(y_true)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    fbeta_vals = np.empty(len(thresholds))

    for i, t in enumerate(thresholds):
        y_pred = (y_scores >= t).astype(int)
        tp   = ((y_pred == 1) & (y_true_arr == 1)).sum()
        fp   = ((y_pred == 1) & (y_true_arr == 0)).sum()
        fn   = ((y_pred == 0) & (y_true_arr == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        fbeta_vals[i] = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-9)

    best_idx       = int(np.argmax(fbeta_vals))
    best_threshold = float(thresholds[best_idx])
    best_fbeta     = float(fbeta_vals[best_idx])

    return dict(
        thresholds=thresholds,
        fbeta_vals=fbeta_vals,
        beta=beta,
        best_fbeta_threshold=best_threshold,
        best_fbeta=best_fbeta,
        best_idx=best_idx,
    )


# ═══════════════════════════════════════════════════════════════
#  PLOT LAYER  — receives a stats-dict from a compute_* function
# ═══════════════════════════════════════════════════════════════

def plot_pr_curve(
    stats: dict,
    model_name: str = "Model",
    highlight_thresholds: list[tuple] | None = None,
) -> None:
    """Plots the PR curve from the result of compute_pr_stats().

    Args:
        stats                : dict returned by compute_pr_stats().
        model_name           : Label for the plot title.
        highlight_thresholds : optional list of (threshold, label) to mark on the curve.
                               Each threshold is looked up in stats['thresholds'] to find
                               the corresponding (recall, precision) point.
    """
    precision  = stats['precision']
    recall     = stats['recall']
    thresholds = stats['thresholds']
    best_idx   = stats['best_idx']
    auc_pr     = stats['auc_pr']
    best_f1    = stats['best_f1']
    best_t     = stats['best_f1_threshold']
    baseline   = stats['baseline']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines', name=f'{model_name}  (AUC-PR={auc_pr:.4f})',
        line=dict(color='steelblue', width=2),
        hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>',
    ))
    fig.add_hline(
        y=baseline, line_dash='dash', line_color='gray',
        annotation_text=f'Random baseline ({baseline:.3f})',
        annotation_position='bottom right',
    )
    fig.add_trace(go.Scatter(
        x=[recall[best_idx]], y=[precision[best_idx]],
        mode='markers', name=f'Best F1={best_f1:.4f}  (t={best_t:.3f})',
        marker=dict(color='crimson', size=10, symbol='star'),
    ))

    colors = ['darkorange', 'green', 'purple', 'brown']
    for idx, (t, label) in enumerate(highlight_thresholds or []):
        nearest_idx = int(np.argmin(np.abs(thresholds - t)))
        fig.add_trace(go.Scatter(
            x=[recall[nearest_idx]], y=[precision[nearest_idx]],
            mode='markers', name=f'{label}  (t={t:.3f})',
            marker=dict(color=colors[idx % len(colors)], size=10, symbol='diamond'),
        ))

    fig.update_layout(
        title=f'{model_name} — Precision-Recall Curve',
        xaxis_title='Recall', yaxis_title='Precision',
        yaxis=dict(range=[0, 1.05]), xaxis=dict(range=[0, 1.0]),
        legend=dict(xanchor='right', x=0.99, yanchor='top', y=0.99),
        template='plotly_white', width=700, height=450,
    )
    fig.show()


def plot_profit_curve(
    stats: dict,
    model_name: str = "Model",
    highlight_thresholds: list[tuple] | None = None,
) -> None:
    """
    Plots the profit-vs-threshold curve from the result of compute_profit_curve().

    Args:
        stats                : dict returned by compute_profit_curve().
        highlight_thresholds : optional list of (threshold, label) to overlay.
    """
    thresholds   = stats['thresholds']
    profits      = stats['profits']
    best_t       = stats['best_profit_threshold']
    best_profit  = stats['best_profit']
    amount_label = stats['amount_label']

    y_min = float(profits.min())
    y_max = float(profits.max())
    step  = (y_max - y_min) * 0.10  # stagger step

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds, y=profits,
        mode='lines', name='Expected profit',
        line=dict(color='steelblue', width=2),
        hovertemplate='Threshold: %{x:.3f}<br>Profit: %{y:,.2f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=[best_t, best_t, best_t], y=[y_min, y_max, y_max],
        mode='lines+text',
        name=f'Profit-opt  t={best_t:.3f}',
        text=['', f't={best_t:.3f}', ''],
        textposition='top center',
        line=dict(color='crimson', dash='dash', width=1.5),
    ))
    colors = ['darkorange', 'green', 'purple']
    for idx, (t, label) in enumerate(highlight_thresholds or []):
        color = colors[idx % len(colors)]
        text_y = y_max - step * (idx + 1)  # stagger label position
        fig.add_trace(go.Scatter(
            x=[t, t, t], y=[y_min, text_y, y_max],
            mode='lines+text',
            name=f'{label}  (t={t:.3f})',
            text=['', label, ''],
            textposition='top center',
            line=dict(color=color, dash='dot', width=1.5),
        ))
    fig.update_layout(
        title=f'{model_name} — Profit vs Threshold  {amount_label}',
        xaxis_title='Decision Threshold',
        yaxis_title=f'Expected Profit {amount_label}',
        legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='center', x=0.5),
        margin=dict(b=90),
        template='plotly_white', width=700, height=450,
    )
    fig.show()


def plot_confusion_matrix(stats: dict, model_name: str = "Model") -> None:
    """Plots a confusion matrix heatmap from the result of compute_confusion_matrix_data()."""
    cm        = stats['cm']
    z_text    = stats['z_text']
    labels    = stats['labels']
    threshold = stats['threshold']
    tp, fp, fn, tn = stats['tp'], stats['fp'], stats['fn'], stats['tn']

    fig = ff.create_annotated_heatmap(
        z=cm[::-1],
        x=labels, y=labels[::-1],
        annotation_text=z_text[::-1],
        colorscale='Blues', showscale=True,
    )
    fig.update_layout(
        title=f'{model_name} — Confusion Matrix  (threshold = {threshold:.3f})',
        xaxis_title='Predicted', yaxis_title='Actual',
        template='plotly_white', width=500, height=420,
    )
    fig.show()
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}"
          f"  |  Precision={stats['precision']:.4f}  Recall={stats['recall']:.4f}")


def plot_fbeta_by_threshold(
    stats: dict,
    model_name: str = "Model",
    highlight_thresholds: list[tuple] | None = None,
) -> None:
    """
    Plots F-beta vs threshold from the result of compute_fbeta_curve().

    Args:
        stats                : dict returned by compute_fbeta_curve().
        highlight_thresholds : optional list of (threshold, label) to overlay.
    """
    thresholds = stats['thresholds']
    fbeta_vals = stats['fbeta_vals']
    beta       = stats['beta']
    best_t     = stats['best_fbeta_threshold']
    best_fbeta = stats['best_fbeta']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds, y=fbeta_vals,
        mode='lines', name=f'F{beta} score',
        line=dict(color='mediumseagreen', width=2),
        hovertemplate='Threshold: %{x:.3f}<br>F-beta: %{y:.4f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=[best_t, best_t, best_t], y=[0, 1.0, 1.0],
        mode='lines+text',
        name=f'Best F{beta}  t={best_t:.3f}',
        text=['', f't={best_t:.3f}', ''],
        textposition='top center',
        line=dict(color='crimson', dash='dash', width=1.5),
    ))
    colors = ['steelblue', 'darkorange', 'purple']
    for idx, (t, label) in enumerate(highlight_thresholds or []):
        color = colors[idx % len(colors)]
        text_y = 1.0 - 0.10 * (idx + 1)  # stagger label position
        fig.add_trace(go.Scatter(
            x=[t, t, t], y=[0, text_y, 1.0],
            mode='lines+text',
            name=f'{label}  (t={t:.3f})',
            text=['', label, ''],
            textposition='top center',
            line=dict(color=color, dash='dot', width=1.5),
        ))
    fig.update_layout(
        title=f'{model_name} — F{beta} Score vs Threshold',
        xaxis_title='Decision Threshold', yaxis_title=f'F{beta} Score',
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='center', x=0.5),
        margin=dict(b=90),
        template='plotly_white', width=700, height=450,
    )
    fig.show()
    print(f"Best F{beta} threshold : {best_t:.4f}  →  F{beta} = {best_fbeta:.4f}")
