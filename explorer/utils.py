"""Data loading and chart-building helpers for the results explorer app."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DATA_DIR = Path(__file__).parent / "data"

# Consistent color mapping so a given model keeps the same color across tabs.
MODEL_COLORS = {
    "Logistic Regression": "#4C78A8",
    "XGBoost": "#F58518",
    "Random Forest": "#54A24B",
    "CNN": "#B279A2",
    "CNN (GroupNorm)": "#B279A2",
    "BatchNorm CNN": "#E45756",
    "GroupNorm CNN": "#B279A2",
}

CLASSICAL_MODELS = ["Logistic Regression", "XGBoost", "Random Forest"]
ALL_MODELS = CLASSICAL_MODELS + ["CNN"]


# ---------------------------------------------------------------------------
# Data loading (cached so CSVs are only read once per session)
# ---------------------------------------------------------------------------


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def load_classical_ml_results() -> pd.DataFrame:
    return _load("classical_ml_results.csv")


def load_classical_ml_summary() -> pd.DataFrame:
    return _load("classical_ml_summary.csv")


def load_cnn_results() -> pd.DataFrame:
    return _load("cnn_results.csv")


def load_cnn_summary() -> pd.DataFrame:
    return _load("cnn_summary.csv")


def load_cnn_vs_classical() -> pd.DataFrame:
    return _load("cnn_vs_classical.csv")


def load_error_analysis() -> pd.DataFrame:
    return _load("error_analysis.csv")


def load_fp_rate_by_machine() -> pd.DataFrame:
    return _load("fp_rate_by_machine.csv")


def load_latency_results() -> pd.DataFrame:
    return _load("latency_results.csv")


def load_latency_correction() -> pd.DataFrame:
    return _load("latency_correction.csv")


def load_compression_results() -> pd.DataFrame:
    return _load("compression_results.csv")


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def classical_metric_by_machine_chart(df: pd.DataFrame, metric: str, models: list[str]) -> go.Figure:
    """Grouped bar chart of one metric, by held-out machine, one bar group per model."""
    sub = df[df["model"].isin(models)]
    fig = px.bar(
        sub,
        x="machine",
        y=metric,
        color="model",
        barmode="group",
        color_discrete_map=MODEL_COLORS,
        labels={"machine": "Held-out machine", metric: metric.replace("_", " ").upper()},
    )
    fig.update_layout(legend_title_text="Model", height=450)
    return fig


def classical_metric_by_model_chart(df: pd.DataFrame, metric: str, models: list[str]) -> go.Figure:
    """Bar chart of mean metric per model (averaged across held-out machines)."""
    sub = df[df["model"].isin(models)]
    agg = sub.groupby("model", as_index=False)[metric].mean()
    fig = px.bar(
        agg,
        x="model",
        y=metric,
        color="model",
        color_discrete_map=MODEL_COLORS,
        labels={"model": "Model", metric: f"Mean {metric.replace('_', ' ').upper()}"},
    )
    fig.update_layout(showlegend=False, height=450)
    return fig


def batchnorm_vs_groupnorm_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing BatchNorm vs GroupNorm PR-AUC per machine, with chance line."""
    fig = go.Figure()
    fig.add_bar(x=df["machine"], y=df["batchnorm_pr_auc"], name="BatchNorm CNN",
                marker_color=MODEL_COLORS["BatchNorm CNN"])
    fig.add_bar(x=df["machine"], y=df["groupnorm_pr_auc"], name="GroupNorm CNN",
                marker_color=MODEL_COLORS["GroupNorm CNN"])
    fig.add_trace(go.Scatter(
        x=df["machine"], y=df["chance_pr_auc"], name="Chance PR-AUC",
        mode="markers", marker=dict(symbol="line-ew", size=22, color="black", line=dict(width=2)),
    ))
    fig.update_layout(
        barmode="group",
        yaxis_title="PR-AUC",
        xaxis_title="Held-out machine",
        legend_title_text="",
        height=450,
    )
    return fig


def cnn_vs_classical_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: CNN vs best classical model, per machine."""
    fig = go.Figure()
    fig.add_bar(x=df["machine"], y=df["cnn_pr_auc"], name="CNN (GroupNorm)",
                marker_color=MODEL_COLORS["CNN (GroupNorm)"])
    fig.add_bar(
        x=df["machine"], y=df["best_classical_pr_auc"], name="Best classical",
        marker_color="#4C78A8",
        text=df["best_classical_model"], textposition="outside",
    )
    fig.update_layout(
        barmode="group",
        yaxis_title="PR-AUC",
        xaxis_title="Held-out machine",
        legend_title_text="",
        height=450,
    )
    return fig


def error_rate_comparison_chart(df: pd.DataFrame, models: list[str]) -> go.Figure:
    """Raw total-wrong vs normalized unique-error-rate, side by side."""
    sub = df[df["model"].isin(models)]
    fig = go.Figure()
    fig.add_bar(x=sub["model"], y=sub["total_wrong"], name="Total wrong (raw count)",
                marker_color="#B0B0B0", yaxis="y1")
    fig.add_bar(x=sub["model"], y=sub["unique_wrong"], name="Unique wrong (raw count)",
                marker_color="#4C78A8", yaxis="y1")
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Clip count"),
        xaxis_title="Model",
        legend_title_text="",
        height=420,
    )
    return fig


def unique_error_rate_chart(df: pd.DataFrame, models: list[str]) -> go.Figure:
    sub = df[df["model"].isin(models)]
    fig = px.bar(
        sub, x="model", y="unique_error_rate", color="model",
        color_discrete_map=MODEL_COLORS,
        labels={"unique_error_rate": "Unique error rate (normalized)", "model": "Model"},
    )
    fig.update_layout(showlegend=False, height=420)
    return fig


def fp_rate_heatmap(df: pd.DataFrame, models: list[str]) -> go.Figure:
    cols = [m for m in models if m in df.columns]
    heat_df = df.set_index("machine")[cols]
    fig = px.imshow(
        heat_df.T,
        color_continuous_scale="Reds",
        zmin=0, zmax=1,
        labels=dict(x="Held-out machine", y="Model", color="FP rate"),
        text_auto=".2f",
    )
    fig.update_layout(height=350)
    return fig


def latency_chart(df: pd.DataFrame, column: str, title: str) -> go.Figure:
    fig = px.bar(
        df, x="model", y=column, color="model",
        color_discrete_map=MODEL_COLORS,
        labels={column: title, "model": "Model"},
    )
    fig.update_layout(showlegend=False, height=420)
    return fig


def throughput_correction_chart(df: pd.DataFrame) -> go.Figure:
    """Before/after grouped bar for the benchmarking-artifact correction."""
    pivot = df.set_index("metric")
    fig = go.Figure()
    fig.add_bar(
        x=["Classical (avg)", "CNN (GroupNorm)"],
        y=[pivot.loc["classical_avg_throughput_clips_per_sec", "before_fix"],
           pivot.loc["cnn_throughput_clips_per_sec", "before_fix"]],
        name="Before fix (redundant STFT bug)",
        marker_color="#E45756",
    )
    fig.add_bar(
        x=["Classical (avg)", "CNN (GroupNorm)"],
        y=[pivot.loc["classical_avg_throughput_clips_per_sec", "after_fix"],
           pivot.loc["cnn_throughput_clips_per_sec", "after_fix"]],
        name="After fix",
        marker_color="#54A24B",
    )
    fig.update_layout(
        barmode="group",
        yaxis_title="Throughput (clips/sec)",
        legend_title_text="",
        height=420,
    )
    return fig


def compression_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    fig = px.bar(
        df, x="variant", y=metric,
        labels={metric: metric.replace("_", " ").title(), "variant": "Variant"},
        color="variant",
    )
    fig.update_layout(showlegend=False, height=420, xaxis_tickangle=-20)
    return fig
