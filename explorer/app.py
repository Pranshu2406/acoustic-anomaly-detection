"""Interactive companion to the Real-Time Acoustic Anomaly Detection report.

Explore the classical ML vs. CNN comparison on the MIMII dataset:
run with `streamlit run explorer/app.py` from the project root.
"""

import streamlit as st

import utils

st.set_page_config(
    page_title="Acoustic Anomaly Detection Explorer",
    page_icon="🔊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar: global model filter used across tabs
# ---------------------------------------------------------------------------

st.sidebar.title("🔊 Explorer controls")
st.sidebar.markdown(
    "Filter which models appear in the charts below. "
    "This selection applies across every tab."
)
selected_models = st.sidebar.multiselect(
    "Models to compare",
    options=utils.ALL_MODELS,
    default=utils.ALL_MODELS,
)
if not selected_models:
    st.sidebar.warning("Select at least one model to see charts.")
    selected_models = utils.ALL_MODELS

selected_classical = [m for m in selected_models if m in utils.CLASSICAL_MODELS]

st.sidebar.markdown("---")
st.sidebar.caption(
    "Source: *Real-Time Acoustic Anomaly Detection for Industrial Processes* — "
    "a comparative study of classical ML and a PyTorch CNN on the MIMII fan-machine "
    "dataset, evaluated with 4-fold leave-one-machine-out cross-validation."
)

st.title("Real-Time Acoustic Anomaly Detection — Results Explorer")
st.caption(
    "Classical ML vs. PyTorch CNN on the MIMII industrial-machine-sound dataset. "
    "All numbers below are pulled directly from the written project report."
)

tabs = st.tabs([
    "Overview",
    "Classical ML",
    "CNN Deep Dive",
    "Head-to-Head",
    "Error Analysis",
    "Latency & Efficiency",
    "Compression",
])

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Executive summary")
    st.markdown(
        """
This project asks whether a machine-learning system can distinguish normal from
anomalous industrial machine sounds, and compares four model families — logistic
regression, XGBoost, random forest, and a convolutional neural network (CNN) built
in PyTorch — on accuracy, error characteristics, and inference latency. Evaluation
throughout uses **leave-one-machine-out cross-validation**: each model is tested on
an entire physical machine it never saw during training, a stricter and more
realistic test of generalization than a random train/test split.

**Headline result:** the CNN and the best classical models perform in the **same
tier** — a difference well within the noise expected from only four held-out
machines. This is not a "deep learning wins" result, and the report treats that as
the honest finding rather than reframing it as a win for either side.

Three points in this project involved catching an overstated or incorrect
first-pass finding and correcting it before it became a conclusion:
- An initial CNN using **BatchNorm failed catastrophically** (below-chance accuracy)
  on two of four held-out machines — diagnosed as a real, measured input-distribution
  shift between machines, and fixed by switching to **GroupNorm**.
- An early reading of the error analysis suggested the CNN made unusually distinct
  errors. Normalizing by each model's total error count showed this was mostly an
  artifact of the CNN simply being wrong more often, not genuine error diversity.
- An early latency result showing the CNN with a **3.4x** throughput advantage was
  partly caused by a redundant-computation bug in the classical feature-extraction
  code. Fixing it reduced the advantage to a still-real **~2.4x**.
        """
    )

    cnn_vs_classical = utils.load_cnn_vs_classical()
    mean_cnn = cnn_vs_classical["cnn_pr_auc"].mean()
    mean_classical = cnn_vs_classical["best_classical_pr_auc"].mean()

    st.subheader("Headline PR-AUC comparison")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean CNN PR-AUC (GroupNorm)", f"{mean_cnn:.3f}")
    col2.metric("Mean best-classical PR-AUC", f"{mean_classical:.3f}")
    col3.metric("Gap", f"{mean_cnn - mean_classical:+.3f}", help="Within noise for n=4 held-out machines")

    wins = (cnn_vs_classical["winner"] == "CNN").sum()
    losses = (cnn_vs_classical["winner"] == "Classical").sum()
    st.info(
        f"Per-machine split: CNN wins on {wins} of 4 held-out machines, classical wins on "
        f"{losses} of 4 — a 2–2 split with all margins modest (roughly 0.03–0.09 PR-AUC)."
    )

# ---------------------------------------------------------------------------
# Classical ML
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Classical ML baselines")
    st.markdown(
        "Three classical models — Logistic Regression, XGBoost, and Random Forest — "
        "trained on 10 hand-engineered scalar audio features per clip "
        "(mean/std of spectral centroid, bandwidth, rolloff, zero-crossing rate, "
        "and RMS energy)."
    )

    metric = st.radio(
        "Metric to display", ["pr_auc", "roc_auc", "f1"],
        format_func=lambda m: m.replace("_", " ").upper(),
        horizontal=True,
    )

    classical_df = utils.load_classical_ml_results()

    st.subheader("By held-out machine")
    st.plotly_chart(
        utils.classical_metric_by_machine_chart(classical_df, metric, selected_classical),
        use_container_width=True,
    )

    st.subheader("By model (mean across held-out machines)")
    st.plotly_chart(
        utils.classical_metric_by_model_chart(classical_df, metric, selected_classical),
        use_container_width=True,
    )

    with st.expander("Notes on flagged results"):
        st.markdown(
            """
- **XGBoost on id_02** scored PR-AUC 0.244, below that fold's own chance baseline
  (0.261) — a genuine failure, diagnosed as XGBoost-specific overfitting (random
  forest scored 0.601 on the same held-out machine, confirming id_02 itself isn't
  fundamentally hard for tree methods).
- **Random Forest on id_04** shows near-zero F1 (0.006) despite healthy PR-AUC/ROC-AUC
  (0.600 / 0.711) — a threshold-calibration artifact (shallow trees produce poorly
  separated default-threshold probabilities), not a ranking failure.
            """
        )

    st.subheader("Raw data")
    st.dataframe(classical_df[classical_df["model"].isin(selected_classical)], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# CNN Deep Dive
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("CNN deep dive: the BatchNorm failure and the GroupNorm fix")

    cnn_df = utils.load_cnn_results()

    col_chart, col_narrative = st.columns([3, 2])
    with col_chart:
        st.plotly_chart(utils.batchnorm_vs_groupnorm_chart(cnn_df), use_container_width=True)
        st.caption("Black tick marks show each fold's chance-level PR-AUC.")

    with col_narrative:
        st.markdown(
            """
**5.1 — The failure.** A small CNN (3 conv+BatchNorm+ReLU+MaxPool blocks) was
trained under the same 4-fold leave-one-machine-out protocol. On two of four
held-out machines, ROC-AUC dropped **below 0.5** — worse than random guessing,
meaning predicted probabilities were systematically miscalibrated, not merely weak.

**5.2 — Diagnosis, not assumption.** Before touching the architecture:
1. Ruled out an evaluation bug (model was correctly in eval mode).
2. Confirmed the collapse was reproducible across random seeds.
3. Inspected predictions directly — the model was collapsing to predicting
   "abnormal" for nearly all clips on the affected machine, not flipping labels.
4. Measured the input distribution: z-scored stats showed the failing machines
   were **systematically louder and lower-dynamic-range** than the training
   distribution (id_06 +2.4 dB, id_02 +6.8 dB) — a measured shift, not a hypothesis.
5. Confirmed visually via spectrograms — elevated energy across the full frequency
   range with no quiet high-frequency band on the failing machine.

**5.3 — The fix.** GroupNorm was substituted for every BatchNorm layer (identical
architecture and hyperparameters otherwise), since it normalizes within each sample
rather than relying on cross-domain running statistics. This eliminated both
below-chance failures and cut cross-fold PR-AUC standard deviation from **0.323 to
0.086**, at the cost of giving up some of BatchNorm's peak performance on the one
machine (id_04) that happened to closely resemble the training distribution.
            """
        )

    st.subheader("Aggregate: mean ± std across the 4 folds")
    summary = utils.load_cnn_summary()
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Raw per-machine data")
    st.dataframe(cnn_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Head-to-Head
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("Head-to-head: CNN (GroupNorm) vs. best classical model")

    h2h_df = utils.load_cnn_vs_classical()
    st.plotly_chart(utils.cnn_vs_classical_chart(h2h_df), use_container_width=True)

    st.info(
        "**Headline finding:** a 2–2 split, all margins modest (roughly 0.03–0.09 "
        "PR-AUC). Mean CNN PR-AUC (0.674) and mean best-classical-per-machine PR-AUC "
        "(0.683) are effectively tied. With only 4 machine IDs, differences this size "
        "are well within what noise alone would produce — the honest conclusion is "
        "that these approaches are in the same performance tier on this task and "
        "dataset size, not that either reliably wins."
    )

    st.subheader("Raw data")
    st.dataframe(h2h_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Error Analysis
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("Error analysis")

    error_df = utils.load_error_analysis()
    fp_df = utils.load_fp_rate_by_machine()

    st.subheader("Raw vs. normalized unique-error rate (the corrected finding)")
    st.markdown(
        "An initial comparison of raw error counts suggested the CNN was making "
        "unusually distinct errors from the classical models. This didn't control "
        "for the CNN's higher total error count — a noisier model naturally "
        "accumulates more unique errors simply by being wrong more often. "
        "**Normalizing by each model's total error count** shows the CNN's "
        "unique-error rate (0.252) actually falls *within* the range set by the "
        "classical models (0.039–0.295), not above them."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(utils.error_rate_comparison_chart(error_df, selected_models), use_container_width=True)
    with c2:
        st.plotly_chart(utils.unique_error_rate_chart(error_df, selected_models), use_container_width=True)

    st.subheader("False-positive rate by held-out machine")
    st.plotly_chart(utils.fp_rate_heatmap(fp_df, selected_models), use_container_width=True)
    st.caption(
        "id_00 is broadly difficult across every model family; id_04 is easy for "
        "everyone. The CNN additionally collapses specifically on id_02, echoing the "
        "distribution-shift story from the CNN Deep Dive tab."
    )

    st.subheader("Raw data")
    st.dataframe(error_df[error_df["model"].isin(selected_models)], use_container_width=True, hide_index=True)
    st.dataframe(fp_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Latency & Efficiency
# ---------------------------------------------------------------------------
with tabs[5]:
    st.header("Latency and efficiency")
    st.markdown(
        "All models benchmarked on CPU (not the MPS acceleration used for CNN "
        "training), reflecting a realistic resource-constrained deployment scenario."
    )

    latency_df = utils.load_latency_results()
    latency_df_f = latency_df[latency_df["model"].isin(selected_models) | (latency_df["model"] == "CNN (GroupNorm)")]

    st.subheader("A caught benchmarking artifact: 3.4x → 2.4x")
    st.markdown(
        "An initial result showed the CNN with roughly **3.4x** higher full-pipeline "
        "throughput than the classical models. Before reporting this as an "
        "architectural finding, the classical feature-extraction code was checked for "
        "redundant computation — three of five hand-engineered features were each "
        "independently recomputing a full STFT from the raw waveform instead of "
        "sharing one. Fixing this (verified bit-identical feature values before/after) "
        "reduced the CNN's throughput advantage to a still-real **~2.4x**."
    )
    st.plotly_chart(utils.throughput_correction_chart(utils.load_latency_correction()), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model-only latency")
        st.plotly_chart(utils.latency_chart(latency_df, "model_only_latency_mean_ms", "Model-only latency, mean (ms)"), use_container_width=True)
    with col2:
        st.subheader("Model size")
        st.plotly_chart(utils.latency_chart(latency_df, "size_kb", "Size (KB)"), use_container_width=True)

    st.info(
        "Under a representative near-real-time threshold (100ms/clip), all four "
        "models pass comfortably — the worst observed p95 latency (~20ms) is roughly "
        "5x under budget. There is no accuracy-latency tradeoff forcing a choice "
        "between approaches on this task."
    )

    st.subheader("Raw data")
    st.dataframe(latency_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------
with tabs[6]:
    st.header("Compression: quantization and pruning (stretch goal)")
    st.caption(
        "Caveat: single-fold evidence. Unlike every other result in this report, "
        "these numbers come from one held-out machine (id_00) with no repeated runs, "
        "not the 4-fold cross-validation used elsewhere. Differences under roughly "
        "0.05 PR-AUC between variants should be read as noise-level, not a reliable effect."
    )

    comp_df = utils.load_compression_results()

    comp_metric = st.selectbox(
        "Metric to chart", ["pr_auc", "roc_auc", "f1", "size_kb", "latency_mean_ms"],
        format_func=lambda m: m.replace("_", " ").upper(),
    )
    st.plotly_chart(utils.compression_chart(comp_df, comp_metric), use_container_width=True)

    st.markdown(
        """
- **Quantization** (post-training static int8) gave a real, mechanism-consistent
  **65% reduction in model size** (105.2 KB → 36.4 KB), but no latency improvement
  and a small accuracy cost — GroupNorm has less mature quantization support than
  BatchNorm in PyTorch, and several GroupNorm layers fell back to fp32.
- **Pruning** (unstructured magnitude pruning at 30/50/70% sparsity) shows
  essentially flat size and latency, as expected — unstructured pruning zeroes
  weights without changing dense tensor shapes, so it doesn't reduce computation on
  standard CPU inference. What it does show is **accuracy headroom**: the network
  tolerates removing up to half its weights with no meaningful accuracy loss.
        """
    )

    st.subheader("Raw data")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
