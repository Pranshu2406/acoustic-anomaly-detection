"""Cross-model error analysis using data/processed/all_predictions.csv.

Caveat: random forest and CNN are known from earlier phases to have threshold
calibration issues at 0.5 (good ranking / PR-AUC, but predict() at 0.5 can be
skewed toward one class on some folds). False positive/negative counts below
are therefore approximate evidence of *where* each model struggles, not a
precise or final scorecard -- PR-AUC/ROC-AUC remain the more trustworthy
metrics for ranking model quality.
"""

import os

import librosa.display
import matplotlib.pyplot as plt
import pandas as pd

from src.preprocessing import HOP_LENGTH, SAMPLE_RATE, wav_to_logmel

PREDICTIONS_PATH = "data/processed/all_predictions.csv"
ERROR_LISTS_OUT = "data/processed/error_lists.csv"
HARD_CASES_PLOT_OUT = os.path.join("notebooks", "error_analysis_hard_cases.png")

MODELS = ["logistic_regression", "xgboost", "random_forest", "cnn"]
THRESHOLD = 0.5


def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_PATH)
    df["true"] = (df["label"] == "abnormal").astype(int)
    for model in MODELS:
        df[f"pred_{model}"] = (df[f"proba_{model}"] >= THRESHOLD).astype(int)
        df[f"wrong_{model}"] = (df[f"pred_{model}"] != df["true"]).astype(int)
    return df


def report_false_positives_negatives(df: pd.DataFrame):
    print("=" * 70)
    print("2. False positives / false negatives per model (threshold=0.5)")
    print("=" * 70)

    error_rows = []
    for model in MODELS:
        fp = df[(df[f"pred_{model}"] == 1) & (df["true"] == 0)]
        fn = df[(df[f"pred_{model}"] == 0) & (df["true"] == 1)]

        print(f"\n{model}: {len(fp)} false positives, {len(fn)} false negatives")
        print(f"  sample FPs: {fp['filepath'].head(3).tolist()}")
        print(f"  sample FNs: {fn['filepath'].head(3).tolist()}")

        for _, row in fp.iterrows():
            error_rows.append({"model": model, "error_type": "FP", "filepath": row["filepath"], "machine_id": row["machine_id"]})
        for _, row in fn.iterrows():
            error_rows.append({"model": model, "error_type": "FN", "filepath": row["filepath"], "machine_id": row["machine_id"]})

    error_df = pd.DataFrame(error_rows)
    error_df.to_csv(ERROR_LISTS_OUT, index=False)
    print(f"\nFull FP/FN filepath lists saved to {os.path.abspath(ERROR_LISTS_OUT)} ({len(error_df)} rows)")


def report_cross_model_agreement(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("3. Cross-model agreement analysis")
    print("=" * 70)

    def pattern(row):
        return "".join(str(row[f"wrong_{m}"]) for m in MODELS)

    df["pattern"] = df.apply(pattern, axis=1)

    print(f"\nPattern legend (order = {','.join(MODELS)}): 0=correct, 1=wrong")
    counts = df["pattern"].value_counts().sort_values(ascending=False)
    print("\nagreement pattern -> count of clips")
    for pat, count in counts.items():
        print(f"  {pat}: {count}")

    all_wrong = (df["pattern"] == "1111").sum()
    all_right = (df["pattern"] == "0000").sum()
    cnn_only_wrong = (df["pattern"] == "0001").sum()
    classical_only_wrong = (df["pattern"] == "1110").sum()

    print("\nHeadline counts:")
    print(f"  All 4 models wrong (hard/ambiguous cases): {all_wrong}")
    print(f"  All 4 models right:                        {all_right}")
    print(f"  Only CNN wrong (all 3 classical right):     {cnn_only_wrong}")
    print(f"  Only classical wrong, all 3 (CNN right):    {classical_only_wrong}")

    if cnn_only_wrong > 0 or classical_only_wrong > 0:
        if cnn_only_wrong > classical_only_wrong:
            print("  => CNN makes more genuinely distinct errors than the classical models do vs. it.")
        elif classical_only_wrong > cnn_only_wrong:
            print("  => Classical models collectively make more genuinely distinct errors than the CNN does vs. them.")
        else:
            print("  => CNN and classical models make comparably many genuinely distinct errors vs. each other.")

    return df


def report_unique_error_rate(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("3b. Unique error rate per model (normalizes for differing total error counts)")
    print("=" * 70)
    print(
        "\nunique_error_rate = (clips where ONLY this model is wrong) / (this model's total wrong count)\n"
        "Controls for CNN having a higher raw error count than the classical models, so raw\n"
        "'unique wrong' counts alone don't say whether it's finding different failure modes or\n"
        "just failing more often."
    )

    rows = []
    for i, model in enumerate(MODELS):
        total_wrong = df[f"wrong_{model}"].sum()
        only_this_wrong = (df["pattern"].str[i] == "1") & (df["pattern"].str.count("1") == 1)
        unique_wrong = only_this_wrong.sum()
        rows.append(
            {
                "model": model,
                "total_wrong": int(total_wrong),
                "unique_wrong": int(unique_wrong),
                "unique_error_rate": unique_wrong / total_wrong if total_wrong else float("nan"),
            }
        )

    rate_df = pd.DataFrame(rows)
    print()
    print(rate_df.to_string(index=False))

    cnn_rate = rate_df.loc[rate_df["model"] == "cnn", "unique_error_rate"].iloc[0]
    classical_rates = rate_df.loc[rate_df["model"] != "cnn", "unique_error_rate"]
    print(f"\nCNN unique_error_rate: {cnn_rate:.3f}")
    print(f"Classical models' unique_error_rate range: {classical_rates.min():.3f}-{classical_rates.max():.3f}")
    if cnn_rate > classical_rates.max():
        print("=> CNN's unique_error_rate is higher than ALL classical models, even after normalizing for its")
        print("   larger total error count -- it's finding genuinely different failure modes, not just failing more.")
    elif cnn_rate < classical_rates.min():
        print("=> CNN's unique_error_rate is lower than ALL classical models once normalized -- despite the")
        print("   higher raw error count, its errors overlap with the classical models' more than theirs overlap")
        print("   with each other; proportionally it is not finding more distinct failure modes.")
    else:
        print("=> CNN's unique_error_rate falls within the classical models' range once normalized -- roughly")
        print("   proportional to its higher overall error count, not evidence of disproportionately more")
        print("   distinct failure modes.")

    return rate_df


def report_per_machine_breakdown(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("4. Errors by machine_id, per model")
    print("=" * 70)

    counts = df.groupby("machine_id")[[f"wrong_{m}" for m in MODELS]].sum()
    counts.columns = MODELS
    totals = df.groupby("machine_id").size()
    counts["n_clips"] = totals

    print("\nMisclassification counts (FP+FN combined) by machine_id:")
    print(counts.to_string())

    print("\nAs a fraction of that machine's clips:")
    frac = counts[MODELS].div(counts["n_clips"], axis=0)
    print(frac.round(3).to_string())

    # Distinguish scattered errors from total threshold collapse: for each
    # model x machine, what fraction of that machine's NORMAL clips are
    # false positives? A value near 1.0 means "predicts abnormal for nearly
    # everything on this machine" (calibration collapse), not scattered
    # misclassification -- important context before reading FP/FN counts as
    # evidence of case-by-case difficulty.
    print("\nFalse-positive rate on NORMAL clips specifically, by machine_id (near 1.0 = total collapse, not scattered errors):")
    fp_rate_rows = []
    for machine_id in sorted(df["machine_id"].unique()):
        machine_normal = df[(df["machine_id"] == machine_id) & (df["label"] == "normal")]
        row = {"machine_id": machine_id, "n_normal": len(machine_normal)}
        for model in MODELS:
            row[model] = machine_normal[f"wrong_{model}"].mean() if len(machine_normal) else float("nan")
        fp_rate_rows.append(row)
    fp_rate_df = pd.DataFrame(fp_rate_rows).set_index("machine_id")
    print(fp_rate_df.round(3).to_string())


def plot_hard_cases(df: pd.DataFrame, n: int = 3):
    print("\n" + "=" * 70)
    print("5. Spectrograms of 'all 4 models wrong' cases")
    print("=" * 70)

    hard_cases = df[df["pattern"] == "1111"]
    if hard_cases.empty:
        print("No clips where all 4 models are wrong -- nothing to plot.")
        return

    by_machine_label = hard_cases.groupby(["machine_id", "label"]).size()
    print("\nComposition of the all-4-wrong set (machine_id, label -> count):")
    print(by_machine_label.to_string())

    # Most of this set is id_00/id_02 "normal" clips the CNN over-predicts as
    # abnormal almost unconditionally on those two machines (see caveat above)
    # -- not necessarily genuinely ambiguous content. Deliberately diversify
    # the sample instead of taking the first n alphabetically, so the plot
    # includes both the dominant calibration-collapse pattern (id_00 normal)
    # and a case from id_04 -- the CNN's strongest fold overall, where an
    # all-4-wrong clip is a better candidate for genuine ambiguity.
    picks = []
    id00_normal = hard_cases[(hard_cases["machine_id"] == "id_00") & (hard_cases["label"] == "normal")]
    if not id00_normal.empty:
        picks.append(id00_normal.iloc[0])
    id04_abnormal = hard_cases[(hard_cases["machine_id"] == "id_04") & (hard_cases["label"] == "abnormal")]
    if not id04_abnormal.empty:
        picks.append(id04_abnormal.iloc[0])
    id02_abnormal = hard_cases[(hard_cases["machine_id"] == "id_02") & (hard_cases["label"] == "abnormal")]
    if not id02_abnormal.empty:
        picks.append(id02_abnormal.iloc[0])

    remaining_slots = n - len(picks)
    if remaining_slots > 0:
        already_picked = pd.DataFrame(picks)["filepath"] if picks else pd.Series(dtype=str)
        extra = hard_cases[~hard_cases["filepath"].isin(already_picked)].head(remaining_slots)
        picks.extend(row for _, row in extra.iterrows())

    chosen = pd.DataFrame(picks[:n])
    print(f"\nPlotting {len(chosen)} of {len(hard_cases)} all-4-wrong clips (deliberately diversified, not alphabetical):")
    for _, row in chosen.iterrows():
        print(f"  {row['filepath']}  (true={row['label']}, machine={row['machine_id']})")

    fig, axes = plt.subplots(1, len(chosen), figsize=(6 * len(chosen), 5))
    if len(chosen) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, chosen.iterrows()):
        logmel = wav_to_logmel(row["filepath"])
        img = librosa.display.specshow(
            logmel,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            ax=ax,
            vmin=-80,
            vmax=0,
        )
        ax.set_title(f"true={row['label']} ({row['machine_id']})\nall 4 models wrong")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    fig.tight_layout()
    fig.savefig(HARD_CASES_PLOT_OUT, dpi=150)
    print(f"\nSaved to {os.path.abspath(HARD_CASES_PLOT_OUT)}")


def main():
    print(__doc__)
    df = load_predictions()
    print(f"Loaded {len(df)} clips x {len(MODELS)} models from {PREDICTIONS_PATH}\n")

    report_false_positives_negatives(df)
    df = report_cross_model_agreement(df)
    report_unique_error_rate(df)
    report_per_machine_breakdown(df)
    plot_hard_cases(df)


if __name__ == "__main__":
    main()
