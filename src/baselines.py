"""Classical ML baselines (logistic regression + XGBoost + random forest)
evaluated with 4-fold leave-one-machine-out cross-validation.

This establishes the bar a later CNN needs to beat.
"""

from typing import Iterator, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURE_COLS = [
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "zcr_mean",
    "zcr_std",
    "rms_mean",
    "rms_std",
]


def leave_one_machine_out_splits(df: pd.DataFrame) -> Iterator[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield (held_out_machine_id, train_df, test_df) for each machine_id in df.

    Each fold's train_df excludes every row whose machine_id equals the held-out
    id, so the held-out machine's recordings are never seen during fitting --
    this is what prevents leakage across machines/sessions.
    """
    for machine_id in sorted(df["machine_id"].unique()):
        test_df = df[df["machine_id"] == machine_id]
        train_df = df[df["machine_id"] != machine_id]
        yield machine_id, train_df, test_df


def _prepare_xy(df: pd.DataFrame, feature_cols=FEATURE_COLS):
    X = df[feature_cols].values
    y = (df["label"] == "abnormal").astype(int).values
    return X, y


def run_cross_validation(
    df: pd.DataFrame,
    feature_cols=FEATURE_COLS,
    random_state: int = 42,
    return_predictions: bool = False,
):
    """Run 4-fold leave-one-machine-out CV with logistic regression, XGBoost, and random forest.

    Returns one row per (machine_id_held_out, model) with f1, roc_auc, pr_auc,
    and chance_pr_auc computed on that fold's held-out machine. chance_pr_auc
    is the held-out machine's own abnormal-class prevalence -- PR-AUC's chance
    level equals the positive rate of the test set being evaluated, and that
    rate varies per machine, so the overall dataset rate (~27%) is not the
    right baseline to compare each fold against.

    If `return_predictions` is True, also returns a second dataframe with one
    row per (clip, model): filepath, machine_id, label, model, proba -- the
    held-out-fold predicted probability for every clip, for error analysis.
    Since each machine is held out exactly once, this gives exactly one
    out-of-fold prediction per clip per model.
    """
    def _record_predictions(model_name: str, proba):
        if not return_predictions:
            return
        predictions.append(
            pd.DataFrame(
                {
                    "filepath": test_df["filepath"].values,
                    "machine_id": test_df["machine_id"].values,
                    "label": test_df["label"].values,
                    "model": model_name,
                    "proba": proba,
                }
            )
        )

    results = []
    predictions = []
    for machine_id, train_df, test_df in leave_one_machine_out_splits(df):
        X_train, y_train = _prepare_xy(train_df, feature_cols)
        X_test, y_test = _prepare_xy(test_df, feature_cols)
        chance_pr_auc = y_test.mean()

        # Logistic regression is scale-sensitive; fit the scaler on the
        # training fold only so no test-fold statistics leak into training.
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = LogisticRegression(max_iter=1000, random_state=random_state)
        lr.fit(X_train_scaled, y_train)
        lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
        lr_pred = lr.predict(X_test_scaled)

        results.append(
            {
                "machine_id_held_out": machine_id,
                "model": "logistic_regression",
                "f1": f1_score(y_test, lr_pred),
                "roc_auc": roc_auc_score(y_test, lr_proba),
                "pr_auc": average_precision_score(y_test, lr_proba),
                "chance_pr_auc": chance_pr_auc,
            }
        )
        _record_predictions("logistic_regression", lr_proba)

        # XGBoost handles unscaled features natively.
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=random_state,
        )
        xgb.fit(X_train, y_train)
        xgb_proba = xgb.predict_proba(X_test)[:, 1]
        xgb_pred = xgb.predict(X_test)

        results.append(
            {
                "machine_id_held_out": machine_id,
                "model": "xgboost",
                "f1": f1_score(y_test, xgb_pred),
                "roc_auc": roc_auc_score(y_test, xgb_proba),
                "pr_auc": average_precision_score(y_test, xgb_proba),
                "chance_pr_auc": chance_pr_auc,
            }
        )
        _record_predictions("xgboost", xgb_proba)

        # Random forest, like XGBoost, is a tree-based ensemble that needs no scaling.
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=random_state,
        )
        rf.fit(X_train, y_train)
        rf_proba = rf.predict_proba(X_test)[:, 1]
        rf_pred = rf.predict(X_test)

        results.append(
            {
                "machine_id_held_out": machine_id,
                "model": "random_forest",
                "f1": f1_score(y_test, rf_pred),
                "roc_auc": roc_auc_score(y_test, rf_proba),
                "pr_auc": average_precision_score(y_test, rf_proba),
                "chance_pr_auc": chance_pr_auc,
            }
        )
        _record_predictions("random_forest", rf_proba)

    results_df = pd.DataFrame(results)
    if return_predictions:
        return results_df, pd.concat(predictions, ignore_index=True)
    return results_df


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of each metric across folds, per model."""
    return results_df.groupby("model")[["f1", "roc_auc", "pr_auc"]].agg(["mean", "std"])


def main():
    df = pd.read_csv("data/processed/scalar_features.csv")
    results = run_cross_validation(df)

    print("Per-fold results (chance_pr_auc = held-out machine's own abnormal prevalence):")
    print(results.to_string(index=False))

    print("\nSummary (mean / std across 4 folds), PR-AUC is primary given class imbalance:")
    print(summarize_results(results))

    at_or_below_chance = results[results["pr_auc"] <= results["chance_pr_auc"]]
    print("\nFolds at or below their own chance-level PR-AUC:")
    if at_or_below_chance.empty:
        print("  none")
    else:
        for _, row in at_or_below_chance.iterrows():
            print(
                f"  FLAG: {row['model']} on held-out {row['machine_id_held_out']} -- "
                f"pr_auc={row['pr_auc']:.3f} <= chance_pr_auc={row['chance_pr_auc']:.3f}"
            )

    rf_id02 = results[(results["model"] == "random_forest") & (results["machine_id_held_out"] == "id_02")].iloc[0]
    xgb_id02 = results[(results["model"] == "xgboost") & (results["machine_id_held_out"] == "id_02")].iloc[0]
    print("\nid_02 diagnosis -- does random forest also fail like XGBoost, or hold up like logistic regression?")
    print(
        f"  xgboost:       pr_auc={xgb_id02['pr_auc']:.3f} vs chance={xgb_id02['chance_pr_auc']:.3f} "
        f"-> {'AT/BELOW chance' if xgb_id02['pr_auc'] <= xgb_id02['chance_pr_auc'] else 'above chance'}"
    )
    print(
        f"  random_forest: pr_auc={rf_id02['pr_auc']:.3f} vs chance={rf_id02['chance_pr_auc']:.3f} "
        f"-> {'AT/BELOW chance' if rf_id02['pr_auc'] <= rf_id02['chance_pr_auc'] else 'above chance'}"
    )
    if rf_id02["pr_auc"] <= rf_id02["chance_pr_auc"]:
        print(
            "  => Random forest ALSO fails on id_02: this looks like a genuinely hard "
            "machine for tree-based methods, not XGBoost-specific overfitting."
        )
    else:
        print(
            "  => Random forest stays above chance where XGBoost didn't: this looks like "
            "XGBoost-specific overfitting on id_02, not a fundamentally hard machine for trees."
        )


if __name__ == "__main__":
    main()
