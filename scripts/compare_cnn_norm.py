"""Compare the BatchNorm CNN (SmallCNN) vs. the GroupNorm variant (CNNGroupNorm)
on the same 4-fold leave-one-machine-out setup, with special attention to
whether GroupNorm resolves the below-chance collapse on id_06 and id_02.
"""

import pandas as pd

from src.cnn_model import CNNGroupNorm, SmallCNN
from src.train_cnn import run_cnn_cross_validation

WATCH_MACHINES = ("id_02", "id_06")


def main():
    batchnorm_results = run_cnn_cross_validation(model_cls=SmallCNN, model_name="batchnorm_cnn")
    groupnorm_results = run_cnn_cross_validation(model_cls=CNNGroupNorm, model_name="groupnorm_cnn")

    all_results = pd.concat([batchnorm_results, groupnorm_results], ignore_index=True)

    print("\n\nPer-fold results, both models:")
    print(all_results.sort_values(["machine_id_held_out", "model"]).to_string(index=False))

    print("\nSummary (mean / std across 4 folds), per model:")
    print(all_results.groupby("model")[["f1", "roc_auc", "pr_auc"]].agg(["mean", "std"]))

    print("\nSide-by-side, per machine (pr_auc vs. that fold's chance_pr_auc):")
    pivot = all_results.pivot(index="machine_id_held_out", columns="model", values="pr_auc")
    chance = all_results.groupby("machine_id_held_out")["chance_pr_auc"].first()
    pivot["chance_pr_auc"] = chance
    print(pivot.to_string())

    print("\nid_02 / id_06 diagnosis -- did GroupNorm bring them above chance?")
    for machine_id in WATCH_MACHINES:
        bn_row = batchnorm_results[batchnorm_results["machine_id_held_out"] == machine_id].iloc[0]
        gn_row = groupnorm_results[groupnorm_results["machine_id_held_out"] == machine_id].iloc[0]
        bn_status = "AT/BELOW chance" if bn_row["pr_auc"] <= bn_row["chance_pr_auc"] else "above chance"
        gn_status = "AT/BELOW chance" if gn_row["pr_auc"] <= gn_row["chance_pr_auc"] else "above chance"
        print(
            f"  {machine_id}: batchnorm pr_auc={bn_row['pr_auc']:.3f} ({bn_status}, chance={bn_row['chance_pr_auc']:.3f})"
            f"  |  groupnorm pr_auc={gn_row['pr_auc']:.3f} ({gn_status}, chance={gn_row['chance_pr_auc']:.3f})"
        )
        if bn_status == "AT/BELOW chance" and gn_status == "above chance":
            print(f"    => GroupNorm RESOLVED the collapse on {machine_id}")
        elif bn_status == "AT/BELOW chance" and gn_status == "AT/BELOW chance":
            print(f"    => GroupNorm did NOT resolve the collapse on {machine_id}; still at/below chance")
        else:
            print(f"    => {machine_id} was not below chance under batchnorm to begin with")


if __name__ == "__main__":
    main()
