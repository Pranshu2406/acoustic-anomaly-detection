import pandas as pd

from src.baselines import leave_one_machine_out_splits


def test_leave_one_machine_out_excludes_held_out_machine_from_training():
    df = pd.DataFrame(
        {
            "machine_id": ["id_00"] * 3 + ["id_02"] * 3 + ["id_04"] * 3 + ["id_06"] * 3,
            "label": ["normal", "abnormal", "normal"] * 4,
        }
    )

    folds = list(leave_one_machine_out_splits(df))
    assert len(folds) == 4

    all_machine_ids = set(df["machine_id"].unique())
    for held_out_id, train_df, test_df in folds:
        assert held_out_id not in train_df["machine_id"].unique()
        assert (test_df["machine_id"] == held_out_id).all()
        assert set(train_df["machine_id"].unique()) == all_machine_ids - {held_out_id}
