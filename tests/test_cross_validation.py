from aurora.evaluation.cross_validation import generate_rolling_cv_folds


def test_generate_rolling_cv_folds() -> None:
    folds = generate_rolling_cv_folds("2025-01-01", "2025-06-30", train_months=3)

    assert len(folds) == 2
    assert folds[0].fold_id == 0
    assert folds[0].train_start.month == 1
    assert folds[0].val_start.month == 4
    assert folds[0].test_start.month == 5
