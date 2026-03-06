import pandas as pd


def time_based_split(
    df,
    features,
    target,
    test_size=0.2,
    val_size=None,
    strategy='split_last',
):
    """
    Performs a time-based train / (optional val) / test split.

    Args:
        df (pd.DataFrame): Must contain a 'fecha' column for sorting.
        features (list): Feature column names for X sets.
        target (str): Target column name.
        test_size (float): Proportion of data reserved for the test set.
        val_size (float | None): Proportion for validation. If None, no val split.
        strategy (str): Allocation strategy when val_size is provided.
            'split_last'  — val and test occupy consecutive tail periods:
                            [------train------][---val---][---test---]
            'same_period' — val and test are drawn from the same tail period
                            (total tail = test_size + val_size), interleaved
                            by row index so both cover the same time range:
                            [------train------][val/test interleaved]

    Returns:
        Without val : (X_train, X_test,       y_train, y_test)
        With val    : (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    df_sorted = df.sort_values('fecha')
    n = len(df_sorted)

    # ── No validation ────────────────────────────────────────────────────────
    if val_size is None:
        split_idx = int(n * (1 - test_size))
        train_df  = df_sorted.iloc[:split_idx]
        test_df   = df_sorted.iloc[split_idx:]

        X_train, y_train = train_df[features], train_df[target]
        X_test,  y_test  = test_df[features],  test_df[target]

        print("Time-based split (no validation):")
        print(f"  Train : {X_train.shape}")
        print(f"  Test  : {X_test.shape}")
        return X_train, X_test, y_train, y_test

    # ── With validation ───────────────────────────────────────────────────────
    if strategy == 'split_last':
        # Chronological order: [train][val][test]
        train_end = int(n * (1 - test_size - val_size))
        val_end   = int(n * (1 - test_size))
        train_df  = df_sorted.iloc[:train_end]
        val_df    = df_sorted.iloc[train_end:val_end]
        test_df   = df_sorted.iloc[val_end:]

    elif strategy == 'same_period':
        # Same tail period for both; interleave by row (even→val, odd→test)
        train_end = int(n * (1 - test_size - val_size))
        train_df  = df_sorted.iloc[:train_end]
        held_df   = df_sorted.iloc[train_end:]
        val_df    = held_df.iloc[::2]   # even rows
        test_df   = held_df.iloc[1::2]  # odd rows

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose 'split_last' or 'same_period'."
        )

    X_train, y_train = train_df[features], train_df[target]
    X_val,   y_val   = val_df[features],   val_df[target]
    X_test,  y_test  = test_df[features],  test_df[target]

    print(f"Time-based split (strategy='{strategy}'):")
    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test
