from pathlib import Path
import pandas as pd

from . import DATA_RAW_PATH, DATA_PROCESSED_DIR, RESULTS_DIR

def load_raw_mercadolibre_dataset() -> pd.DataFrame:
    """
    Loads the MercadoLibre Data Scientist Technical Challenge - Dataset.csv
    from the standard location and returns it as a pandas DataFrame.

    Args:
        None

    Returns:        
        pd.DataFrame: The loaded dataset.
    """
    print(f"Loading raw MercadoLibre dataset from {DATA_RAW_PATH}...")
    df = pd.read_csv(DATA_RAW_PATH)
    print(f"Raw MercadoLibre dataset loaded successfully from {DATA_RAW_PATH} as a pandas DataFrame.")
    return df

def save_processed_features(df: pd.DataFrame, filename: str = "processed_features.csv") -> None:
    """
    Saves processed features to the results directory.
    
    Args:
        df (pd.DataFrame): The processed features dataframe.
        filename (str): The name of the file to save.
    
    Returns:
        None
    """
    results_path = DATA_PROCESSED_DIR / filename
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    print(f"Processed features saved to {results_path}")


def load_processed_features(filename: str = "processed_features.csv") -> pd.DataFrame:
    """
    Loads processed features from the results directory.
    
    Args:
        filename (str): The name of the file to load.
    
    Returns:
        pd.DataFrame: The loaded processed features dataframe.
    """
    results_path = DATA_PROCESSED_DIR/ filename
    df = pd.read_csv(results_path)
    print(f"Processed features loaded from {results_path}")
    return df


def load_features_descriptions(filename: str = "phase_02_manual_features_descriptions.csv") -> pd.DataFrame:
    """
    Loads the manual features suggestion table from the notebooks results directory.

    Args:
        filename (str): The name of the CSV file to load.

    Returns:
        pd.DataFrame: DataFrame with columns Variable, Source, dtype,
                      Recommend Use, Use in Tree Model, Use in Linear Model,
                      and Observations.
    """
    path = RESULTS_DIR / filename
    df = pd.read_csv(path)
    print(f"Features suggestion loaded from {path}")
    return df