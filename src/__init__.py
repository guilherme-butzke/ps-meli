from pathlib import Path

#Define the raw data path
DATA_RAW_DIR = Path("dataset", "raw")
DATA_PROCESSED_DIR = Path("dataset", "processed")

DATA_RAW_FILENAME = "MercadoLibre Data Scientist Technical Challenge - Dataset.csv"
DATA_RAW_PATH = DATA_RAW_DIR / DATA_RAW_FILENAME

#Define the results directory
RESULTS_DIR = Path("notebooks","results")   

# Create the results directory if it doesn't exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
