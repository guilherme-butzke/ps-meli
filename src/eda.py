from pathlib import Path
import pandas as pd
from ydata_profiling import ProfileReport

from . import RESULTS_DIR

def generate_data_profiling(df: pd.DataFrame, title: str, output_filename: str, output_dir: Path = RESULTS_DIR) -> None:
    """
    Generate a data profiling report using the ydata_profiling library.

    Args:
        df: The pandas DataFrame to profile.
        title: The title of the report.
        output_filename: The name of the output file.
        output_dir: The directory to save the report.
    """
    print(f"Generating data profiling report for {title}...")
    profile = ProfileReport(df, title=title, explorative=True)
    profile.to_file(output_dir / output_filename)
    print(f"Profile report saved to: {output_dir / output_filename}")
