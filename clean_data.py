import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd()


def main():
    """ Clean the data from the clinical dataset

    Here Unnecessary columns are removed, and unique ID's for patients
    are stored for later use
    """
    # -------------------------------------------------------------------------
    if len(sys.argv) < 2:
        print("Please provide the name of the CSV file as a command line argument.")
        return

    filename = sys.argv[1]
    filepath = PROJECT_ROOT / 'data' / filename

    output_filename = filename.split('.')[0] + '_cleaned.csv'
    output_dir = PROJECT_ROOT / 'data' / 'cleaned_data'
    # -------------------------------------------------------------------------
    # Print Terminal Output if input file is not found
    if not filepath.exists():
        print(f"File {filepath} not found!")
        return

    if not output_dir.exists():
        output_dir.mkdir()

    db = pd.read_csv(filepath)
    # check if 'X' and 'Unnamed: 0' exist and drop them if they do
    if 'X' in db.columns:
        db.drop('X', axis=1, inplace=True)
    if 'Unnamed: 0' in db.columns:
        db.drop('Unnamed: 0', axis=1, inplace=True)
    db.drop(db[db['sample'] == 'utrecht'].index, inplace=True)
    db.to_csv(output_dir / output_filename, index=False)


if __name__ == "__main__":
    main()
