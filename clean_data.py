import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path.cwd()


def main():
    """ Clean the data from the clinical dataset

    Here Unnecessary columns are removed, and unique ID's for patients
    are stored for later use
    """
    # -------------------------------------------------------------------------
    filepath = PROJECT_ROOT / 'data' / 'megasample.csv'

    output_filename = 'megasample_cleaned.csv'
    output_dir = PROJECT_ROOT / 'outputs'
    # -------------------------------------------------------------------------

    db = pd.read_csv(filepath)
    db.drop(['X', 'ID', 'Unnamed: 0', 'subID'], axis=1, inplace=True)
    db.drop(db[db['sample'] == 'utrecht'].index, inplace=True)
    db.to_csv(output_dir / output_filename, index=False)


if __name__ == "__main__":
    main()
