
import os
import pandas as pd
from tqdm import tqdm

from config import LABEL_FILE, CSV_FOLDER, OUTPUT_FOLDER
from preprocess.file_manage import load_csv


def label_and_drop(file_name = "output.csv"):

    df = load_csv(f"{CSV_FOLDER}/full_features.csv")

    label_dict = {}

    outcome2label = {}

    # errors = []

    with open(LABEL_FILE, 'r') as fin:
        lines = fin.readlines()

        for line in lines:
            outcome, label = line.strip().split('\t')
            outcome = outcome.split(',')[-1].strip()
            outcome2label[outcome] = label

    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Labeling..."):

        nctid = row['nctid']
        outcome = row['reason']

        label = int(outcome2label.get(outcome, -1))
        label_dict[nctid] = label

        # if label == -1 and not pd.isna(outcome):
        #     errors.append([nctid,outcome])

    df['label'] = df['nctid'].map(label_dict)

    filtered_rows = [row for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Filtering...") if row['label'] >= 0]

    df = pd.DataFrame(filtered_rows).reset_index(drop=True)

    csv_path = os.path.join(OUTPUT_FOLDER, file_name)
    df.to_csv(csv_path, index=False)

    # edf = pd.DataFrame(errors).reset_index(drop=True)

    # err_path = os.path.join(CSV_FOLDER, "errs")
    # edf.to_csv(err_path, index=False)

if __name__ == "__main__":
    label_and_drop()
