from preprocess.file_manage import load_dict, save_dict, load_pkls, save_csv
from config import CSV_FOLDER, PKL_FOLDER, columns
import os
import pandas as pd
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
torch.set_num_threads(16)


def stack_features():
    file_names = load_pkls(PKL_FOLDER)
    features_dict = {}

    for name in tqdm(file_names):
        file_path = os.path.join(PKL_FOLDER, name)
        file_dict = load_dict(file_path)

        for id, data in file_dict.items():
            if id not in features_dict:
                features_dict[id] = []

            if isinstance(data, dict):
                features_dict[id].extend(data.values())
            else:
                features_dict[id].append(data)

    pkl_path = os.path.join(PKL_FOLDER, "full_features.pkl")
    save_dict(features_dict, pkl_path)

    df = pd.DataFrame.from_dict(features_dict, orient='index').reset_index()
    df.columns = columns

    print("DataFrame Structure:")
    print(df)

    csv_path = os.path.join(CSV_FOLDER, "full_features.csv")
    save_csv(csv_path, df)


if __name__ == "__main__":
    stack_features()
