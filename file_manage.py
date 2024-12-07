import os
import pickle
import json
import ast
import pandas as pd
from config import PKL_FOLDER


def reset_files(paths):
    for file_path in paths:

        if os.path.isfile(file_path):
            os.remove(file_path)

        with open(file_path, 'w') as f:
            f.write("")


def load_pkls(file_dir):
    names = [name for name in os.listdir(file_dir) if name.endswith(
        '.pkl') and not name.startswith('full')]
    return sorted(names)


def save_dict(data, file_path, append=False):
    if not append or os.path.getsize(file_path) == 0:
        old_data = data
    else:
        with open(file_path, 'rb') as f:
            old_data = pickle.load(f)
        old_data.update(data)

    with open(file_path, 'wb') as f:
        pickle.dump(old_data, f)


def load_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_json(data, file_path):
    with open(file_path, 'w', encoding='UTF-8') as f:
        json.dump(data, f, ensure_ascii=False)


def load_json(file_path):
    with open(file_path, 'r', encoding='UTF-8') as f:
        return json.load(f)


def load_csv(file_path, sep='|', tolist=True):
    df = pd.read_csv(file_path, sep=sep)

    if not tolist:
        return df

    for col in df.columns:
        if col not in ['nctid', 'label', 'criteria']:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x))

    return df

def save_csv(file_path, df):
    df.to_csv(file_path, index=False, sep='|')

def printPkl(file):
    f = load_dict(file)

    df = pd.DataFrame.from_dict(f, orient="index")

    print(df)


def splitPkl(file_path, temp_folder=f"{PKL_FOLDER}/temp", split_size=2**15):
    full_dict = load_dict(file_path)

    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    for i, chunk_start in enumerate(range(0, len(full_dict), split_size)):
        chunk_dict = dict(list(full_dict.items())[
                          chunk_start:chunk_start + split_size])

        chunk_file = os.path.join(temp_folder, f"chunk_{i}.pkl")
        save_dict(chunk_dict, chunk_file)


def clearPklTemp(temp_path=f"{PKL_FOLDER}/temp"):
    names = load_pkls(temp_path)

    for file_name in names:

        if os.path.isfile(f"{temp_path}/{file_name}"):
            os.remove(f"{temp_path}/{file_name}")

    os.removedirs(temp_path)
