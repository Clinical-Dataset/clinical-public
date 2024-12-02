from preprocess.file_manage import load_csv, save_json
from config import OUTPUT_FOLDER, JSON_FOLDER
from tqdm import tqdm


def identify_unique(target_file="output.csv"):
    target_file=f"{OUTPUT_FOLDER}/{target_file}"
    df = load_csv(target_file)

    columns_to_extract = [
        "gender", "phase", "locations", "countries",
        "states", "cities", "intervention",
        "drug", "condition", "reason"
    ]

    unique_values = {}

    for col in columns_to_extract:
        if col in df.columns:
            all_values = []
            for value in df[col].dropna():
                if isinstance(value, list):
                    all_values.extend(value)

        unique_values[col] = set(map(str, all_values))

    for key, values in tqdm(unique_values.items(), desc="Saving JSON files"):
        save_json(sorted(list(values)), f"{JSON_FOLDER}/{key}.json")


if __name__ == "__main__":
    identify_unique()
