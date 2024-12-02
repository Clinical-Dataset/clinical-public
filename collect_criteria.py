
from config import XML_FOLDER, OUTPUT_FOLDER
from preprocess.file_manage import load_csv, reset_files, save_dict, load_dict, save_csv
from lxml import etree
from tqdm import tqdm


def extract_nctid(file):
    df = load_csv(file)

    ids = df['nctid'].tolist()

    return ids


def select_files(nctids):
    id_set = set(nctids)
    filtered_paths = []

    with open(f"{XML_FOLDER}/trials/all_xml.txt", "r", encoding="UTF-8") as file:
        xml_paths = [line.strip() for line in file]

    for path in tqdm(xml_paths, desc="Filtering Paths", total=len(xml_paths)):
        for nctid in id_set.copy():
            if nctid in path:
                filtered_paths.append(path)
                id_set.discard(nctid)
                break

        if not id_set:
            print("Path extraction early completed. No error occured.")
            break

    return filtered_paths


def xml2criteria(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()

    criteria = root.findtext('.//criteria/textblock')

    return (criteria)


def extract_criteria(target_file="output.csv", batch_size=2**12):

    file_name = f"{OUTPUT_FOLDER}/{target_file.split('.')[0]}_criteria.pkl"
    reset_files([file_name])

    nctids = extract_nctid(f"{OUTPUT_FOLDER}/{target_file}")
    xml_paths = select_files(nctids)

    criteria_dict = {}

    target_path = f"{OUTPUT_FOLDER}/{target_file}"
    original_df = load_csv(target_path)

    for i, path in tqdm(enumerate(xml_paths), desc="Saving criterias", total=len(xml_paths)):
        criteria = xml2criteria(path)

        id = path.split('/')[-1].split('.')[0].strip()

        criteria_dict[id] = criteria

        if (i+1) % batch_size == 0 or (i+1) == len(xml_paths):
            save_dict(criteria_dict, file_name)

            criteria_dict.clear()

    tqdm.pandas(desc="Mapping criterias")
    criteria_dict = load_dict(file_name)
    original_df["criteria"] = original_df["nctid"].progress_apply(
        lambda x: criteria_dict.get(x, None))

    save_csv(target_path, original_df)

    columns = ['age', 'date', 'drug', 'intervention', 'countries', 'states',
               'cities', 'reason', 'gender', 'phase', 'locations', 'condition', 'label', 'criteria']

    nctid_dict = original_df.set_index("nctid")[columns].apply(lambda row: row.values.tolist(), axis=1).to_dict()

    pkl_name = f"{target_file.split('.')[0]}.pkl"
    save_dict(nctid_dict, f"{OUTPUT_FOLDER}/{pkl_name}")

    return None


if __name__ == "__main__":
    extract_criteria()
