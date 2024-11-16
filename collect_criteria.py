
from config import XML_FOLDER, OUTPUT_FOLDER
from preprocess.file_manage import load_csv, reset_files, save_dict
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

    for i, path in tqdm(enumerate(xml_paths), desc="Saving criterias", total=len(xml_paths)):
        criteria = xml2criteria(path)

        id = path.split('/')[-1].split('.')[0].strip()

        criteria_dict[id] = criteria

        if (i+1) % batch_size == 0 or (i+1) == len(xml_paths):
            save_dict(criteria_dict, file_name)

            criteria_dict.clear()

    return None


if __name__ == "__main__":
    extract_criteria("failure_reasoning.csv")
