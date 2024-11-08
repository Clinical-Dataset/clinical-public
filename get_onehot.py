from preprocess.file_manage import save_dict, load_dict, load_pkls, load_csv
from config import PKL_FOLDER, OUTPUT_FOLDER
from tqdm import tqdm

def get_bin_from_files(nctid, folder):
    # cities, countries, states
    pkl_names = load_pkls(folder)
    pkl_paths = [f"{folder}/{name}" for name in pkl_names]

    for path in pkl_paths:
        dt = load_dict(path)
        
        if nctid in dt:
            return dt[nctid]
        
    return None

def get_bin_from_file(nctid, file):
    # age, drug, duration, gender, intervention type......
    dt = load_dict(file)

    if nctid in dt:
        return dt[nctid]
    
    return None

def get_binary(file_name, split_size = 2**12):

    full_path = f"{OUTPUT_FOLDER}/{file_name}"
    df = load_csv(full_path)

    ids = df['nctid'].tolist()

    files = load_pkls(f"{PKL_FOLDER}/bin")
    file_paths = [f"{PKL_FOLDER}/bin/{name}" for name in files]
    # always sorted order
    # age span / drug / duration / gender / intervention type / max age / min age / phase / reason

    dirs = ["cities","countries","states"]
    file_dirs = [f"{PKL_FOLDER}/bin/{name}" for name in dirs]
      
    print(file_paths)


if __name__ == "__main__":
    get_binary("failure_reasoning.csv")