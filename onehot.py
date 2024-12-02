from preprocess.file_manage import save_dict, load_dict, save_json, load_json, reset_files, load_csv, splitPkl, clearPklTemp, load_pkls
from config import OUTPUT_FOLDER, JSON_FOLDER, PKL_FOLDER
from multiprocessing import Pool, Lock, Manager
from encode.util import calc_stdv
from tqdm import tqdm
import time

import os
import torch
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
torch.set_num_threads(16)


def intlist2vec(mean, std, val):
    z = (val-mean) / std if std != 0 else 0

    return (
        [1, 0, 0, 0, 0, 0, 0] if val == -1 else
        [0, 1, 0, 0, 0, 0, 0] if z < -2 else
        [0, 0, 1, 0, 0, 0, 0] if z < -1 else
        [0, 0, 0, 1, 0, 0, 0] if z < 0 else
        [0, 0, 0, 0, 1, 0, 0] if z < 1 else
        [0, 0, 0, 0, 0, 1, 0] if z < 2 else
        [0, 0, 0, 0, 0, 0, 1] if 2 <= z else
        [0, 0, 0, 0, 0, 0, 0]
    )


def dur2vec(duration):
    return (
        [1, 0, 0, 0, 0, 0] if duration == -1 else
        [0, 1, 0, 0, 0, 0] if 0 <= duration <= 500 else
        [0, 0, 1, 0, 0, 0] if duration <= 1000 else
        [0, 0, 0, 1, 0, 0] if duration <= 2000 else
        [0, 0, 0, 0, 1, 0] if duration <= 5000 else
        [0, 0, 0, 0, 0, 1] if duration >= 5001 else
        [0, 0, 0, 0, 0, 0]
    )


def age2vec(ages):
    def min_age_bucket(min_age):
        return (
            [1, 0, 0, 0, 0, 0] if min_age == -1 else
            [0, 1, 0, 0, 0, 0] if 0 <= min_age <= 17 else
            [0, 0, 1, 0, 0, 0] if min_age == 18 else
            [0, 0, 0, 1, 0, 0] if 19 <= min_age <= 39 else
            [0, 0, 0, 0, 1, 0] if 40 <= min_age <= 59 else
            [0, 0, 0, 0, 0, 1] if min_age >= 60 else
            [0, 0, 0, 0, 0, 0]
        )

    def max_age_bucket(max_age):
        return (
            [1, 0, 0, 0, 0, 0] if max_age == -1 else
            [0, 1, 0, 0, 0, 0] if 0 <= max_age <= 18 else
            [0, 0, 1, 0, 0, 0] if 19 <= max_age <= 39 else
            [0, 0, 0, 1, 0, 0] if 40 <= max_age <= 59 else
            [0, 0, 0, 0, 1, 0] if 60 <= max_age <= 79 else
            [0, 0, 0, 0, 0, 1] if max_age >= 80 else
            [0, 0, 0, 0, 0, 0]
        )

    def age_span_bucket(age_span):
        return (
            [1, 0, 0, 0, 0, 0] if age_span == -1 else
            [0, 1, 0, 0, 0, 0] if 0 <= age_span <= 19 else
            [0, 0, 1, 0, 0, 0] if 20 <= age_span <= 39 else
            [0, 0, 0, 1, 0, 0] if 40 <= age_span <= 59 else
            [0, 0, 0, 0, 1, 0] if 60 <= age_span <= 79 else
            [0, 0, 0, 0, 0, 1] if age_span >= 80 else
            [0, 0, 0, 0, 0, 0]
        )

    return min_age_bucket(ages[0]), max_age_bucket(ages[1]), age_span_bucket(ages[2])


def match2vec(element, source_file):
    ls = load_json(source_file)
    ls = [s for s in ls if s not in ["N/A"]]

    return [1 if s in element else 0 for s in ls]


lock = None


def init(l):
    global lock
    lock = l


def process_chunk(split_file, index):
    split_dict = load_dict(split_file)

    file_names = [f"cities/city_{index}.pkl", f"states/state_{index}.pkl",
                  f"countries/country_{index}.pkl"]
    # f"diseases/disease_{index}.pkl", f"drugs/drug_{index}.pkl"
    file_paths = [f"{OUTPUT_FOLDER}/bin/{name}" for name in file_names]

    with lock:
        reset_files(file_paths)

    duration_vecs = {}
    min_age_vecs = {}
    max_age_vecs = {}
    age_span_vecs = {}
    gender_vecs = {}
    phase_vecs = {}
    reason_vecs = {}

    # drug_vecs = {}
    # diseases_vecs = {}
    intervensions_vecs = {}

    city_vecs = {}
    state_vecs = {}
    country_vecs = {}

    sta_liat = load_json(f"{OUTPUT_FOLDER}/start_ages.json")
    eda_list = load_json(f"{OUTPUT_FOLDER}/end_ages.json")
    span_list = load_json(f"{OUTPUT_FOLDER}/spans.json")
    dur_list = load_json(f"{OUTPUT_FOLDER}/durations.json")

    sta_stats = calc_stdv(sta_liat)
    eda_stats = calc_stdv(eda_list)
    span_stats = calc_stdv(span_list)
    dur_stats = calc_stdv(dur_list)

    for nctid, data in split_dict.items():
        # duration_vecs[nctid] = dur2vec(int(data[1][2]))
        # min_age_vecs[nctid], max_age_vecs[nctid], age_span_vecs[nctid] = age2vec(
        #     data[0])
        duration_vecs[nctid] = intlist2vec(dur_stats[0], dur_stats[1], data[1][2])
        min_age_vecs[nctid] = intlist2vec(sta_stats[0], sta_stats[1], data[0][0])
        max_age_vecs[nctid] = intlist2vec(eda_stats[0], eda_stats[1], data[0][1])
        age_span_vecs[nctid] = intlist2vec(span_stats[0], span_stats[1], data[0][2])

        gender_vecs[nctid] = match2vec(data[8], f"{JSON_FOLDER}/gender.json")
        phase_vecs[nctid] = match2vec(data[9], f"{JSON_FOLDER}/phase.json")
        reason_vecs[nctid] = match2vec(data[7], f"{JSON_FOLDER}/reason.json")

        intervensions_vecs[nctid] = match2vec(
            data[3], f"{JSON_FOLDER}/intervention.json")
        # drug_vecs[nctid] = match2vec(data[0], f"{JSON_FOLDER}/drugs.json")
        # diseases_vecs[nctid] = match2vec(
        #     data[7], f"{JSON_FOLDER}/condition.json")

        city_vecs[nctid] = match2vec(data[6], f"{JSON_FOLDER}/cities.json")
        state_vecs[nctid] = match2vec(data[5], f"{JSON_FOLDER}/states.json")
        country_vecs[nctid] = match2vec(
            data[4], f"{JSON_FOLDER}/countries.json")

    # No lock needed

    # save_dict(drug_vecs, f"{OUTPUT_FOLDER}/bin/drugs/drug_{index}.pkl", True)

    # drug_vecs.clear()

    # save_dict(diseases_vecs,
    #             f"{OUTPUT_FOLDER}/bin/diseases/condition_{index}.pkl", True)

    # diseases_vecs.clear()

    # Disabled because it is too long, and we have drugbank data to connect
    # TODO!!

    save_dict(city_vecs, f"{OUTPUT_FOLDER}/bin/cities/city_{index}.pkl", True)

    city_vecs.clear()

    save_dict(
        state_vecs, f"{OUTPUT_FOLDER}/bin/states/state_{index}.pkl", True)

    state_vecs.clear()

    save_dict(country_vecs,
              f"{OUTPUT_FOLDER}/bin/countries/country_{index}.pkl", True)

    country_vecs.clear()

    with lock:
        save_dict(duration_vecs, f"{OUTPUT_FOLDER}/bin/duration.pkl", True)
        save_dict(min_age_vecs, f"{OUTPUT_FOLDER}/bin/min_age.pkl", True)
        save_dict(max_age_vecs, f"{OUTPUT_FOLDER}/bin/max_age.pkl", True)
        save_dict(age_span_vecs, f"{OUTPUT_FOLDER}/bin/age_span.pkl", True)
        save_dict(gender_vecs, f"{OUTPUT_FOLDER}/bin/gender.pkl", True)
        save_dict(phase_vecs, f"{OUTPUT_FOLDER}/bin/phase.pkl", True)
        save_dict(reason_vecs, f"{OUTPUT_FOLDER}/bin/reason.pkl", True)
        save_dict(intervensions_vecs,
                  f"{OUTPUT_FOLDER}/bin/intervention_type.pkl", True)

        duration_vecs.clear()
        min_age_vecs.clear()
        max_age_vecs.clear()
        age_span_vecs.clear()
        gender_vecs.clear()
        phase_vecs.clear()
        reason_vecs.clear()
        intervensions_vecs.clear()


def process_with(index_and_path):
    index, path = index_and_path

    return process_chunk(path, index)


def encode_features(target_name="output.csv", batch_size=2**11, max_cpu=12):

    df = load_csv(f"{OUTPUT_FOLDER}/{target_name}")
    start_age_list = [age[0] for age in df["age"]  if age[0] != -1]
    end_age_list = [age[1] for age in df["age"]  if age[1] != -1]
    span_list = [age[2] for age in df["age"] if age[2] != -1]
    duration_list = [date[2] for date in df["date"] if date[2] != -1]

    save_json(start_age_list, f"{OUTPUT_FOLDER}/start_ages.json")
    save_json(end_age_list, f"{OUTPUT_FOLDER}/end_ages.json")
    save_json(span_list, f"{OUTPUT_FOLDER}/spans.json")
    save_json(duration_list, f"{OUTPUT_FOLDER}/durations.json")

    splitPkl(
        f"{OUTPUT_FOLDER}/{target_name.split('.')[0]}.pkl", split_size=batch_size)
    splited_file_names = load_pkls(f"{PKL_FOLDER}/temp")
    splited_paths = [
        f"{PKL_FOLDER}/temp/{name}" for name in splited_file_names]

    file_names = ["gender.pkl", "duration.pkl", "min_age.pkl", "max_age.pkl", "age_span.pkl",
                  "phase.pkl", "reason.pkl", "intervention_type.pkl"]
    file_paths = [f"{OUTPUT_FOLDER}/bin/{name}" for name in file_names]

    reset_files(file_paths)

    indexed_paths = list(enumerate(splited_paths))

    lock = Lock()
    # Mutex

    with Pool(processes=max_cpu, initializer=init, initargs=(lock,)) as pool:
        list(tqdm(pool.imap(process_with, indexed_paths), total=len(indexed_paths)))

    clearPklTemp()
    os.remove(f"{OUTPUT_FOLDER}/start_ages.json")
    os.remove(f"{OUTPUT_FOLDER}/end_ages.json")
    os.remove(f"{OUTPUT_FOLDER}/spans.json")
    os.remove(f"{OUTPUT_FOLDER}/durations.json")


if __name__ == "__main__":

    st = time.time()
    encode_features(max_cpu=16)
    ed = time.time()

    print(f"Total time: {int(ed-st)} sec.")

    # printPkl(f"{OUTPUT_FOLDER}/bin/city.pkl")
    # printPkl(f"{OUTPUT_FOLDER}/bin/state.pkl")
    # printPkl(f"{OUTPUT_FOLDER}/bin/country.pkl")
