from config import XML_FOLDER, PKL_FOLDER, JSON_FOLDER, CSV_FOLDER
from preprocess.file_manage import save_dict, save_json, reset_files, load_csv
import os
from datetime import datetime
from lxml import etree
import re
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
torch.set_num_threads(16)


def parse_age(age_text):
    if age_text == '' or age_text == 'N/A':
        return -1
    age = int(age_text.split()[0])
    return age


def xmlfile2loc(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()

    countries = []
    states = []
    cities = []

    for facility in root.xpath(".//location/facility"):
        country = facility.findtext(".//country")
        city = facility.findtext(".//city")
        state = facility.findtext(".//state")
        if country:
            countries.append(country)
        if city:
            # cities.append(city)
            cities.append(re.sub(r'[^\w\s\-\(\).]', '', city).strip())
        if state:
            states.append(state)

    countries = list(set(countries))
    states = list(set(states))
    cities = list(set(cities))

    return countries, states, cities


def xmlfile2age(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()

    min_age_elem = root.find('.//minimum_age')
    max_age_elem = root.find('.//maximum_age')

    min_age = min_age_elem.text if (
        min_age_elem is not None and min_age_elem.text != "N/A") else ''
    max_age = max_age_elem.text if (
        max_age_elem is not None and max_age_elem.text != "N/A") else ''

    return min_age, max_age


def xmlfile2date(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()

    start_date = root.findtext('start_date', default='')
    completion_date = root.findtext('primary_completion_date', default='')
    if not completion_date:
        completion_date = root.findtext('completion_date', default='')

    return start_date, completion_date


def xmlfile2str(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()

    gender = root.findtext('.//gender', default='N/A')
    phase = root.findtext('.//phase', default='N/A')
    locations = [name.text for name in root.findall(
        ".//location/facility/name")]

    condition = root.findtext(".//condition", default='N/A')

    intervention_types = root.findall(".//intervention/intervention_type")
    intervention_names = root.findall(".//intervention/intervention_name")

    intervention_types_list = [
        type.text for type in intervention_types if type is not None]
    intervention_names_list = [
        name.text for name in intervention_names if name is not None]

    intervention = {}

    for type, name in zip(intervention_types_list, intervention_names_list):
        name = re.sub(r'[^a-zA-Z0-9\s\-]', '', name).strip()
        
        if type in intervention:
            intervention[type].append(name)
        else:
            intervention[type] = [name]

    # if len(intervention.items()) == 0:
    #     intervention = None

    return gender, phase, locations, condition, intervention


# def xml2whystop(xml_file):
#     # not used due to simplified, better, clean version of stop reason was found
#     tree = etree.parse(xml_file)
#     root = tree.getroot()

#     return root.findtext('why_stopped', default='').lower().strip()


def csv2whystop(csv_file):
    df = load_csv(csv_file, sep= ',', tolist=False)

    reason_dict = {}
    reasons = []

    for outcome in df['trialOutcome']:
        if "Terminated" in outcome:
            reason = outcome.split(
                ", ", 1)[1] if ", " in outcome else "No reason"
            reasons.append(reason)
        elif "Completed" in outcome:
            reason = outcome.split(
                ", ", 1)[1] if ", " in outcome else "Completed"
            reasons.append(reason)
        else:
            reasons.append("")
    for id, reason in zip(df['studyid'], reasons):
        reason_dict[id] = reason

    return reason_dict


def parse_date(date_str):
    try:
        output = datetime.strptime(date_str, "%B %d, %Y")
    except:
        output = datetime.strptime(date_str, "%B %Y")
    return output


def calculate_duration(start_date, completion_date):
    if start_date and completion_date:
        start_date = parse_date(start_date)
        completion_date = parse_date(completion_date)
        duration = (completion_date - start_date).days
    else:
        duration = -1
    return duration


def get_features(batch_size=2**13):
    age_dict = {}
    str_dict = {}
    date_dict = {}
    location_dict = {}
    whystop_dict = {}
    intervention_dict = {}
    drug_dict = {}

    gender_unique = set()
    phase_unique = set()
    locations_unique = set()
    countries_unique = set()
    states_unique = set()
    cities_unique = set()
    intervention_type_unique = set()
    drug_unique = set()
    diseases_unique = set()

    file_names = ['age_dict.pkl', 'location_dict.pkl',
                  'date_dict.pkl', 'str_dict.pkl', 'stop_reason_dict.pkl', 'intervention_dict.pkl', 'drug_dict.pkl']
    file_paths = [f"{PKL_FOLDER}/{name}" for name in file_names]

    reset_files(file_paths)

    with open(f"{XML_FOLDER}/trials/all_xml.txt", "r", encoding="UTF-8") as file:
        xml_paths = [line.strip() for line in file]

    reason_dict = csv2whystop(f"{CSV_FOLDER}/IQVIA_trial_outcomes.csv")
    reason_unique = set(reason_dict.values())

    for i, xml_path in tqdm(enumerate(xml_paths), total=len(xml_paths)):
        nct_id = re.search(r"/([^/]+)\.xml$", xml_path).group(1)

        min_age, max_age = xmlfile2age(xml_path)
        countries, states, cities = xmlfile2loc(xml_path)
        start_date, completion_date = xmlfile2date(xml_path)
        gender, phase, locations, condition, intervention = xmlfile2str(
            xml_path)
        whystop = reason_dict[nct_id] if nct_id in reason_dict else "N/A"

        gender_unique.add(gender)
        phase_unique.add(phase)
        locations_unique.update(locations)
        countries_unique.update(countries)
        states_unique.update(states)
        cities_unique.update(cities)
        intervention_type_unique.update(intervention.keys())
        diseases_unique.add(condition)

        intervention_dict[nct_id] = list(set(intervention.keys()))

        if "Drug" in intervention.keys():
            drug_dict[nct_id] = intervention["Drug"]
            drug_unique.update(intervention["Drug"])
        else:
            drug_dict[nct_id] = ["N/A"]

        str_dict[nct_id] = {
            "gender": [gender],
            "phase": [phase],
            "locations": locations,
            "condition": [condition],
        }

        location_dict[nct_id] = {
            "countries": countries,
            "states": states,
            "cities": cities
        }

        duration = calculate_duration(
            start_date, completion_date) if start_date and completion_date else -1
        date_dict[nct_id] = [start_date, completion_date, int(duration)]

        min_age = parse_age(min_age)
        max_age = parse_age(max_age)
        age_span = (max_age - min_age) if (min_age != -
                                           1 and max_age != -1) else -1

        age_dict[nct_id] = [min_age, max_age, age_span]
        whystop_dict[nct_id] = [whystop]

        if (i + 1) % batch_size == 0 or (i + 1) == len(xml_paths):
            save_dict(age_dict, f'{PKL_FOLDER}/age_dict.pkl', append=True)
            save_dict(location_dict,
                      f'{PKL_FOLDER}/location_dict.pkl', append=True)
            save_dict(date_dict, f'{PKL_FOLDER}/date_dict.pkl', append=True)
            save_dict(str_dict, f'{PKL_FOLDER}/str_dict.pkl', append=True)
            save_dict(whystop_dict,
                      f'{PKL_FOLDER}/stop_reason_dict.pkl', append=True)
            save_dict(intervention_dict, f'{PKL_FOLDER}/intervention_dict.pkl', append=True)
            save_dict(drug_dict, f'{PKL_FOLDER}/drug_dict.pkl', append=True)

            age_dict.clear()
            str_dict.clear()
            date_dict.clear()
            location_dict.clear()
            whystop_dict.clear()
            intervention_dict.clear()
            drug_dict.clear()

            break
            # for smaller test calc

    save_json(sorted(gender_unique), f'{JSON_FOLDER}/gender.json')
    save_json(sorted(phase_unique), f'{JSON_FOLDER}/phase.json')
    save_json(sorted(locations_unique), f'{JSON_FOLDER}/locations.json')
    save_json(sorted(countries_unique), f'{JSON_FOLDER}/countries.json')
    save_json(sorted(states_unique), f'{JSON_FOLDER}/states.json')
    save_json(sorted(cities_unique), f'{JSON_FOLDER}/cities.json')
    save_json(sorted(reason_unique), f'{JSON_FOLDER}/reason.json')
    save_json(sorted(intervention_type_unique), f'{JSON_FOLDER}/intervention_types.json')
    save_json(sorted(drug_unique), f'{JSON_FOLDER}/drugs.json')
    save_json(sorted(diseases_unique), f'{JSON_FOLDER}/diseases.json')

    return None


if __name__ == "__main__":

    get_features()
