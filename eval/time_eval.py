from tqdm import tqdm 
from xml.etree import ElementTree as ET
from datetime import datetime
import re
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np


def save_dict(file, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(file, f)

def load_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def parse_date(date_str):
    try:
        output = datetime.strptime(date_str, "%B %d, %Y")
    except:
        try:
            output = datetime.strptime(date_str, "%B %Y")
        except Exception as e:
            print(e)
            raise e
    return output

def calculate_duration(start_date, completion_date):
    # Unit: days
    if start_date and completion_date:
        start_date = parse_date(start_date)
        completion_date = parse_date(completion_date)
        duration = (completion_date - start_date).days
    else:
        duration = -1

    return duration

def xmlfile2date(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    try:
        start_date = root.find('start_date').text
    except:
        start_date = ''
    try:
        completion_date = root.find('primary_completion_date').text
    except:
        try:
            completion_date = root.find('completion_date').text 
        except:
            completion_date = ''

    return start_date, completion_date


def get_time():
    
    date_dict = {}

    # 478504 lines
    with open("data/trials/all_xml.txt", "r") as file:
        for xml_path in tqdm(file):
            xml_path = f"data/{xml_path.strip()}"

            # NCT00000150 <- raw_data/NCT0000xxxx/NCT00000150.xml
            nct_id = re.search(r"/([^/]+)\.xml$", xml_path).group(1)
            
            start_date, completion_date = xmlfile2date(xml_path)

            #print(start_date, completion_date)

            if start_date and completion_date:
                duration = calculate_duration(start_date, completion_date)
            else:
                duration = -1

            date_dict[nct_id] = [start_date, completion_date, duration]
    
    print(date_dict['NCT00000102'])
    print(len(date_dict))
    

    
    return date_dict



if __name__ == "__main__":
    
    date_dict = load_dict("data/date_dict.pkl")
    

    data_greater_than_10_years = {key: [value[0], value[1], value[2]/ 365] for key, value in date_dict.items() if value[2] > 10 * 365}
    
    print(f"all: {len(date_dict)}, >10 years: {len(data_greater_than_10_years)}, ratio: {1 - len(data_greater_than_10_years) / len(date_dict)}")

    # Number of entries with duration more than 10 years
    num_greater_than_10_years = len(data_greater_than_10_years)

    #print(data_greater_than_10_years)
    print(f"Number of entries with duration more than 10 years: {num_greater_than_10_years}")

    durations = [value[2] for value in data_greater_than_10_years.values()]

    # Calculate the statistics
    average = np.mean(durations)
    minimum = np.min(durations)
    first_quartile = np.percentile(durations, 25)
    median = np.median(durations)
    third_quartile = np.percentile(durations, 75)
    maximum = np.max(durations)

    print(f"Average: {average}")
    print(f"Minimum: {minimum}")
    print(f"1st Quartile: {first_quartile}")
    print(f"Median: {median}")
    print(f"3rd Quartile: {third_quartile}")
    print(f"Maximum: {maximum}")

    # date_dict = load_dict('data/date_dict.pkl')
    # print(date_dict)
    
    # Plot the histogram

    plt.hist(durations, bins=50, edgecolor='black')
    plt.title('Histogram of Durations')
    plt.xlabel('Days')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig("duration__distribution_>10years_unit=year.png")