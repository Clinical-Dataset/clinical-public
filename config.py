"""
Source and Path management file.
"""

import subprocess
import os


def load_config(file_path):
    # Source the config.sh file and export the variables
    command = f"bash -c 'source {file_path} && env'"
    result = subprocess.run(command, shell=True,
                            capture_output=True, text=True, check=True)

    config_vars = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        config_vars[key] = value

    return config_vars


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SH = "config.sh"
config = load_config(SH)

DATA_FOLDER = config.get("DATA_FOLDER")
XML_FOLDER = config.get("XML_FOLDER")
PKL_FOLDER = config.get("PKL_FOLDER")
JSON_FOLDER = config.get("JSON_FOLDER")
CSV_FOLDER = config.get("CSV_FOLDER")
OUTPUT_FOLDER = config.get("OUTPUT_FOLDER")

LABEL_FILE = config.get("LABEL")

columns = ['nctid', 'age', 'date', 'drug', 'intervention', 'countries', 'states',
           'cities', 'reason', 'gender', 'phase', 'locations', 'condition']  # manually set
