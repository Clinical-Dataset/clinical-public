# ClinicalTrialDataset

## Setup

- Run the setup file
  - ```./env_setup.sh```
  - If permission denined, do:
    - ```chmod +x env_setup.sh```
    - ```chmod +x config.sh```

## Data management

### Download the data

- Run the downlad file
  - ```./download_data.sh```
  - 'y' if message appears.

### Processing the data

- Activate conda

- ```python -m process -h```
  - It will show what available options are present.

- ```python -m process```
  - It will produce feature extraction file under data/csv/full_features.csv.

- ```python -m process -e```
  - It will create one-hot encoded files under pkl/bin folder.
  - Takes time to produce.

- ```python -m process -t <output_name.csv>```
  - It will produce csv file named as defined.
  - If no name is input, then it will create "output.csv" by default.