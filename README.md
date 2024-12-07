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

#### Running the full code

- For run all the code below at once:
  - ```python -m process -f <output_name.csv>```

#### Running only the specific case

- ```python -m process -h```
  - It will show what available options are present.

- ```python -m process```
  - It will produce feature extraction file under data/csv/full_features.csv.

- ```python -m process -l <output_name.csv>```
  - It will produce csv file named as defined.
  - Label is saved under label_defualt.txt
    - You can change this directly, or to other file using config.sh
  - If no name is input, then it will create "output.csv" by default.

- ```python -m process -e <output_name.csv>```
  - It will create one-hot encoded files under output/bin folder.
  - Then it creates sentence embedding files from the criteria.
  - Takes time to produce.
