
export ENV_NAME="C-data"
export PYTHON_VERSION="3.10"
export PYTORCH_VERSION="2.3"

export BASE_DIR="$(pwd)"
export DATA_FOLDER="$BASE_DIR/data"
export XML_FOLDER="$DATA_FOLDER/xml"
export PKL_FOLDER="$DATA_FOLDER/pkl"
export CSV_FOLDER="$DATA_FOLDER/csv"
export JSON_FOLDER="$DATA_FOLDER/json"
export LEGACY_FOLDER="$CSV_FOLDER/legacy"
export OUTPUT_FOLDER="$BASE_DIR/output"

export FOLDERS_TO_CREATE=(
    "$DATA_FOLDER"
    "$XML_FOLDER"
    "$OUTPUT_FOLDER/bin/cities"
    "$OUTPUT_FOLDER/bin/states"
    "$OUTPUT_FOLDER/bin/countries"
    "$PKL_FOLDER/bin/cities"
    "$PKL_FOLDER/bin/states"
    "$PKL_FOLDER/bin/countries"
    "$PKL_FOLDER/bin/diseases"
    "$PKL_FOLDER/bin/drugs"
    "$JSON_FOLDER"
    "$CSV_FOLDER"
    "$LEGACY_FOLDER"
)

export XML_URL="https://clinicaltrials.gov/AllPublicXML.zip"
export CSV_URL="https://raw.githubusercontent.com/futianfan/clinical-trial-outcome-prediction/refs/heads/main/IQVIA/trial_outcomes_v1.csv"

export LPY_URLS="https://raw.githubusercontent.com/futianfan/clinical-trial-outcome-prediction/refs/heads/main/IQVIA/nctid2label.py"
export L1_URL="https://raw.githubusercontent.com/futianfan/clinical-trial-outcome-prediction/refs/heads/main/IQVIA/outcome2label.txt"
export L2_URL="https://raw.githubusercontent.com/futianfan/clinical-trial-outcome-prediction/refs/heads/main/IQVIA/outcome2label2.txt"

################ User Config Variables ################

export LABEL="$BASE_DIR/label_default.txt"