#!/bin/bash

# Define variables
source "./config.sh"

for folder in "${FOLDERS_TO_CREATE[@]}"; do
    mkdir -p "$folder"
done

wget "$XML_URL" -O "$XML_FOLDER/AllPublicXML.zip"

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "File downloaded successfully and saved to $XML_FOLDER/AllPublicXML.zip"
    
    unzip "$XML_FOLDER/AllPublicXML.zip" -d "$XML_FOLDER/trials"
    find "$XML_FOLDER/trials" -name "NCT*.xml" | sort > "$XML_FOLDER/trials/all_xml.txt"

    echo "XML files have been extracted and listed in $XML_FOLDER/trials/all_xml.txt"

    wget "$CSV_URL" -O "$CSV_FOLDER/IQVIA_trial_outcomes.csv"

    if [ $? -eq 0 ]; then
        echo "CSV file downloaded successfully and saved to $CSV_FOLDER/IQVIA_trial_outcomes.csv"
    else
        echo "Failed to download the CSV file."
    fi

    wget "$LPY_URL" -O "$LEGACY_FOLDER/nctid2label.py"
    wget "$L1_URL" -O "$LEGACY_FOLDER/outcome2label.txt"
    wget "$L2_URL" -O "$LEGACY_FOLDER/outcome2label2.txt"

    if [ $? -eq 0 ]; then
        echo "legacy labels downloaded to $LEGACY_FOLDER"
    else
        echo "Failed to download the legacy files."
    fi

else
    echo "Failed to download file. Please refer to README: Issues."
fi
