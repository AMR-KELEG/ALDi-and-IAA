set -e

RAW_DATA_DIR="data/raw_data/"
MPOLD_URL="https://github.com/shammur/Arabic-Offensive-Multi-Platform-SocialMedia-Comment-Dataset/raw/master/data/Arabic_offensive_comment_detection_annotation_4000_selected.xlsx"

ALJAZEERA_COMMENTS_URL="http://alt.qcri.org/~hmubarak/offensive/AJCommentsClassification-CF.xlsx"
ARSAS_URL="https://homepages.inf.ed.ac.uk/wmagdy/Resources/ArSAS.zip"

ARABIC_DIALECT_FAMILIARITY_URL="https://raw.githubusercontent.com/iabufarha/arabic-dialect-familiarity/main/dialect_familiarity_sarcasm.csv"

mkdir -p "${RAW_DATA_DIR}"
cd "${RAW_DATA_DIR}"

wget -c "${MPOLD_URL}"

wget -c "${ALJAZEERA_COMMENTS_URL}"
wget -c "${ARSAS_URL}"
unzip "ArSAS.zip"

wget -c "${ARABIC_DIALECT_FAMILIARITY_URL}" -O "arabic_dialect_familiarity.csv"

# All the remaining datasets need to be either downloaded manually or granted by their respective authors!
