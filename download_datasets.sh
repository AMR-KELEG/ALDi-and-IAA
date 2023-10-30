set -e

RAW_DATA_DIR="data/raw_data/"
MPOLD_URL="https://github.com/shammur/Arabic-Offensive-Multi-Platform-SocialMedia-Comment-Dataset/raw/master/data/Arabic_offensive_comment_detection_annotation_4000_selected.xlsx"

# TODO: Fix this!
YOUTUBE_CYBERBULLYING_URL="https://public.by.files.1drv.com/y4mJqvi-IXqf2Ma9DqHaqtImXcRFhp5HHvP386PgaWnoKsq1HZjnMXLS3POrJC9IizXeqNKlKfm1Jg45SsrCoWQOX1lkPemQNtjQF27pDYmma83iWfDpXYc-Ex8vpsNkyedzkf90HCn_tfuCCKYR_zI03rbnY8JUTdkALx6s0CkBOGJmyQysvB6HwysIljeSymPvSx07D1FutorNXw6BVzkEukMJtbeZnrQSwOrMX4DBes"
YOUTUBE_CYBERBULLYING_ZIP="YouTube_cyberbullying.zip"
# wget -c "${YOUTUBE_CYBERBULLYING_URL}" -O "${YOUTUBE_CYBERBULLYING_ZIP}"
# unzip "${YOUTUBE_CYBERBULLYING_ZIP}"
# mv "LabeledDataset.xlsx" "${YOUTUBE_CYBERBULLYING_FILE}"

YOUTUBE_CYBERBULLYING_FILE="YouTube_cyberbullying.xlsx"
ALJAZEERA_COMMENTS_URL="http://alt.qcri.org/~hmubarak/offensive/AJCommentsClassification-CF.xlsx"
ARSAS_URL="https://homepages.inf.ed.ac.uk/wmagdy/Resources/ArSAS.zip"
ISARCASM_URL="https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/third-party%20annotations/arabic_task_a.csv"

# TODO: Fix this!
DART_URL="https://ucba8efb7f65c5060d175400df00.dl.dropboxusercontent.com/cd/0/get/CGYTSJtLFncDf9srywe2VxGoZpx34i8ldPRd7e_kwp1eJd7wjaDfVZtB1kW5Nd1V9Y3bfcIJQhi2vE-XanaE7yFtDY1uHnbq_FQUV8BdUVH9zOXj_GYtiAi2XBrXoykQeQ-UVqCKibM4KlNBec6ZRUIv/file?_download_id=9413793138464943757929744010247891295413869371612546038916206848&_notify_domain=www.dropbox.com&dl=1"
### DRAFT ###
# MAWQIF_TRAIN_URL="https://raw.githubusercontent.com/NoraAlt/Mawqif-Arabic-Stance/main/Data/Mawqif_AllTargets_Train.csv"
# MAWQIF_TEST_URL="https://raw.githubusercontent.com/NoraAlt/Mawqif-Arabic-Stance/main/Data/Mawqif_AllTargets_Test.csv"

mkdir -p "${RAW_DATA_DIR}"
cd "${RAW_DATA_DIR}"

wget -c "${MPOLD_URL}"

wget -c "${ALJAZEERA_COMMENTS_URL}"
wget -c "${ARSAS_URL}"
unzip "ArSAS.zip"

wget -c "${ISARCASM_URL}"
mv arabic_task_a.csv "iSarcasm_third_party.csv"

### DRAFT ###
# wget -c "${MAWQIF_TRAIN_URL}"
# wget -c "${MAWQIF_TEST_URL}"

# cat Mawqif_AllTargets_*.csv > Mawqif_AllTargets.csv
# rm Mawqif_AllTargets_*.csv
