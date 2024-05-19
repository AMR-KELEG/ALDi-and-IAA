# ALDi-and-IAA

The codebase accompanying the `Estimating the Level of Dialectness Predicts Interannotator Agreement in Multi-dialectal Arabic Datasets` paper, accepted to [ACL 2024](https://2024.aclweb.org/).

## Environment and Dependencies
```
conda create -n "ALDI_IAA" python=3.10
pip install -r requirements.txt

camel_data -i defaults
```

## Datasets
|| Dataset  | Link  |
|---|---|---|
1| MPOLD | [GitHub](https://github.com/shammur/Arabic-Offensive-Multi-Platform-SocialMedia-Comment-Dataset/raw/master/data/Arabic_offensive_comment_detection_annotation_4000_selected.xlsx)  |
2| YouTube Cyberbullying  | [OneDrive](https://onedrive.live.com/?authkey=%21ACDXj%5FZNcZPqzy0&cid=6EF6951FBF8217F9&id=6EF6951FBF8217F9%21110&parId=6EF6951FBF8217F9%21105&o=OneUp) |
3| DCD | [Personal Site](http://alt.qcri.org/~hmubarak/offensive/AJCommentsClassification-CF.xlsx) |
4| ArSAS| [Personal Site](https://homepages.inf.ed.ac.uk/wmagdy/Resources/ArSAS.zip) |
5| ArSarcasm-v1 | Provided by the authors |
6| iSarcasm | [GitHub](https://raw.githubusercontent.com/iabufarha/arabic-dialect-familiarity/main/dialect_familiarity_sarcasm.csv)|
7| DART | [Dropbox](https://www.dropbox.com/s/jslg6fzxeu47flu/DART.zip?dl=0) |
8| Mawqif | Provided by the authors |
9| ASAD | Provided by the authors |

## Generating the ALDi-IAA Plots
```
conda activate ALDI_IAA

# 1) MANUALLY Download the dataset files to `data/raw_data/`

# 2) Augment the dataset files with ALDi scores, and dialect labels
python prepare_datasets.py

# 3) Generate the Agreement plots
python compute_agreement_percentages.py
```
