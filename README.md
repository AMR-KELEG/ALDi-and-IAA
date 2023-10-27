# ALDi-and-annotator-agreement

## Environment
```
conda create -n "ALDI_IAA" python=3.10
pip install -r requirements.txt

camel_data -i defaults
```

## Datasets
| Dataset  | Link  |
|---|---|
| MPOLD | [GitHub](https://github.com/shammur/Arabic-Offensive-Multi-Platform-SocialMedia-Comment-Dataset/raw/master/data/Arabic_offensive_comment_detection_annotation_4000_selected.xlsx)  |
| YouTube Cyberbullying  | [OneDrive](https://onedrive.live.com/?authkey=%21ACDXj%5FZNcZPqzy0&cid=6EF6951FBF8217F9&id=6EF6951FBF8217F9%21110&parId=6EF6951FBF8217F9%21105&o=OneUp) |
| DCD | [Personal Site](http://alt.qcri.org/~hmubarak/offensive/AJCommentsClassification-CF.xlsx) |
| ArSAS| [Personal Site](https://homepages.inf.ed.ac.uk/wmagdy/Resources/ArSAS.zip) |
| ArSarcasm-v1 | Provided by the authors |
| iSarcasm (third-party annotations) | [GitHub](https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/third-party%20annotations/arabic_task_a.csv)| 
| DART | [Personal Site](https://www.dropbox.com/s/jslg6fzxeu47flu/DART.zip?dl=0) | 

## Running the experiments
```
# Download the dataset files
# NOTE: YouTube Cyberbullying and DART need to be manually downloaded!
./download_datasets.sh

```