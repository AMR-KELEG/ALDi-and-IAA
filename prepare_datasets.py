import os
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, BertForSequenceClassification
from tqdm import tqdm

tqdm.pandas()
from camel_tools.dialectid import DIDModel6, DIDModel26

# Load the model
MODEL_NAME = "AMR-KELEG/Sentence-ALDi"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

# Load the DI model
di_model6 = DIDModel6.pretrained()
di_model26 = DIDModel26.pretrained()

TEXT_COLUMN = "text"
ALDi_COLUMN = "ALDi"
DIALECT6_COLUMN = "predicted_dialect_6"
DIALECT26_COLUMN = "predicted_dialect_26"
COMMON_COLUMNS = [TEXT_COLUMN, ALDi_COLUMN, DIALECT6_COLUMN, DIALECT26_COLUMN]


# Compute the score
def compute_score(sentence):
    features = tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True)
    o = model(**features).logits[0].tolist()[0]
    return o


def augment_with_labels(df):
    """Add analysis columns to the dataframe.
    Note: the dataframe should have a column TEXT_COLUMN

    Args:
        df: A df augmented with new analsis columns
    """

    assert TEXT_COLUMN in df.columns

    # Compute ALDi
    df[ALDi_COLUMN] = df[TEXT_COLUMN].progress_apply(compute_score)

    # Augment with predicted dialect
    df[DIALECT6_COLUMN] = df[TEXT_COLUMN].progress_apply(
        lambda t: di_model6.predict([t])[0].top
    )
    df[DIALECT26_COLUMN] = df[TEXT_COLUMN].progress_apply(
        lambda t: di_model26.predict([t])[0].top
    )
    return df


if __name__ == "__main__":
    OUTPUT_DIR = "data/processed"
    os.makedirs(OUTPUT_DIR)
    ############ DATASET ############
    # YouTube Cyberbullying
    LABEL = "hate_speech"
    df = pd.read_excel("data/raw_data/YouTube_cyberbullying.xlsx")

    # Fix the inconsistent label values used
    individual_label_columns = ["annotator1", "annotator2", "annotator3", "Label"]
    labels_map = {"n": "Not", "p": "HateSpeech"}
    for col in individual_label_columns:
        df[col] = df[col].apply(lambda l: labels_map[l.strip().lower()])

    # Make sure each row is either a comment or a reply to a comment
    df["has_text"] = df.apply(
        lambda row: str(row["commentText"]) != "nan"
        or str(row["replies.commentText"]) != "nan",
        axis=1,
    )
    assert set(df["has_text"].tolist()) == {True}

    # Extract the comment or reply into a TEXT_COLUMN field
    df[TEXT_COLUMN] = df.apply(
        lambda row: row["commentText"]
        if str(row["commentText"]) != "nan"
        else row["replies.commentText"],
        axis=1,
    )

    # Form the aggregated label column
    df[LABEL] = df.apply(
        lambda row: [row[f"annotator{i}"] for i in range(1, 4)], axis=1
    )
    augment_with_labels(df)
    df[COMMON_COLUMNS + [LABEL]].to_csv(
        str(Path(OUTPUT_DIR, "YouTube_cyberbullying.tsv")), sep="\t", index=False
    )

    ############ DATASET ############
    # MPOLD
    LABEL = "offensive"
    df = pd.read_excel(
        "data/raw_data/Arabic_offensive_comment_detection_annotation_4000_selected.xlsx"
    )

    df[TEXT_COLUMN] = df["Comment"]
    df[LABEL] = df.apply(
        lambda row: [row["Majority_Label"]] * 3
        if int(row["Agreement"]) == 100
        else ["Offensive", "No-Offensive", row["Majority_Label"]],
        axis=1,
    )
    augment_with_labels(df)

    df[COMMON_COLUMNS + [LABEL, "Platform"]].to_csv(
        str(Path(OUTPUT_DIR, "MPOLD.tsv")), sep="\t", index=False
    )

    ############ DATASET ############
    # DCD
    LABEL = "offensive"
    labels_dictionary = {-2: "Obscene", -1: "Offensive", 0: "Clean"}
    df = pd.read_excel("data/raw_data/AJCommentsClassification-CF.xlsx").head(n=100)

    # The "Title" is shown during annotation
    df["Title"] = df["articletitle"]

    df[TEXT_COLUMN] = df["body"]
    df[LABEL] = df.apply(
        lambda row: (
            labels_dictionary[row["languagecomment"]],
            row["languagecomment:confidence"],
        ),
        axis=1,
    )
    augment_with_labels(df)

    df[COMMON_COLUMNS + [LABEL, "Title"]].to_csv(
        str(Path(OUTPUT_DIR, "DCD.tsv")), sep="\t", index=False
    )

    ############ DATASET ############
    # ArSAS
    LABEL1 = "sentiment"
    LABEL2 = "speech_act"
    df = pd.read_csv("data/raw_data/ArSAS..txt", sep="\t")

    # Map the columns with confidence scores
    # NOTE: Confidence scores are nearly categorical (1/3, 2/3, 3/3)
    df[LABEL1] = df.apply(
        lambda row: (row["Sentiment_label"], row["Sentiment_label_confidence"]), axis=1
    )
    df[LABEL2] = df.apply(
        lambda row: (row["Speech_act_label"], row["Speech_act_label_confidence"]),
        axis=1,
    )
    df[TEXT_COLUMN] = df["Tweet_text"]
    augment_with_labels(df)

    df[COMMON_COLUMNS + [LABEL1, LABEL2, "Topic"]].to_csv(
        str(Path(OUTPUT_DIR, "ArSAS.tsv")), sep="\t", index=False
    )

    ############ DATASET ############
    # ArSarcasm-v1
    LABEL1 = "dialect"
    LABEL2 = "sentiment"
    LABEL3 = "sarcasm"

    df = pd.read_csv("data/raw_data/ArSarcasm-v1.csv")
    for label in [LABEL1, LABEL2, LABEL3]:
        df[label] = df[label].apply(lambda l: [a.strip() for a in l[1:-1].split(",")])
    df[TEXT_COLUMN] = df["tweet"]
    augment_with_labels(df)

    df[COMMON_COLUMNS + [LABEL1, LABEL2, LABEL3]].to_csv(
        str(Path(OUTPUT_DIR, "ArSarcasm-v1.tsv")), sep="\t", index=False
    )

    ############ DATASET ############
    # iSaracasm (third party)
    LABEL = "sarcasm"
    df = pd.read_csv("data/raw_data/iSarcasm_third_party.csv")
    df[LABEL] = df.apply(
        lambda row: row["#humans_sarcasm"] * ["Not Sarcasm"]
        + (5 - row["#humans_sarcasm"]) * ["Sarcasm"],
        axis=1,
    )
    df["original_dialect"] = df["dialect"]

    augment_with_labels(df)

    df[COMMON_COLUMNS + [LABEL, "original_dialect"]].to_csv(
        str(Path(OUTPUT_DIR, "iSarcasm_third_party.tsv")), sep="\t", index=False
    )

    ############ DATASET ############
    # DART
    # TODO: Start from raw files
    df = pd.read_csv("data/raw_data/DART_with_ALDi.tsv", sep="\t")
    LABEL = "dialect"
    df[LABEL] = df.apply(lambda row: (row["dialect"], row["score(/3)"] / 3), axis=1)
    df[TEXT_COLUMN] = df["tweet_text"]
    augment_with_labels(df)

    df[COMMON_COLUMNS + [LABEL]].to_csv(
        str(Path(OUTPUT_DIR, "DART.tsv")), sep="\t", index=False
    )
