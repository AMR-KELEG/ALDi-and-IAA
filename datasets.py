import os
import pandas as pd
from pathlib import Path
from enum import Enum
from collections import Counter

LabelType = Enum("LabelType", ["CONF", "INDIV", "PROPORTION"])

DATA_DIR = "data/processed/"


def get_majority_vote(individual_annotations):
    label, count = Counter(individual_annotations).most_common(1)[0]
    return (label, count / len(individual_annotations))


class IndividualLabelsDataset:
    def __init__(self, dataset_name, labels, labels_types):
        self.df = pd.read_csv(Path(DATA_DIR, f"{dataset_name}.tsv"), sep="\t")
        self.dataset_name = dataset_name
        self.labels = labels
        self.labels_types = labels_types

        self.df.loc[self.df["ALDi"] < 0, "ALDi"] = 0
        self.df.loc[self.df["ALDi"] > 1, "ALDi"] = 1

        for label, label_type in zip(labels, labels_types):
            # TODO: Apply preprocessing for the other types
            if label_type == LabelType.INDIV:
                self.df[label] = self.df[label].apply(lambda s: s[1:-1].split(", "))
                self.df[f"{label}_majority_vote"] = self.df[label].apply(
                    lambda indiv_labels: get_majority_vote(indiv_labels)
                )
            elif label_type == LabelType.CONF or label_type == LabelType.PROPORTION:
                self.df[label] = self.df[label].apply(lambda s: s[1:-1].split(","))
                # Parse the confidence score (the last value in the tuple)
                self.df[label] = self.df[label].apply(lambda t: t[:-1] + [float(t[-1])])

            n_labels_per_sample = self.df[label].apply(lambda l: len(l)).tolist()
            try:
                assert len(set(n_labels_per_sample)) == 1
            except:
                print(
                    dataset_name,
                    "Different no of labels per sample!",
                    ",".join([str(n_labels) for n_labels in set(n_labels_per_sample)]),
                )

                # TODO: Refactor this!
                print(Counter(n_labels_per_sample).most_common())
                print("Discarding samples with no. of labels != 3!")
                self.df = self.df[
                    self.df.apply(lambda row: len(row[label]) == 3, axis=1)
                ]

    def export(self, output_dir):
        self.df.to_csv(
            str(Path(output_dir, f"{self.dataset_name}.tsv")), sep="\t", index=False
        )


def load_datasets():
    dataset_names, labels_lists, labels_types_lists = [], [], []

    ##### 1) Confidence scores!
    dataset_names.append("ArSAS")
    labels_lists.append(["sentiment", "speech_act"])
    labels_types_lists.append([LabelType.CONF, LabelType.CONF])

    dataset_names.append("DCD")
    labels_lists.append(["offensive"])
    labels_types_lists.append([LabelType.CONF])

    ##### 2) Proportion of agreeing annotators!
    dataset_names.append("DART")
    labels_lists.append(["dialect"])
    labels_types_lists.append([LabelType.PROPORTION])

    ##### 3) Individual annotations
    dataset_names.append("ArSarcasm-v1")
    labels_lists.append(["dialect", "sentiment", "sarcasm"])
    labels_types_lists.append([LabelType.INDIV, LabelType.INDIV, LabelType.INDIV])

    dataset_names.append("MPOLD")
    labels_lists.append(["offensive"])
    labels_types_lists.append([LabelType.INDIV])

    dataset_names.append("YouTube_cyberbullying")
    labels_lists.append(["hate_speech"])
    labels_types_lists.append([LabelType.INDIV])

    dataset_names.append("Mawqif_stance")
    labels_lists.append(["stance"])
    labels_types_lists.append([LabelType.INDIV])

    dataset_names.append("Mawqif_sarcasm")
    labels_lists.append(["sarcasm", "sentiment"])
    labels_types_lists.append([LabelType.INDIV, LabelType.INDIV])

    dataset_names.append("arabic_dialect_familiarity")
    labels_lists.append(["dialect", "sarcasm"])
    labels_types_lists.append([LabelType.INDIV, LabelType.INDIV])

    dataset_names.append("LetMI")
    labels_lists.append(["misogyny_general", "misogyny_specific"])
    labels_types_lists.append([LabelType.INDIV, LabelType.INDIV])

    dataset_names.append("qweet")
    labels_lists.append(["qweet"])
    labels_types_lists.append([LabelType.CONF])

    datasets = []
    for dataset_name, labels, labels_types in zip(
        dataset_names, labels_lists, labels_types_lists
    ):
        dataset = IndividualLabelsDataset(
            dataset_name=dataset_name, labels=labels, labels_types=labels_types
        )
        datasets.append(dataset)

    return datasets
