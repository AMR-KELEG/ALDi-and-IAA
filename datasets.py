import pandas as pd
from pathlib import Path
from enum import Enum

LabelType = Enum("LabelType", ["CONF", "INDIV", "PERC"])

DATA_DIR = "data/processed/"


class IndividualLabelsDataset:
    def __init__(self, dataset_name, labels, labels_types):
        self.df = pd.read_csv(Path(DATA_DIR, f"{dataset_name}.tsv"), sep="\t")
        self.dataset_name = dataset_name
        self.labels = labels
        self.labels_types = labels_types

        for label, label_type in zip(labels, labels_types):
            # TODO: Apply preprocessing for the other types
            if label_type == LabelType.INDIV:
                self.df[label] = self.df[label].apply(lambda s: s[1:-1].split(", "))


def load_datasets():
    dataset_names, labels_lists, labels_types_lists = [], [], []

    ##### 1) Confidence scores!
    dataset_names.append("ArSAS")
    labels_lists.append(["sentiment", "speech_act"])
    labels_types_lists.append([LabelType.CONF, LabelType.CONF])

    dataset_names.append("DCD")
    labels_lists.append(["offensive"])
    labels_types_lists.append([LabelType.CONF])

    ##### 2) Percentage of agreeing annotators!
    dataset_names.append("DART")
    labels_lists.append(["dialect"])
    labels_types_lists.append([LabelType.PERC])

    ##### 3) Individual annotations
    dataset_names.append("ArSarcasm-v1")
    labels_lists.append(["dialect", "sentiment", "sarcasm"])
    labels_types_lists.append([LabelType.INDIV, LabelType.INDIV, LabelType.INDIV])

    dataset_names.append("iSarcasm_third_party")
    labels_lists.append(["sarcasm"])
    labels_types_lists.append([LabelType.INDIV])

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

    datasets = []
    for dataset_name, labels, labels_types in zip(
        dataset_names, labels_lists, labels_types_lists
    ):
        dataset = IndividualLabelsDataset(
            dataset_name=dataset_name, labels=labels, labels_types=labels_types
        )
        datasets.append(dataset)

    return datasets
