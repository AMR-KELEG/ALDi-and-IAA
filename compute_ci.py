import os
from pathlib import Path
from collections import Counter
from datasets import LabelType, load_datasets
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_percentage_of_full_agreement(
    annotations_df, label_column, threshold_number_of_agreeing_annotators=None
):
    """Compute the percentage of samples that have full agreement among the annotators.

    Args:
        annotations_df: A dataframe of annotated samples.
        label_column: The label's column name.
        threshold_number_of_agreeing_annotators: The minimum number of annotators for assuming full agreement.

    Returns:
        The percentage of samples that have full agreement among the annotators.
    """
    if not threshold_number_of_agreeing_annotators:
        # Make sure all the samples have the same number of annotations
        assert (
            len(
                set(
                    [
                        len(annotations)
                        for annotations in annotations_df[label_column].tolist()
                    ]
                )
            )
            == 1
        )

        # Infer the total number of annotators from the dataset
        threshold_number_of_agreeing_annotators = len(
            annotations_df[label_column].iloc[0]
        )

    n_agreeing_on_majority_votes = [
        Counter(annotations).most_common(1)[0][1]
        for annotations in annotations_df[label_column].tolist()
    ]
    return (
        100
        * sum(
            [
                n_agreeing >= threshold_number_of_agreeing_annotators
                for n_agreeing in n_agreeing_on_majority_votes
            ]
        )
        / annotations_df.shape[0]
    )


def compute_percentage_high_confidence(
    annotations_df, label_column, threshold_confidence=0.98
):
    """Compute the percentage of samples that have high annotation confidence.

    Args:
        annotations_df: A dataframe of annotated samples.
        label_column: The label's column name.
        threshold_confidence: The minimum threshold for confidence.
            Confidence is a function of agreement among the annotators, and the accuracy of the annotators's labels on test samples.

    Returns:
        The percentage of samples that have high annotation confidence.
    """
    return 100 * (
        annotations_df[
            annotations_df[label_column].apply(
                lambda t: float(t[-1]) >= threshold_confidence
            )
        ].shape[0]
        / annotations_df.shape[0]
    )


def compute_agreement_score(
    annotations_df,
    label_name,
    label_type,
    threshold_number_of_agreeing_annotators=None,
    threshold_confidence=0.98,
):
    """Compute the agreement score for a given label according to the label type.

    Args:
        annotations_df: A dataframe of annotated samples.
        label_name: The label's column name.
        label_type: The label's type.
        threshold_number_of_agreeing_annotators: The minimum number of annotators for assuming full agreement.
        threshold_confidence: The minimum threshold for confidence.
            Confidence is a function of agreement among the annotators, and the accuracy of the annotators's labels on test samples.

    Returns:
        The agreement score for a given label according to the label type.
    """
    if label_type == LabelType.INDIV:
        return compute_percentage_of_full_agreement(
            annotations_df,
            label_name,
            threshold_number_of_agreeing_annotators=threshold_number_of_agreeing_annotators,
        )
    elif label_type == LabelType.CONF:
        return compute_percentage_high_confidence(
            annotations_df, label_name, threshold_confidence=threshold_confidence
        )
    elif label_type == LabelType.PROPORTION:
        return compute_percentage_high_confidence(
            annotations_df, label_name, threshold_confidence=threshold_confidence
        )


def compute_confidence_interval(df1, df2, label, label_type, N_bootstrap=5000):
    """Compute the confidence interval for the difference in agreement scores between two datasets.

    Args:
        df1:
        df2: _description_
        label: _description_
        label_type: _description_
        N_bootstrap: .

    Returns:
        _description_
    """
    differences = []
    for i in tqdm(range(N_bootstrap)):
        bootstrapped_df1 = df1.sample(df1.shape[0], random_state=i, replace=True)
        bootstrapped_df2 = df2.sample(df2.shape[0], random_state=i, replace=True)

        agreement_1 = compute_agreement_score(
            bootstrapped_df1, label_name=label, label_type=label_type
        )
        agreement_2 = compute_agreement_score(
            bootstrapped_df2, label_name=label, label_type=label_type
        )
        differences.append(agreement_1 - agreement_2)
    return differences


def split_data_by_MSA(df):
    """Split the data into MSA and DA samples.

    Args:
        df: The dataframe of annotated samples.

    Returns:
        A tuple of dataframes, where the first is the MSA samples, and the second is the DA samples.
    """
    MSA_df = df[df["predicted_dialect_26"] == "MSA"]
    DA_df = df[df["predicted_dialect_26"] != "MSA"]
    return (MSA_df, DA_df)


def split_data_by_ALDi(df):
    """Split the data into low and high ALDi samples.

    Args:
        df: The dataframe of annotated samples.

    Returns:
        A tuple of dataframes, where the first is the low ALDi samples, and the second is the high ALDi samples.
    """
    # Discard MSA samples
    DA_df = df[df["predicted_dialect_26"] != "MSA"]

    low_ALDi_df = DA_df[DA_df["ALDi"] < 0.5]
    high_ALDi_df = DA_df[DA_df["ALDi"] >= 0.5]

    return (low_ALDi_df, high_ALDi_df)


if __name__ == "__main__":
    OUTPUT_DIR = "output/ci_plots/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    evaluation_datasets = load_datasets()
    for evaluation_dataset in evaluation_datasets:
        evaluation_df = evaluation_dataset.df
        print(evaluation_dataset.dataset_name)

        MSA_df, DA_df = split_data_by_MSA(evaluation_df)
        low_ALDi_df, high_ALDi_df = split_data_by_ALDi(evaluation_df)

        print(MSA_df.shape[0], DA_df.shape[0])
        print(low_ALDi_df.shape[0], high_ALDi_df.shape[0])

        try:
            for label, label_type in zip(
                evaluation_dataset.labels, evaluation_dataset.labels_types
            ):
                figure = plt.figure()

                agreement_differences = compute_confidence_interval(
                    MSA_df, DA_df, label, label_type, N_bootstrap=100
                )
                plt.hist(
                    agreement_differences, alpha=0.5, label="MSA - DA", color="grey"
                )

                diff_MSA_DA_agreement = compute_agreement_score(
                    MSA_df, label, label_type
                ) - compute_agreement_score(DA_df, label, label_type)

                plt.plot(
                    [diff_MSA_DA_agreement, diff_MSA_DA_agreement],
                    [0, 30],
                    color="grey",
                )

                agreement_differences = compute_confidence_interval(
                    low_ALDi_df, high_ALDi_df, label, label_type, N_bootstrap=100
                )
                plt.hist(
                    agreement_differences,
                    alpha=0.5,
                    label="low ALDi - high ALDi",
                    color="green",
                )
                diff_low_high_ALDi_agreement = compute_agreement_score(
                    low_ALDi_df, label, label_type
                ) - compute_agreement_score(high_ALDi_df, label, label_type)
                plt.plot(
                    [diff_low_high_ALDi_agreement, diff_low_high_ALDi_agreement],
                    [0, 30],
                    color="green",
                )

                plt.legend(title=evaluation_dataset.dataset_name + ": " + label)

                plt.savefig(
                    str(
                        Path(
                            OUTPUT_DIR, f"{evaluation_dataset.dataset_name}_{label}.pdf"
                        )
                    ),
                    bbox_inches="tight",
                )

        except Exception as e:
            print(e)
