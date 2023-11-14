from collections import Counter
from datasets import LabelType, load_datasets
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_percentage_of_full_agreement(
    annotations_df, label_column, threshold_number_of_agreeing_annotators=None
):
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
    MSA_df = df[df["predicted_dialect_26"] == "MSA"]
    DA_df = df[df["predicted_dialect_26"] != "MSA"]
    return (MSA_df, DA_df)


def split_data_by_ALDi(df):
    DA_df = df[df["predicted_dialect_26"] != "MSA"]
    low_ALDi_df = DA_df[DA_df["ALDi"] < 0.5]
    high_ALDi_df = DA_df[DA_df["ALDi"] >= 0.5]

    return (low_ALDi_df, high_ALDi_df)


if __name__ == "__main__":
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
        except Exception as e:
            print(e)
