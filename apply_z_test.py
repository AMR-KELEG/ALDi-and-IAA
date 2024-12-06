import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels
from statsmodels.stats.weightstats import DescrStatsW


def compute_proportion_full_agreement(df, label):
    """Computes the proportion of samples with full agreement for a given label.

    Args:
        df: The dataframe containing the samples.
        label: The label for which the proportion of full agreement is computed.

    Returns:
        The proportion of samples with full agreement for the given label.
    """
    return df[label].apply(lambda l: len(set(l)) == 1).sum() / df.shape[0]


def generate_bootstrap_samples(df, n_samples=1000, sample_size=100):
    """Generates bootstrap samples from a given dataframe."""
    samples = []
    for i in range(n_samples):
        # Form a bootstrap sample of `n_samples` number of annotated examples.
        samples.append(df.sample(n=sample_size, random_state=i, replace=True))
    return samples


def compute_two_sample_ztest(u1, u2, s1, s2, n1, n2):
    """Computes the two-sample z-test for the difference of means.

    Args:
        u1: Mean of the first sample.
        u2: Mean of the second sample.
        s1: Standard deviation of the first sample.
        s2: Standard deviation of the second sample.
        n1: Size of the first sample.
        n2: Size of the second sample.

    Returns:
        The z-score of the two-sample z-test.
    """
    return (u1 - u2) / np.sqrt(s1**2 / n1 + s2**2 / n2)


def main(dataset):

    df = pd.read_csv(dataset, sep="\t")
    if "sarcasm" in df.columns:
        df["sarcasm"] = df["sarcasm"].apply(
            lambda l: [i[1:-1] for i in l[1:-1].split(", ")]
        )
        df = df[df["sarcasm"].apply(lambda l: len(l) == 3)]
    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].apply(
            lambda l: [i[1:-1] for i in l[1:-1].split(", ")]
        )

    group1_df = df[(df["ALDi"] > 0.11) & (df["ALDi"] < 0.44)]
    group2_df = df[df["ALDi"] >= 0.77]

    # group1_df = df[df["predicted_dialect_26"] == "MSA"]
    # group2_df = df[df["predicted_dialect_26"] != "MSA"]

    # LABEL = "sentiment"
    LABEL = "sarcasm"
    group1_grouped_samples = generate_bootstrap_samples(
        group1_df, n_samples=1000, sample_size=100
    )
    group2_grouped_samples = generate_bootstrap_samples(
        group2_df, n_samples=1000, sample_size=100
    )

    group1_proportions = [
        compute_proportion_full_agreement(sample, label=LABEL)
        for sample in group1_grouped_samples
    ]
    group2_proportions = [
        compute_proportion_full_agreement(sample, label=LABEL)
        for sample in group2_grouped_samples
    ]

    plt.hist(group1_proportions, bins=10, alpha=0.5, label="group1")
    plt.hist(group2_proportions, bins=10, alpha=0.5, label="group2")
    plt.legend()

    u1 = np.mean(group1_proportions)
    u2 = np.mean(group2_proportions)
    s1 = np.std(group1_proportions)
    s2 = np.std(group2_proportions)
    n1 = len(group1_proportions)
    n2 = len(group2_proportions)
    compute_two_sample_ztest(u1, u2, s1, s2, n1, n2)

    statsmodels.stats.weightstats.CompareMeans(
        DescrStatsW(group1_proportions), DescrStatsW(group2_proportions)
    ).ztest_ind(usevar="unequal", alternative="larger", value=0)


if __name__ == "__main__":
    # dataset = "ArSarcasm-v1.tsv"
    dataset = "DCD.tsv"
    main(dataset)
