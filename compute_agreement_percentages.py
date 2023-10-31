import os
import re
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
from datasets import load_datasets, LabelType

EPS = 1e-6
font = {"weight": "bold", "size": 8}
matplotlib.rc("font", **font)

VIOLET = "#7F007F"
GREY = "#A9A6A7"
GREY = "#D0CFCF"
DARK_GREY = "#837f81"
GREEN = "#1B7837"

PLOTS_DIR = "plots/"
os.makedirs(PLOTS_DIR, exist_ok=True)


def generate_scatter_plot(
    dataset_name,
    label_name,
    bins_stats,
    figsize=(6.3 / 4, 1.25),
):
    plt.figure(figsize=figsize)
    bins_boundaries = [bin_boundaries for bin_boundaries, bin_stats in bins_stats]
    n_bins = len(bins_boundaries)

    # Plot the bin boundaries
    for bin_st, _ in bins_boundaries[1:]:
        plt.plot([bin_st, bin_st], [0, 100], "k:", alpha=0.1)

    x = [(bin_st + bin_end) / 2 for bin_st, bin_end in bins_boundaries]
    agreement_percentages = [
        bin_stats["%_complete_agreement"] for _, bin_stats in bins_stats
    ]

    plt.plot(
        [0, 1],
        [min(agreement_percentages), min(agreement_percentages)],
        "k--",
        alpha=0.2,
    )
    plt.plot(
        [0, 1],
        [max(agreement_percentages), max(agreement_percentages)],
        "k--",
        alpha=0.2,
    )

    plt.scatter(x=x, y=agreement_percentages, color=VIOLET, label=label_name, s=4)
    # plt.ylabel("% of samples of complete agreement")
    # plt.xlabel("ALDi scores",)

    plt.ylim(
        40
        if not (
            dataset_name == "arabic_dialect_familiarity" and label_name == "dialect"
        )
        else 0,
        105
        if not (
            dataset_name == "arabic_dialect_familiarity" and label_name == "dialect"
        )
        else 65,
    )
    # plt.legend(title="", frameon=False, prop={"size": 5})
    plt.xlim(0, 1)

    ### Fit a polynomial curve
    ### xp = np.linspace(0, 1, 100)

    ### 2nd degree polynomial
    ### coef = np.polyfit(x, agreement_percentages, 2)
    ### yp = [coef[-1] + coef[1]*x + coef[0] * (x**2) for x in xp]

    ### coef = np.polyfit(x, agreement_percentages, 1)
    ### yp = [coef[-1] + coef[0] * x for x in xp]
    ### _ = plt.plot(xp, yp, "--", color=VIOLET, alpha=0.2)

    pearson_coef = np.corrcoef(x, agreement_percentages)[1, 0]

    # plt.title(
    #     f"Pearson's coefficient={round(pearson_coef, 2)}, "
    #     f"Aggrement(ALDi)={round(coef[0], 2)}* ALDi + {round(coef[-1], 2)}"
    # )

    plt.title(
        re.sub("_", " ", label_name.capitalize()),
        fontsize=8,
    )

    ax = plt.gca()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    # plt.yticks(list(plt.yticks()[0]) + [min(agreement_percentages), max(agreement_percentages)], fontsize=6)

    plt.tight_layout()
    plt.savefig(
        Path(PLOTS_DIR, f"{dataset_name}_{n_bins}_{label_name}.pdf"),
        bbox_inches="tight",
    )


def compute_percentage_of_agreement(
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


def generate_ALDi_histograms(
    ALDi_scores, dataset_name, figsize=(6.3 * 0.3, 1.5), bins=None
):
    plt.figure(figsize=figsize)
    plt.hist(ALDi_scores, bins=bins, color=VIOLET)
    ax = plt.gca()

    n_samples_per_bin = {
        (bin_st, bin_end): ((ALDi_scores >= bin_st) & (ALDi_scores < bin_end)).sum()
        if bin_end != 1
        else ((ALDi_scores >= bin_st) & (ALDi_scores <= bin_end)).sum()
        for bin_st, bin_end in zip(bins[:-1], bins[1:])
    }

    least_n_samples_per_bin = min(n_samples_per_bin.values())

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    # Set the ylim
    Y_LIM = None
    if dataset_name in ["ArSarcasm-v1", "YouTube_cyberbullying"]:
        Y_LIM = 7000
    elif dataset_name in [
        "iSarcasm_third_party",
        "MPOLD",
        "Mawqif_stance",
        "Mawqif_sarcasm",
        "arabic_dialect_familiarity",
        "LetMI",
    ]:
        Y_LIM = 1400
    else:
        Y_LIM = 15000
    plt.ylim(0, Y_LIM)

    plt.xticks(bins[1:], rotation=90)
    plt.yticks(list(plt.yticks()[0])[1:] + [least_n_samples_per_bin])

    plt.tight_layout()
    plt.savefig(
        Path(PLOTS_DIR, f"{dataset_name}_{len(bins)-1}_ALDi.pdf"),
        bbox_inches="tight",
    )


def generate_bins(df, label_name, label_type, bins_boundaries):
    total_agreement_percentages = []
    for bin_st, bin_end in zip(bins_boundaries[:-1], bins_boundaries[1:]):
        if bin_end == 1:
            bin_label_values = df.loc[
                (df["ALDi"] >= bin_st) & df["ALDi"] <= bin_end, label_name
            ]
        else:
            bin_label_values = df.loc[
                (df["ALDi"] >= bin_st) & df["ALDi"] < bin_end, label_name
            ]

        if label_type == LabelType.CONF:
            # Complete agreement samples
            n_complete_agreement_samples = sum(
                [confidence == 1 for label, confidence in bin_label_values.tolist()]
            )
        elif label_type == LabelType.PROPORTION:
            # Complete agreement samples
            n_complete_agreement_samples = sum(
                [
                    abs(confidence - 1) < EPS
                    for label, confidence in bin_label_values.tolist()
                ]
            )
        else:
            n_complete_agreement_samples = sum(
                [len(set(labels)) == 1 for labels in bin_label_values.tolist()]
            )

        n_samples = bin_label_values.shape[0]

        total_agreement_percentages.append(
            (
                (bin_st, bin_end),
                {
                    "#_samples": n_samples,
                    "#_complete_agreement": n_complete_agreement_samples,
                    "%_complete_agreement": round(
                        100 * n_complete_agreement_samples / n_samples, 2
                    ),
                },
            )
        )

    print(label_name, label_type)
    print([stats["%_complete_agreement"] for bin, stats in total_agreement_percentages])
    return total_agreement_percentages


if __name__ == "__main__":
    i = 0
    datasets = load_datasets()
    for dataset in datasets:
        dataset_name = dataset.dataset_name
        df = dataset.df
        generate_ALDi_histograms(
            df["ALDi"],
            dataset_name=dataset_name,
            bins=[0, 0.11, 0.44, 0.77, 1],
        )
        generate_ALDi_histograms(
            df["ALDi"],
            dataset_name=dataset_name,
            bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        )

        for label, label_type in zip(dataset.labels, dataset.labels_types):
            print(i)
            bins_stats = generate_bins(
                df,
                label,
                label_type,
                # bins_boundaries=[0, 0.11, 0.44, 0.77, 1],
                bins_boundaries=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            )
            i += 1
            generate_scatter_plot(dataset_name, label, bins_stats)
