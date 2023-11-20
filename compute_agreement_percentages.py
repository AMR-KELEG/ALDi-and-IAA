import os
import re
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
from datasets import load_datasets, LabelType
from matplotlib.ticker import FormatStrFormatter
from pprint import pprint

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


def plot_bins_boundaries(bins_boundaries):
    ax_other = plt.gca()
    ax = ax_other.twinx()

    # Plot the bin boundaries
    for bin_st, _ in bins_boundaries[1:]:
        ax.plot([bin_st, bin_st], [0, 100], "k:", alpha=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def generate_scatter_plot(
    dataset_name,
    label_name,
    bins_stats,
    plot_boundaries=False,
    plot_hist=True,
    figsize=(6.3 / 4, 1.25),
    ALDi_scores=None,
):
    plt.figure(figsize=figsize)
    bins_boundaries = [bin_boundaries for bin_boundaries, bin_stats in bins_stats]
    n_bins = len(bins_boundaries)

    ax = plt.gca()

    if plot_boundaries:
        plot_bins_boundaries(bins_boundaries)

    if plot_hist:
        generate_ALDi_histograms(
            ALDi_scores,
            dataset_name,
            figsize=None,
            bins=[0] + [bin_end for bin_st, bin_end in bins_boundaries],
            save_fig=False,
            alpha=0.2,
            show_yticks=False,
        )

    x = [(bin_st + bin_end) / 2 for bin_st, bin_end in bins_boundaries]
    agreement_percentages = [
        bin_stats["%_complete_agreement"] for _, bin_stats in bins_stats
    ]

    print("Diff", max(agreement_percentages) - min(agreement_percentages))
    # Plot horizontal lines for range of agreement scores
    ax.plot(
        [0, 1],
        [min(agreement_percentages), min(agreement_percentages)],
        "k--",
        alpha=0.2,
    )
    ax.plot(
        [0, 1],
        [max(agreement_percentages), max(agreement_percentages)],
        "k--",
        alpha=0.2,
    )

    ax.scatter(x=x, y=agreement_percentages, color=VIOLET, label=label_name, s=4)
    ax.set_ylabel("% full agree", fontsize=6)
    ax.set_xlabel("ALDi", fontsize=6)

    # plt.legend(title="", frameon=False, prop={"size": 5})
    ax.set_xlim(0, 1)

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
        re.sub("_", " ", label_name.capitalize())
        + f"- {re.sub('_', ' ', dataset_name.capitalize())}",
        fontsize=8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)

    RANGE = 40
    lower_y = 10 * (min(agreement_percentages) // 10)
    upper_y = 10 * (1 + max(agreement_percentages) // 10)

    if upper_y - lower_y < RANGE:
        offset = RANGE - (upper_y - lower_y)
        lower_y -= offset / 2
        upper_y += offset / 2

    if upper_y > 105:
        upper_y = 105
        lower_y = upper_y - RANGE
    ax.set_ylim(lower_y, upper_y)

    xticks = [0.2, 0.4, 0.6, 0.8]
    ax.set_xticks(
        ticks=xticks, labels=[str(round(xtick, 1)) for xtick in xticks], fontsize=5
    )
    yticks = [v for v in range(int(lower_y), int(upper_y), 10)]
    ax.set_yticks(ticks=yticks, labels=yticks, fontsize=5)

    plt.tight_layout()
    plt.savefig(
        Path(
            PLOTS_DIR,
            f"{dataset_name}_{n_bins}_{label_name}{'_merged' if plot_hist else ''}.pdf",
        ),
        bbox_inches="tight",
    )


def generate_ALDi_histograms(
    ALDi_scores,
    dataset_name,
    figsize=(6.3 * 0.3, 1.5),
    bins=None,
    save_fig=False,
    alpha=1,
    show_yticks=True,
):
    if figsize:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        ax_other = plt.gca()
        ax = ax_other.twinx()

    ax.hist(ALDi_scores, bins=bins, color=VIOLET, alpha=alpha)

    n_samples_per_bin = {
        (bin_st, bin_end): ((ALDi_scores >= bin_st) & (ALDi_scores < bin_end)).sum()
        if bin_end != 1
        else ((ALDi_scores >= bin_st) & (ALDi_scores <= bin_end)).sum()
        for bin_st, bin_end in zip(bins[:-1], bins[1:])
    }

    least_n_samples_per_bin = min(n_samples_per_bin.values())
    most_n_samples_per_bin = max(n_samples_per_bin.values())

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    ax.set_xticks(ticks=bins[1:], labels=bins[1:], rotation=90, fontsize=6)

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
        "qweet",
        "L-HSAB",
    ]:
        Y_LIM = 1400
    elif dataset_name in ["ASAD"]:
        Y_LIM = 25000
    else:
        Y_LIM = 15000

    # yticks = [int(v) for v in list(ax.get_yticks())[1:]] + [least_n_samples_per_bin]
    if show_yticks:
        yticks = [int(v) for v in list(ax.get_yticks())[1:]]
        ax.set_yticks(ticks=yticks, labels=yticks, fontsize=6)
    else:
        yticks = [least_n_samples_per_bin, most_n_samples_per_bin]
        ax.set_yticks(ticks=yticks, labels=yticks, fontsize=4)
    ax.set_ylim(0, Y_LIM)

    plt.tight_layout()

    if save_fig:
        plt.savefig(
            Path(PLOTS_DIR, f"{dataset_name}_{len(bins)-1}_ALDi.pdf"),
            bbox_inches="tight",
        )


def generate_bins(df, label_name, label_type, bins_boundaries):
    total_agreement_percentages = []
    for bin_st, bin_end in zip(bins_boundaries[:-1], bins_boundaries[1:]):
        if bin_end == 1:
            bin_label_values = df.loc[
                (df["ALDi"] >= bin_st) & (df["ALDi"] <= bin_end), label_name
            ]
        else:
            bin_label_values = df.loc[
                (df["ALDi"] >= bin_st) & (df["ALDi"] < bin_end), label_name
            ]

        if label_type == LabelType.CONF:
            # Complete agreement samples
            complete_agreement_labels = [
                label
                for label, confidence in bin_label_values.tolist()
                if abs(confidence - 1) < EPS
            ]

        elif label_type == LabelType.PROPORTION:
            # Complete agreement samples
            complete_agreement_labels = [
                label
                for label, confidence in bin_label_values.tolist()
                if abs(confidence - 1) < EPS
            ]

        else:
            complete_agreement_labels = [
                labels[0]
                for labels in bin_label_values.tolist()
                if len(set(labels)) == 1
            ]

        unique_labels = sorted(set(complete_agreement_labels))
        n_complete_agreement_samples = len(complete_agreement_labels)

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
                    "%_complete_agreement": round(
                        100 * n_complete_agreement_samples / n_samples, 2
                    ),
                },
            )
        )

        for label in unique_labels:
            total_agreement_percentages[-1][-1][
                f"%_complete_agreement_{label_name}_{label}"
            ] = round(
                100 * sum([l == label for l in complete_agreement_labels]) / n_samples,
                2,
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
            save_fig=True,
        )
        generate_ALDi_histograms(
            df["ALDi"],
            dataset_name=dataset_name,
            bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            save_fig=True,
        )

        for label, label_type in zip(dataset.labels, dataset.labels_types):
            print(i, dataset_name)
            bins_stats = generate_bins(
                df,
                label,
                label_type,
                # bins_boundaries=[0, 0.11, 0.44, 0.77, 1],
                bins_boundaries=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            )
            i += 1
            generate_scatter_plot(
                dataset_name,
                label,
                bins_stats,
                ALDi_scores=df["ALDi"],
                plot_boundaries=True,
            )
