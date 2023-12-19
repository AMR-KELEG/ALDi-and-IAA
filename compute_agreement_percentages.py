import os
import re
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from datasets import load_datasets, LabelType
from scipy import stats

EPS = 1e-6
font = {"weight": "bold", "size": 8}
matplotlib.rc("font", **font)

VIOLET = "#7F007F"
GREY = "#A9A6A7"
GREY = "#D0CFCF"
DARK_GREY = "#837f81"
GREEN = "#1B7837"
ORANGE = "#E66101"

PLOTS_DIR = "plots/"
os.makedirs(PLOTS_DIR, exist_ok=True)

DATASET_TITLES_MAPPING = {
    "arabic_dialect_familiarity": "iSarcasm",
    "Mawqif_stance": "Mawqif",
    "Mawqif_sarcasm": "Mawqif",
    "YouTube_cyberbullying": "YTCB",
    "ArSAS": "ArSAS",
    "ASAD": "ASAD",
    "MPOLD": "MPOLD",
    "DCD": "DCD",
    "DART": "DART",
}


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
    agreement_disagreement_mean_ALDis=None,
    plot_boundaries=False,
    plot_hist=True,
    plot_horizontal_agreement_lines=False,
    plot_mean_ALDis=False,
    figsize=(6.3 / 4, 1),
    ALDi_scores=None,
    label_value=None,
    use_log_scale=False,
    streched_yaxis=False,
    n_bins=10,
):
    plt.figure(figsize=figsize)
    bins_boundaries = [bin_boundaries for bin_boundaries, bin_stats in bins_stats]

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
            use_log_scale=use_log_scale,
        )

    x = [(bin_st + bin_end) / 2 for bin_st, bin_end in bins_boundaries]
    agreement_percentages = [
        bin_stats["%_complete_agreement"] for _, bin_stats in bins_stats
    ]

    print("Diff", max(agreement_percentages) - min(agreement_percentages))

    # Plot horizontal lines for range of agreement scores
    if plot_horizontal_agreement_lines:
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
    ax.set_xlabel("ALDi", fontsize=6)  # , fontdict={"weight": 'bold'})
    ax.xaxis.set_label_coords(-0.1, -0.125)

    # plt.legend(title="", frameon=False, prop={"size": 5})
    ax.set_xlim(0, 1)

    # Fit a polynomial curve
    xp = np.linspace(0, 1, 100)

    coef = np.polyfit(x, agreement_percentages, 1)
    yp = [coef[-1] + coef[0] * x for x in xp]
    _ = ax.plot(xp, yp, "--", color=ORANGE, alpha=0.5)

    # TODO: Understand how to use the p-value!
    try:
        pearson_r_result = stats.pearsonr(x, agreement_percentages)
        pearson_coef, p_value = pearson_r_result.statistic, pearson_r_result.pvalue
        pearson_coef = round(pearson_coef, 2)
    except:
        print("Issue in computing Pearson correlation coefficient!")
        pearson_coef = "N/A"
        p_value = "N/A"

    plt.title(
        f"{re.sub('_', ' ', dataset_name.title() if not dataset_name in DATASET_TITLES_MAPPING else DATASET_TITLES_MAPPING[dataset_name])}"
        + (
            f"\n({label_value}), ρ = {pearson_coef}"
            if label_value
            else f" (ρ = {pearson_coef})"
        ),
        fontsize=5,
    )

    legend = ax.legend(
        labels=[],
        title=f"m = {round(coef[0], 2)}",
        frameon=False,
        prop={"size": 5},
        loc="best",
    )
    legend.get_title().set_fontsize("5")
    legend.get_title().set_fontweight("bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)

    if not streched_yaxis:
        RANGE = 50
        lower_y = 10 * (min(agreement_percentages) // 10)
        upper_y = 10 * (1 + max(agreement_percentages) // 10)
        OBSERVED_RANGE = upper_y - lower_y
        RANGE = max(RANGE, OBSERVED_RANGE)

        if upper_y - lower_y < RANGE:
            offset = RANGE - (upper_y - lower_y)
            lower_y -= offset / 2
            upper_y += offset / 2

        if upper_y > 105:
            upper_y = 105
            lower_y = upper_y - RANGE

        if lower_y < 0:
            lower_y = 0
            upper_y = lower_y + RANGE
        yticks = [v for v in range(int(lower_y), int(upper_y), 10)]

    else:
        lower_y = 0
        upper_y = 105
        yticks = [v for v in range(int(lower_y), int(upper_y), 20)]

    ax.set_ylim(lower_y, upper_y)

    if plot_mean_ALDis:
        ax.plot(
            [
                agreement_disagreement_mean_ALDis[0],
                agreement_disagreement_mean_ALDis[0],
            ],
            [lower_y, upper_y],
            "g--",
        )
        ax.plot(
            [
                agreement_disagreement_mean_ALDis[1],
                agreement_disagreement_mean_ALDis[1],
            ],
            [lower_y, upper_y],
            "r--",
        )

    xticks = [0.2, 0.4, 0.6, 0.8]
    ax.set_xticks(
        ticks=xticks, labels=[str(round(xtick, 1)) for xtick in xticks], fontsize=5
    )
    ax.set_yticks(ticks=yticks, labels=yticks, fontsize=5)

    plt.tight_layout()
    label_value = re.sub(r" ", r"-", label_value) if label_value else None
    plt.savefig(
        Path(
            PLOTS_DIR,
            f"{dataset_name}_{n_bins}_{label_name}{'_merged' if plot_hist else ''}{f'_{label_value}' if label_value else ''}.pdf",
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
    use_log_scale=False,
):
    if figsize:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        ax_other = plt.gca()
        ax = ax_other.twinx()

    ax.hist(ALDi_scores, bins=bins, color=VIOLET, alpha=alpha, log=use_log_scale)

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

    if use_log_scale:
        Y_LIM = 100000
        ax.set_ylim(1, Y_LIM)
        if show_yticks:
            yticks = [int(v) for v in list(ax.get_yticks())[1:]]
            ax.set_yticks(ticks=yticks, labels=yticks, fontsize=6)
        else:
            ax.set_yticks([10**p for p in range(0, 6)], fontsize=4)
    else:
        # Set the ylim
        Y_LIMITS = [50, 120, 500, 1400, 7000, 15000, 25000, 100000]

        for lim in Y_LIMITS:
            if lim > most_n_samples_per_bin:
                Y_LIM = lim
                break
        # yticks = [int(v) for v in list(ax.get_yticks())[1:]] + [least_n_samples_per_bin]
        if show_yticks:
            yticks = [int(v) for v in list(ax.get_yticks())[1:]]
            ax.set_yticks(ticks=yticks, labels=yticks, fontsize=6)
        else:
            yticks = [least_n_samples_per_bin, most_n_samples_per_bin]
            ax.set_yticks(ticks=yticks, labels=yticks, fontsize=4)

        # Set Y_LIM based on the number of samples in the most populated bin
        Y_LIM = most_n_samples_per_bin
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

        if n_samples:
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

    print(label_name, label_type)
    print([stats["%_complete_agreement"] for bin, stats in total_agreement_percentages])

    return total_agreement_percentages


def compute_mean_ALDi_scores(df, label_name, label_type):
    """Compute the mean ALDi scores for samples with full agreement and samples with disagreement"""
    is_complete_aggreement = None
    label_annotations = df[label_name].tolist()

    if label_type == LabelType.CONF:
        # Complete agreement samples
        is_complete_aggreement = [
            abs(confidence - 1) < EPS for label, confidence in label_annotations
        ]

    elif label_type == LabelType.PROPORTION:
        # Complete agreement samples
        is_complete_aggreement = [
            abs(confidence - 1) < EPS for label, confidence in label_annotations
        ]

    else:
        is_complete_aggreement = [len(set(labels)) == 1 for labels in label_annotations]
    df["is_complete_aggreement"] = is_complete_aggreement

    mean_ALDi_full_agreement = df.loc[df["is_complete_aggreement"], "ALDi"].mean()
    mean_ALDi_disagreement = df.loc[~df["is_complete_aggreement"], "ALDi"].mean()

    return round(mean_ALDi_full_agreement, 2), round(mean_ALDi_disagreement, 2)


if __name__ == "__main__":
    datasets = load_datasets()
    for dataset in datasets:
        dataset_name = dataset.dataset_name
        df = dataset.df
        boundaries_values = [
            [0, 0.11, 0.44, 0.77, 1.0],
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            [i / 20 for i in range(0, 21)],
        ]

        for bins_boundaries in boundaries_values:
            # Generate the plots for each independent label within the dataset
            for label, label_type in zip(dataset.labels, dataset.labels_types):
                # Visualize the overall trend
                bins_stats = generate_bins(
                    df,
                    label,
                    label_type,
                    bins_boundaries=bins_boundaries,
                )
                # Compute the mean ALDi scores for samples with full agreement and samples with disagreement
                # Note: This is only needed if plot_mean_ALDis=True for the generate_scatter_plot function
                mean_ALDi_scores = compute_mean_ALDi_scores(df, label, label_type)
                generate_scatter_plot(
                    dataset_name,
                    label,
                    bins_stats,
                    ALDi_scores=df["ALDi"],
                    agreement_disagreement_mean_ALDis=mean_ALDi_scores,
                    plot_boundaries=True,
                    plot_horizontal_agreement_lines=False,
                    plot_mean_ALDis=False,
                    use_log_scale=False,
                    n_bins=len(bins_boundaries) - 1,
                )

                # Visualize the per label-value trend
                unique_label_values = sorted(set(df[f"{label}_majority_vote"].unique()))

                for unique_label_value in unique_label_values:
                    df_unique_label_value = df.loc[
                        df[f"{label}_majority_vote"] == unique_label_value
                    ]
                    bins_stats = generate_bins(
                        df_unique_label_value,
                        label,
                        label_type,
                        bins_boundaries=bins_boundaries,
                    )
                    mean_ALDi_scores = compute_mean_ALDi_scores(
                        df_unique_label_value, label, label_type
                    )
                    generate_scatter_plot(
                        dataset_name,
                        label,
                        bins_stats,
                        ALDi_scores=df_unique_label_value["ALDi"],
                        agreement_disagreement_mean_ALDis=mean_ALDi_scores,
                        plot_boundaries=True,
                        plot_horizontal_agreement_lines=False,
                        plot_mean_ALDis=False,
                        label_value=unique_label_value,
                        use_log_scale=False,
                        streched_yaxis=True,
                        n_bins=len(bins_boundaries) - 1,
                    )
