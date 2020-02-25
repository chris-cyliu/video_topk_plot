import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, NullFormatter
from matplotlib.backends.backend_pdf import PdfPages

mpl.rcParams['axes.linewidth'] = 1
mpl.rc("font", family='sans-serif')

# chris: global makers list
makers = ["d", "x", "s", "^", "+", ">", "o"]
fillstyle = ["none", "full"]
LINEWIDTH = 1


def plot_overall(in_path, out_path, k, w):
    with open(in_path, 'rt') as f:
        result = json.load(f)
    idx = None
    for i, ki in enumerate(result[0]["k"]):
        if ki == k:
            idx = i

    dataset_name = [dataset["dataset"] for dataset in result]
    baseline = [dataset["runtime"]["baseline"] for dataset in result]
    everest = [dataset["runtime"]["topk"][idx] + dataset["runtime"]["split"] +
               dataset["runtime"]["train"] + dataset["runtime"]["infer"] +
               dataset["runtime"]["cdf"] + 7000 / 30 for dataset in result]
    x_pos = np.arange(len(result))

    fig, ax = plt.subplots(figsize=(5.4, 3.7))

    ax.bar(x_pos - w / 2 * 1.1, baseline, color="#FFFFFF", edgecolor="#000000",
           width=w, align='center')
    ax.bar(x_pos + w / 2 * 1.1, everest, color="#000000", width=w,
           align='center')
    ax.set_ylabel('runtime(s)', fontsize=15, labelpad=0)
    ax.set_yscale('log')
    ax.set_yticks([10 ** i for i in range(3, 7)])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dataset_name, fontsize=8)
    ax.xaxis.set_tick_params(width=1, which='both', length=4, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.tick_params('x', pad=5)
    fig.subplots_adjust(left=0.15)
    fig.set_figwidth(8)
    for i, y in enumerate(baseline):
        coord = ax.transAxes.inverted().transform(
            ax.transData.transform([i - w / 2 * 1.1, y]))
        ax.text(coord[0], coord[1] + 0.02, "1.0x", horizontalalignment='center',
                fontsize=15, color="grey", transform=ax.transAxes)
    for i, y in enumerate(everest):
        coord = ax.transAxes.inverted().transform(
            ax.transData.transform([i + w / 2 * 1.1, y]))
        ax.text(coord[0], coord[1] + 0.02, "%.1fx" % (baseline[i] / y),
                horizontalalignment='center', fontsize=15, color="black",
                transform=ax.transAxes)
    ax.legend(["baseline", "Everest"], fontsize=15, framealpha=0)
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0)


def plot_quality_vs_k(in_path, out_prefix):
    with open(in_path, 'rt') as f:
        result = json.load(f)

    styles = ['-', '--', '-.', '-', '--', '-.', '-']
    colors = ['#73A9CF', 'c', 'm', 'g', 'r', 'b', 'y']

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xlim([15, 105])
    ax.set_ylim([0, 50])
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Speedup', fontsize=15, labelpad=0)
    ax.set_xlabel('K', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(result):
        k = dataset["k"]
        runtime = dataset["runtime"]
        prev_runtime = runtime["split"] + runtime["train"] + runtime["infer"] + \
                       runtime["cdf"]
        prev_runtime += 7000 / 30
        runtime_i = [t + prev_runtime for t in runtime["topk"]]
        speedup = [runtime["baseline"] / t for t in runtime_i]
        line, = ax.plot(k, speedup, styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, labels, bbox_to_anchor=(0, 0.8, 1, 0.2), loc="upper left",
               mode="expand", ncol=2, fontsize=15, framealpha=0)
    with PdfPages(out_prefix + "speedup_vs_k.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlim([15, 105])
    ax.set_ylim([0, 1.05])
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Precision', fontsize=15, labelpad=0)
    ax.set_xlabel('K', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(result):
        k = dataset["k"]
        precision = dataset["precision"]
        line, = ax.plot(k, precision, styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, labels, bbox_to_anchor=(0, 0, 1, 0.2), loc="lower left",
               mode="expand", ncol=2, fancybox=False, framealpha=0, fontsize=15)
    with PdfPages(out_prefix + "precision_vs_k.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xlim([15, 105])
    ax.set_ylim([0, 2])
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Score Error', fontsize=15, labelpad=0)
    ax.set_xlabel('K', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(result):
        k = dataset["k"]
        score_error = dataset["score error"]
        line, = ax.plot(k, score_error, styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, labels, bbox_to_anchor=(0, 0.8, 1, 0.2), loc="upper left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)
    with PdfPages(out_prefix + "score_error_vs_k.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xlim([15, 105])
    ax.set_ylim([0, 10])
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Rank Distance', fontsize=15, labelpad=0)
    ax.set_xlabel('K', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(result):
        k = dataset["k"]
        rank_distance = dataset["rank distance"]
        line, = ax.plot(k, rank_distance, styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, labels, bbox_to_anchor=(0, 1.05, 1, 0.2),
               loc="upper left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)
    with PdfPages(out_prefix + "rank_distance_vs_k.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0)


def plot_quality_vs_confidence(in_path, out_prefix):
    with open(in_path, 'rt') as f:
        result = json.load(f)

    styles = ['-', '--', '-.', '-', '--', '-.', '-']
    colors = ['#73A9CF', 'c', 'm', 'g', 'r', 'b', 'y']

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xlim([0.5, 1])
    ax.set_ylim([0, 50])
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=5.4, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.set_ylabel('Speedup', fontsize=15, labelpad=0)
    ax.set_xlabel('Confidence', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(result):
        confidence = dataset["conf"]
        runtime = dataset["runtime"]
        prev_runtime = runtime["split"] + runtime["train"] + runtime["infer"] + \
                       runtime["cdf"]
        prev_runtime += 7000 / 30
        runtime_i = [t + prev_runtime for t in runtime["topk"]]
        speedup = [runtime["baseline"] / t for t in runtime_i]
        line, = ax.plot(confidence, speedup, styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, labels, bbox_to_anchor=(0, 0.8, 1, 0.2), loc="upper left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)
    with PdfPages(out_prefix + "speedup_vs_confidence.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlim([0.5, 1])
    ax.set_ylim([0, 1.05])
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=5.4, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.set_ylabel('Precision', fontsize=15, labelpad=0)
    ax.set_xlabel('Confidence', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []

    for i, dataset in enumerate(result):
        confidence = dataset["conf"]
        precision = dataset["precision"]
        line, = ax.plot(confidence, precision, styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=13,
                        markeredgewidth=1)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, labels, bbox_to_anchor=(0, -0.06, 1.0, 0.2),
               loc="lower left", mode="expand", ncol=2, framealpha=0,
               fontsize=15)
    with PdfPages(out_prefix + "precision_vs_confidence.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xlim([0.5, 1])
    ax.set_ylim([0, 2])
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Score Error', fontsize=15, labelpad=0)
    ax.set_xlabel('Confidence', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(result):
        confidence = dataset["conf"]
        score_error = dataset["score error"]
        line, = ax.plot(confidence, score_error, styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, labels, bbox_to_anchor=(0, 0.8, 1, 0.2), loc="upper left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)
    with PdfPages(out_prefix + "score_error_vs_confidence.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xlim([0.5, 1])
    ax.set_ylim([-0.5, 10])
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Rank Distance', fontsize=15, labelpad=0)
    ax.set_xlabel('Confidence', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(result):
        confidence = dataset["conf"]
        rank_distance = dataset["rank distance"]
        line, = ax.plot(confidence, rank_distance, styles[i],
                        linewidth=LINEWIDTH, color="black",
                        marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, labels, bbox_to_anchor=(0, 1, 1, 0.2), loc="lower left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)
    with PdfPages(out_prefix + "rank_distance_vs_confidence.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0)




if __name__ == "__main__":
    plot_overall("result/quality_vs_k.json", "result/fig/runtime.pdf", 50, 0.3)
    plot_quality_vs_k("result/quality_vs_k.json", "result/fig/")
    plot_quality_vs_confidence("result/quality_vs_confidence.json",
                               "result/fig/")
