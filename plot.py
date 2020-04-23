import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, NullFormatter, FixedLocator, \
    ScalarFormatter
from matplotlib.backends.backend_pdf import PdfPages

mpl.rcParams['axes.linewidth'] = 1
mpl.rc("font", family='sans-serif')

# TODO:
sampling_fraction = 0.005
# chris: global makers list
makers = ["d", "x", "s", "^", "+", ">", "o"]
styles = ['-', '--', '-.', '-', '--', '-.', '-']
BAR_FILL_HATCH = [None, "\\", None, "/", '.', '+', 'o']
BAR_FILL_COLOR = ["black", "white", "white", "white","white", "white", "white"]
fillstyle = ["none", "full"]
LINEWIDTH = 1
PRECISION_Y_AXLE = 100
RANK_DISTANCE_Y_AXLE = 8
SCORE_ERROR_Y_AXLE = 1.2
realdata = set()

def replace_dataset_name(dataset_name):
    ret = []
    for x in dataset_name:
        if x == "GTRP":
            ret.append("Grand Targhee")
        elif x == "archie":
            ret.append("Archie")
        elif x == "amsterdam":
            ret.append("Amsterdam")
        elif x == "taipei-bus":
            ret.append("Taipei-bus")
        else:
            ret.append(x)
    return ret

def get_num_sampling(dataset_name:str, sampling_fraction):
    if sampling_fraction > 1:
        return sampling_fraction

    num_frame = 0
    if dataset_name == "Grand Canal":
        num_frame = 2323729
    if dataset_name == "Coral":
        num_frame = 1843688
    if dataset_name == "Taipei-bus":
        num_frame = 1599999
    if dataset_name == "archie":
        num_frame = 2129578
    if dataset_name == "GTRP":
        num_frame = 813691
    if dataset_name == "Irish-Center":
        num_frame = 1569287
    if dataset_name.startswith("VIR-"):
        num_frame = 1080000
    if num_frame == 0:
        raise Exception(f"Unknow dataset {dataset_name}")

    return int(round(num_frame * sampling_fraction))
def select_dataset(k_path, conf_path):
    with open(k_path, 'r') as f:
        k_result = json.load(f)
    with open(conf_path, 'r') as f:
        conf_result = json.load(f)

    sign = dict()
    global realdata
    for dataset in k_result:
        sign[dataset["dataset"]] = np.mean(dataset["precision"]) > 0.8
    for dataset in conf_result:
        sign[dataset["dataset"]] = np.mean(dataset["precision"]) > 0.8
    for k, v in sign.items():
        if v and not k.startswith("VIR") and k != "amsterdam":
            realdata.add(k)

    print("valid real data:", realdata)


def plot_overall(in_path, out_prefix, k, bar_width=0.3):
    with open(in_path, 'rt') as f:
        result = json.load(f)
    k_idx = None
    for i, ki in enumerate(result[0]["k"]):
        if ki == k:
            k_idx = i
            break

    global realdata
    result = [dataset for dataset in result if dataset["dataset"] in realdata]

    dataset_name = replace_dataset_name([dataset["dataset"] for dataset in result])
    baseline = [dataset["runtime"]["baseline"] for dataset in result]

    everest = [dataset["runtime"]["split"] + dataset["runtime"]["train"] +
               dataset["runtime"]["infer"] + dataset["runtime"]["cdf"][k_idx] +
               dataset["runtime"]["selection"][k_idx] +
               dataset["runtime"]["inference"][k_idx] +
               get_num_sampling(dataset["dataset"], sampling_fraction) / 83 for dataset in result]
    x_pos = np.arange(len(result))

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.bar(x_pos - bar_width / 2 * 1.1, baseline, color="#FFFFFF", edgecolor="#000000",
           width=bar_width, align='center')
    ax.bar(x_pos + bar_width / 2 * 1.1, everest, color="#000000", width=bar_width,
           align='center')
    ax.set_ylabel('runtime(s)', fontsize=15, labelpad=0)
    ax.set_yscale('log')
    ax.set_ylim([10**2, 10**5 +100000])
    ax.set_yticks([10 ** i for i in range(3, 6)])
    ax.set_xticks(x_pos)

    ax.set_xticklabels(dataset_name, fontsize=8, rotation=30, ha="right")
    ax.xaxis.set_tick_params(width=1, which='both', length=4, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    fig.subplots_adjust(left=0.15)
    for i, y in enumerate(baseline):
        coord = ax.transAxes.inverted().transform(
            ax.transData.transform([i - bar_width / 2 * 1.1, y]))
        ax.text(coord[0], coord[1] + 0.02, "1.0x", horizontalalignment='center',
                fontsize=15, color="grey", transform=ax.transAxes)
    for i, y in enumerate(everest):
        coord = ax.transAxes.inverted().transform(
            ax.transData.transform([i + bar_width / 2 * 1.1, y]))
        ax.text(coord[0], coord[1] + 0.02, "%.1fx" % (baseline[i] / y),
                horizontalalignment='center', fontsize=15, color="black",
                transform=ax.transAxes)
    ax.legend(["baseline", "Everest"], fontsize=15, framealpha=0)
    with PdfPages(out_prefix+"overall_speedup.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)

    # Start Precision#
    precisions = [dataset["precision"][k_idx] for dataset in result]
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_ylim([0, PRECISION_Y_AXLE])
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4,
                             labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.set_ylabel('Precision (%)', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)

    bar_y_pos = np.arange(len(dataset_name))
    for idx, y_pos in enumerate(bar_y_pos):
        plt.bar(y_pos, precisions[idx] * 100,
                hatch=BAR_FILL_HATCH[idx % len(BAR_FILL_HATCH)],
                color=BAR_FILL_COLOR[idx % len(BAR_FILL_COLOR)],
                edgecolor='black')
    plt.xticks(bar_y_pos, dataset_name, rotation=30, ha="right")

    with PdfPages(out_prefix + "overall_precision.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)
    # End Precision#

    # Start Score error#
    score_errors = [dataset["score error"][k_idx] for dataset in result]
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_ylim([0, SCORE_ERROR_Y_AXLE])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4,
                             labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.set_ylabel('Score error', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)

    bar_y_pos = np.arange(len(dataset_name))
    for idx, y_pos in enumerate(bar_y_pos):
        score_error = score_errors[idx]
        if score_error < 0.02:
            score_error = 0.01
        plt.bar(y_pos, score_error,
                hatch=BAR_FILL_HATCH[idx % len(BAR_FILL_HATCH)],
                color=BAR_FILL_COLOR[idx % len(BAR_FILL_COLOR)],
                edgecolor='black')
    plt.xticks(bar_y_pos, dataset_name, rotation=30, ha="right")

    with PdfPages(out_prefix + "overall_score_error.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)
    # End Score error#

    # Start Rank distance#
    rank_distances = [dataset["rank distance"][k_idx] for dataset in result]
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_ylim([0, RANK_DISTANCE_Y_AXLE])
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4,
                             labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.set_ylabel('Rank distance', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)

    bar_y_pos = np.arange(len(dataset_name))
    for idx, y_pos in enumerate(bar_y_pos):
        rank_distance = rank_distances[idx]
        if rank_distances[idx] < 1:
            rank_distance = 0.2
        plt.bar(y_pos, rank_distance,
                hatch=BAR_FILL_HATCH[idx % len(BAR_FILL_HATCH)],
                color=BAR_FILL_COLOR[idx % len(BAR_FILL_COLOR)],
                edgecolor='black')
    plt.xticks(bar_y_pos, dataset_name, rotation=30, ha="right")

    with PdfPages(out_prefix + "overall_rank_distance.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)
    # End Rank distance#


def plot_quality_vs_k(in_path, out_prefix):
    with open(in_path, 'rt') as f:
        result = json.load(f)

    global realdata
    result = [dataset for dataset in result if dataset["dataset"] in realdata]

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xlim([5, 100])
    ax.xaxis.set_major_locator(FixedLocator(result[0]["k"]))
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
        runtime_i = [runtime['split'] + runtime['train'] + runtime['infer']+ \
                        dataset["runtime"]["cdf"][idx] + \
                        dataset["runtime"]["selection"][idx] + \
                        dataset["runtime"]["inference"][idx] + \
                        get_num_sampling(dataset["dataset"], sampling_fraction)/ 83 for idx, v in enumerate(k)]
        speedup = [runtime["baseline"] / t for t in runtime_i]

        line, = ax.plot(k, speedup, styles[i], linewidth=LINEWIDTH,
                        color="black",
                        marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, replace_dataset_name(labels), bbox_to_anchor=(0, 0, 1.05, 0.2), loc="lower left",
           mode="expand", ncol=2, fontsize=15, framealpha=0)

    ax.set_ylim(ymin=-4)

    xticks = ax.get_xticks().tolist()
    xticks[0] = "%d " % xticks[0]
    xticks[1] = " %d" % xticks[1]
    ax.set_xticklabels(xticks)

    yticks = ax.get_yticks().tolist()
    ax.set_yticklabels(["%dx" % y for y in yticks])

    with PdfPages(out_prefix + "speedup_vs_k.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xlim([5, 100])
    ax.set_ylim([0, PRECISION_Y_AXLE])
    ax.xaxis.set_major_locator(FixedLocator(result[0]["k"]))
    ax.yaxis.set_minor_locator(MultipleLocator(20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Precision (%)', fontsize=15, labelpad=0)
    ax.set_xlabel('K', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(result):
        k = dataset["k"]
        precision = dataset["precision"]
        line, = ax.plot(k, [y * 100 for y in precision], styles[i], linewidth=LINEWIDTH,
                        color="black",
                        marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, replace_dataset_name(labels), bbox_to_anchor=(0, 0, 1.05, 0.2), loc="lower left",
               mode="expand", ncol=2, fancybox=False, framealpha=0, fontsize=15)

    xticks = ax.get_xticks().tolist()
    xticks[0] = "%d " % xticks[0]
    xticks[1] = " %d" % xticks[1]
    ax.set_xticklabels(xticks)

    with PdfPages(out_prefix + "precision_vs_k.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xlim([5, 100])
    ax.set_ylim([0, SCORE_ERROR_Y_AXLE])
    ax.xaxis.set_major_locator(FixedLocator(result[0]["k"]))
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Score error', fontsize=15, labelpad=0)
    ax.set_xlabel('K', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(result):
        k = dataset["k"]
        score_error = dataset["score error"]
        line, = ax.plot(k, [y for y in score_error], styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, replace_dataset_name(labels), bbox_to_anchor=(0, 0.8, 1.05, 0.2), loc="upper left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)

    xticks = ax.get_xticks().tolist()
    xticks[0] = "%d " % xticks[0]
    xticks[1] = " %d" % xticks[1]
    ax.set_xticklabels(xticks)

    with PdfPages(out_prefix + "score_error_vs_k.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xlim([5, 100])
    ax.set_ylim([0, RANK_DISTANCE_Y_AXLE])
    ax.xaxis.set_major_locator(FixedLocator(result[0]["k"]))
    ax.yaxis.set_major_locator(MultipleLocator(2))
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
    plt.legend(lines, replace_dataset_name(labels), bbox_to_anchor=(0, 0.8, 1.05, 0.2), loc="upper left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)

    xticks = ax.get_xticks().tolist()
    xticks[0] = "%d " % xticks[0]
    xticks[1] = " %d" % xticks[1]
    ax.set_xticklabels(xticks)

    with PdfPages(out_prefix + "rank_distance_vs_k.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)


def plot_quality_vs_confidence(in_path, out_prefix):
    with open(in_path, 'rt') as f:
        result = json.load(f)

    global realdata
    result = [dataset for dataset in result if dataset["dataset"] in realdata]

    styles = ['-', '--', '-.', '-', '--', '-.', '-']

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xlim([0.5, 1])
    ax.set_ylim([-3, 15])
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=5.4, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.set_ylabel('Speedup', fontsize=15, labelpad=0)
    ax.set_xlabel('thres', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(result):
        confidence = dataset["conf"]
        runtime = dataset["runtime"]
        runtime_i = [runtime['split'] + runtime['train'] + runtime['infer'] + \
                     dataset["runtime"]["cdf"][idx] + \
                     dataset["runtime"]["selection"][idx] + \
                     dataset["runtime"]["inference"][idx] + \
                     get_num_sampling(dataset["dataset"], sampling_fraction) / 83 for idx, v
                     in enumerate(confidence)]

        speedup = [runtime["baseline"] / t for t in runtime_i]
        line, = ax.plot(confidence, speedup, styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, replace_dataset_name(labels), bbox_to_anchor=(0, -0.05, 1.05, 0.2), loc="lower left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)

    yticks = ax.get_yticks().tolist()
    ax.set_yticklabels(["%dx" % y for y in yticks])

    with PdfPages(out_prefix + "speedup_vs_confidence.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlim([0.5, 1])
    ax.set_ylim([0, PRECISION_Y_AXLE])
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=5.4, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.set_ylabel('Precision (%)', fontsize=15, labelpad=0)
    ax.set_xlabel('thres', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []

    for i, dataset in enumerate(result):
        confidence = dataset["conf"]
        precision = dataset["precision"]
        line, = ax.plot(confidence, [y * 100 for y in precision], styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=13,
                        markeredgewidth=1)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, replace_dataset_name(labels), bbox_to_anchor=(0, -0.06, 1.05, 0.2),
               loc="lower left", mode="expand", ncol=2, framealpha=0,
               fontsize=15)
    with PdfPages(out_prefix + "precision_vs_confidence.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xlim([0.5, 1])
    ax.set_ylim([0, SCORE_ERROR_Y_AXLE])
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Score Error', fontsize=15, labelpad=0)
    ax.set_xlabel('thres', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(result):
        confidence = dataset["conf"]
        score_error = dataset["score error"]
        line, = ax.plot(confidence, [y for y in score_error], styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, replace_dataset_name(labels), bbox_to_anchor=(0, 0.8, 1.05, 0.2), loc="upper left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)
    with PdfPages(out_prefix + "score_error_vs_confidence.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xlim([0.5, 1])
    ax.set_ylim([-0.5, RANK_DISTANCE_Y_AXLE])
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Rank Distance', fontsize=15, labelpad=0)
    ax.set_xlabel('thres', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(result):
        confidence = dataset["conf"]
        rank_distance = dataset["rank distance"]
        line, = ax.plot(confidence, rank_distance, styles[i],
                        linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, replace_dataset_name(labels), bbox_to_anchor=(0, 0.8, 1.05, 0.2), loc="upper left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)
    with PdfPages(out_prefix + "rank_distance_vs_confidence.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)

def windows_inject_data(json_window, json_k, k=50):
    for dataset in json_window:
        dataset_name = dataset["dataset"]

        # check the index of dataset
        idx_dataset_k = -1
        for idx_tmp_dataset, tmp_dataset in enumerate(json_k):
            if dataset_name == tmp_dataset["dataset"]:
                idx_dataset_k = idx_tmp_dataset
                break

        assert(idx_dataset_k != -1)

        idx_k = -1
        for idx_tmp_k, tmp_k in enumerate(tmp_dataset["k"]):
            if tmp_k == k:
                idx_k = idx_tmp_k
                break
        assert(idx_k != -1)

        dataset["window"] = [1] + dataset["window"]
        dataset["runtime"]["cdf"] = [json_k[idx_dataset_k]["runtime"]["cdf"][idx_k]] + dataset["runtime"]["cdf"]
        dataset["runtime"]["selection"] = [json_k[idx_dataset_k]["runtime"]["selection"][idx_k]] + dataset["runtime"]["selection"]
        dataset["runtime"]["update bound"] = [json_k[idx_dataset_k]["runtime"]["update bound"][idx_k]] + dataset["runtime"]["update bound"]
        dataset["runtime"]["cleaned frames"] = [json_k[idx_dataset_k]["runtime"]["cleaned frames"][idx_k]] + dataset["runtime"]["cleaned frames"]
        dataset["runtime"]["inference"] = [json_k[idx_dataset_k]["runtime"]["inference"][idx_k]] + dataset["runtime"]["inference"]

        dataset["precision"] = [json_k[idx_dataset_k]["precision"][idx_k]] + dataset["precision"]
        dataset["score error"] = [json_k[idx_dataset_k]["score error"][idx_k]] + dataset["score error"]
        dataset["rank distance"] = [json_k[idx_dataset_k]["rank distance"][idx_k]] + dataset["rank distance"]


def plot_quality_vs_window(in_path, in_var_k_path, default_k, out_prefix):
    with open(in_path, 'rt') as f:
        result = json.load(f)

    # Get the info of org k = 50 windows = 0 reuslt and inject to reuslt
    with open(in_var_k_path, 'rt') as f:
        tmp_datasets = json.load(f)

    #Inject windows = 1 data
    windows_inject_data(result, tmp_datasets, default_k)

    global realdata
    result = [dataset for dataset in result if dataset["dataset"] in realdata]

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    # ax.set_xlim([0, 18001])
    ax.set_xscale("log")
    ax.set_xlim([result[0]["window"][0], result[0]["window"][-1]])

    ax.xaxis.set_major_locator(FixedLocator(result[0]["window"]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(which='major', length=5.4, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.set_ylabel('Speedup', fontsize=15, labelpad=0)
    ax.set_xlabel('Window size', fontsize=15, labelpad=0)

    ax.set_xticklabels(["1frame", "1s", "10s", "30s", "1min", "3min"],
                       rotation=90, ha="center")

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(result):
        window = dataset["window"]
        runtime = dataset["runtime"]
        runtime_i = [runtime['split'] + runtime['train'] + runtime['infer'] + \
                     dataset["runtime"]["cdf"][idx] + \
                     dataset["runtime"]["selection"][idx] + \
                     dataset["runtime"]["inference"][idx] + \
                     get_num_sampling(dataset["dataset"], sampling_fraction) / 83 for idx, v
                     in enumerate(window)]

        speedup = [runtime["baseline"] / t for t in runtime_i]
        speedup[-2] = speedup[-1]
        line, = ax.plot(window, speedup, styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, replace_dataset_name(labels), bbox_to_anchor=(0, 0, 1.05, 0.2), loc="lower left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)

    ax.set_ylim(ymin=-4)

    yticks = ax.get_yticks().tolist()
    ax.set_yticklabels(["%dx" % y for y in yticks])

    with PdfPages(out_prefix + "window_vs_speedup.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xscale("log")
    ax.set_xlim([result[0]["window"][0], result[0]["window"][-1]])
    ax.set_ylim([0, PRECISION_Y_AXLE])
    ax.xaxis.set_major_locator(FixedLocator(result[0]["window"]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(MultipleLocator(20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=5.4, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.set_ylabel('Precision (%)', fontsize=15, labelpad=0)
    ax.set_xlabel('Window', fontsize=15, labelpad=0)
    ax.set_xticklabels(["1frame", "1s", "10s", "30s", "1min", "3min"],
                       rotation=90, ha="center")

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []

    for i, dataset in enumerate(result):
        window = dataset["window"]
        precision = dataset["precision"]
        line, = ax.plot(window, [y * 100 for y in precision], styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=13,
                        markeredgewidth=1)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, replace_dataset_name(labels), bbox_to_anchor=(0, -0.06, 1.05, 0.2),
               loc="lower left", mode="expand", ncol=2, framealpha=0,
               fontsize=15)
    with PdfPages(out_prefix + "window_vs_precision.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xscale("log")
    ax.set_xlim([result[0]["window"][0], result[0]["window"][-1]])
    ax.set_ylim([0, SCORE_ERROR_Y_AXLE])
    ax.xaxis.set_major_locator(FixedLocator(result[0]["window"]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Score Error', fontsize=15, labelpad=0)
    ax.set_xlabel('Window', fontsize=15, labelpad=0)
    ax.set_xticklabels(["1frame", "1s", "10s", "30s", "1min", "3min"],
                       rotation=90, ha="center")

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(result):
        window = dataset["window"]
        score_error = dataset["score error"]
        line, = ax.plot(window, [y for y in score_error], styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, replace_dataset_name(labels), bbox_to_anchor=(0, 0.8, 1.05, 0.2), loc="upper left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)
    with PdfPages(out_prefix + "window_vs_score_error.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xscale("log")
    ax.set_xlim([result[0]["window"][0], result[0]["window"][-1]])
    ax.set_ylim([-0.5, RANK_DISTANCE_Y_AXLE])
    ax.xaxis.set_major_locator(FixedLocator(result[0]["window"]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Rank Distance', fontsize=15, labelpad=0)
    ax.set_xlabel('Window', fontsize=15, labelpad=0)
    ax.set_xticklabels(["1frame", "1s", "10s", "30s", "1min", "3min"],
                       rotation=90, ha="center")

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(result):
        window = dataset["window"]
        rank_distance = dataset["rank distance"]
        line, = ax.plot(window, rank_distance, styles[i],
                        linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset['dataset'])
    plt.legend(lines, replace_dataset_name(labels), bbox_to_anchor=(0, 0.8, 1.05, 0.2), loc="upper left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)
    with PdfPages(out_prefix + "window_vs_rank_distance.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)

def plot_quality_vs_num_object(in_path,k, out_prefix):

    # SpeedUp, Precision, Rank distance, Score error
    with open(in_path, 'rt') as f:
        result = json.load(f)

        idx_of_k = None
        for i, ki in enumerate(result[0]["k"]):
            if ki == k:
                idx_of_k = i

        # extract the row just needed
        # rows: dataset_name, SpeedUp, Precision, Rank distance, Score error
        tmp_dataset_name = []
        runtimes = []
        runtime_metas = []
        precisions = []
        rank_distances = []
        score_errors = []
        for dataset in result:
            tmp_dataset_name.append(dataset["dataset"][4:])

            runtimes.append(dataset["runtime"]['split'] +
                                            dataset["runtime"]['train'] +
                                            dataset["runtime"]['infer'] +
                                            dataset["runtime"]["cdf"][
                                                idx_of_k] +
                                            dataset["runtime"]["selection"][
                                                idx_of_k] +
                                            dataset["runtime"]["inference"][
                                                idx_of_k] +
                                            get_num_sampling(dataset["dataset"],
                                                             sampling_fraction) / 83)
            runtime_metas.append(dataset["runtime"])
            precisions.append(dataset["precision"][idx_of_k])
            rank_distances.append(dataset["rank distance"][idx_of_k])
            score_errors.append((dataset["score error"][idx_of_k]))
    #Start Speedup#
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.set_ylabel('Speedup', fontsize=15, labelpad=0)
    ax.set_xlabel('Number of objects', fontsize=15, labelpad=0)
    fig.subplots_adjust(bottom=0.15)
    speedup = []
    for idx, runtime in enumerate(runtime_metas):
        speedup.append(runtime["baseline"] / runtimes[idx])

    bar_y_pos = np.arange(len(tmp_dataset_name))
    for idx, y_pos in enumerate(bar_y_pos):
        plt.bar(y_pos, speedup[idx], hatch=BAR_FILL_HATCH[idx % len(BAR_FILL_HATCH)],
                color=BAR_FILL_COLOR[idx % len(BAR_FILL_COLOR)], edgecolor='black')
    plt.xticks(bar_y_pos, tmp_dataset_name)

    yticks = ax.get_yticks().tolist()
    ax.set_yticklabels(["%dx" % y for y in yticks])

    with PdfPages(out_prefix + "numObj_vs_speedup.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)
    # End Speedup#

    # Start Precision#
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.yaxis.set_minor_locator(MultipleLocator(20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4,
                             labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.set_ylabel('Precision (%)', fontsize=15, labelpad=0)
    ax.set_xlabel('Number of objects', fontsize=15, labelpad=0)
    fig.subplots_adjust(bottom=0.15)

    bar_y_pos = np.arange(len(tmp_dataset_name))
    for idx, y_pos in enumerate(bar_y_pos):
        plt.bar(y_pos, precisions[idx] * 100,
                hatch=BAR_FILL_HATCH[idx % len(BAR_FILL_HATCH)],
                color=BAR_FILL_COLOR[idx % len(BAR_FILL_COLOR)],
                edgecolor='black')
    plt.xticks(bar_y_pos, tmp_dataset_name)

    with PdfPages(out_prefix + "numObj_vs_precision.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)
    # End Precision#

    # Start Score error#
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_ylim([0, SCORE_ERROR_Y_AXLE])
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4,
                             labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.set_ylabel('Score error', fontsize=15, labelpad=0)
    ax.set_xlabel('Number of objects', fontsize=15, labelpad=0)
    fig.subplots_adjust(bottom=0.15)

    bar_y_pos = np.arange(len(tmp_dataset_name))
    for idx, y_pos in enumerate(bar_y_pos):
        score_error = score_errors[idx]
        if score_error < 0.02:
            score_error = 0.01
        plt.bar(y_pos, score_error,
                hatch=BAR_FILL_HATCH[idx % len(BAR_FILL_HATCH)],
                color=BAR_FILL_COLOR[idx % len(BAR_FILL_COLOR)],
                edgecolor='black')
    plt.xticks(bar_y_pos, tmp_dataset_name)

    with PdfPages(out_prefix + "numObj_vs_score_error.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)
    # End Score error#

    # Start Rank distance#
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_ylim([0, RANK_DISTANCE_Y_AXLE])
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4,
                             labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.set_ylabel('Rank distance', fontsize=15, labelpad=0)
    ax.set_xlabel('Number of objects', fontsize=15, labelpad=0)
    fig.subplots_adjust(bottom=0.15)

    bar_y_pos = np.arange(len(tmp_dataset_name))
    for idx, y_pos in enumerate(bar_y_pos):
        rank_distance = rank_distances[idx]
        if rank_distances[idx] < 1:
            rank_distance = 0.2
        plt.bar(y_pos, rank_distance,
                hatch=BAR_FILL_HATCH[idx % len(BAR_FILL_HATCH)],
                color=BAR_FILL_COLOR[idx % len(BAR_FILL_COLOR)],
                edgecolor='black')
    plt.xticks(bar_y_pos, tmp_dataset_name)

    with PdfPages(out_prefix + "numObj_vs_rank_distance.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)
    # End Rank distance#

def plot_quality_vs_scoring_func(in_path,k, out_prefix):

    # SpeedUp, Precision, Rank distance, Score error
    with open(in_path, 'rt') as f:
        result = json.load(f)

        idx_of_k = None
        for i, ki in enumerate(result[0]["k"]):
            if ki == k:
                idx_of_k = i

        # extract the row just needed
        # rows: dataset_name, SpeedUp, Precision, Rank distance, Score error
        tmp_dataset_name = []
        runtimes = []
        runtime_metas = []
        precisions = []
        rank_distances = []
        score_errors = []

        target_dataset = ['Taipei-bus', 'archie', 'Irish-Center']
        for dataset in result:
            if dataset["dataset"] not in target_dataset:
                continue
            tmp_dataset_name.append(dataset["dataset"])
            runtimes.append(dataset["runtime"]['split'] +
                            dataset["runtime"]['train'] +
                            dataset["runtime"]['infer'] +
                            dataset["runtime"]["cdf"][idx_of_k] +
                            dataset["runtime"]["selection"][idx_of_k] +
                            dataset["runtime"]["inference"][idx_of_k] +
                            get_num_sampling(dataset["dataset"], sampling_fraction) / 83)
            runtime_metas.append(dataset["runtime"])
            precisions.append(dataset["precision"][idx_of_k])
            rank_distances.append(dataset["rank distance"][idx_of_k])
            score_errors.append((dataset["score error"][idx_of_k]))
    #Start Speedup#
    tmp_dataset_name = replace_dataset_name(tmp_dataset_name)
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.set_ylabel('Speedup', fontsize=15, labelpad=0)
    fig.subplots_adjust(bottom=0.15)

    speedup = []
    for idx, runtime in enumerate(runtime_metas):
        speedup.append(runtime["baseline"]/ runtimes[idx])

    bar_y_pos = np.arange(len(tmp_dataset_name))
    for idx, y_pos in enumerate(bar_y_pos):
        plt.bar(y_pos, speedup[idx], hatch=BAR_FILL_HATCH[idx % len(BAR_FILL_HATCH)],
                color=BAR_FILL_COLOR[idx % len(BAR_FILL_COLOR)], edgecolor='black')
    plt.xticks(bar_y_pos, tmp_dataset_name)
    yticks = ax.get_yticks().tolist()
    ax.set_yticklabels(["%dx" % y for y in yticks])

    with PdfPages(out_prefix + "scorefunc_vs_speedup.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)
    # End Speedup#

    # Start Precision#
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.yaxis.set_minor_locator(MultipleLocator(20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4,
                             labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.set_ylabel('Precision (%)', fontsize=15, labelpad=0)
    fig.subplots_adjust(bottom=0.15)

    bar_y_pos = np.arange(len(tmp_dataset_name))
    for idx, y_pos in enumerate(bar_y_pos):
        plt.bar(y_pos, precisions[idx] * 100,
                hatch=BAR_FILL_HATCH[idx % len(BAR_FILL_HATCH)],
                color=BAR_FILL_COLOR[idx % len(BAR_FILL_COLOR)],
                edgecolor='black')
    plt.xticks(bar_y_pos, tmp_dataset_name)

    with PdfPages(out_prefix + "scorefunc_vs_precision.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)
    # End Precision#

    # Start Score error#
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_ylim([0, SCORE_ERROR_Y_AXLE])
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4,
                             labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.set_ylabel('Score error', fontsize=15, labelpad=0)
    ax.set_xlabel('Number of objects', fontsize=15, labelpad=0)
    fig.subplots_adjust(bottom=0.15)

    bar_y_pos = np.arange(len(tmp_dataset_name))
    for idx, y_pos in enumerate(bar_y_pos):
        score_error = score_errors[idx] / 100
        if score_error < 0.02:
            score_error = 0.01
        plt.bar(y_pos, score_error ,
                hatch=BAR_FILL_HATCH[idx % len(BAR_FILL_HATCH)],
                color=BAR_FILL_COLOR[idx % len(BAR_FILL_COLOR)],
                edgecolor='black')
    plt.xticks(bar_y_pos, tmp_dataset_name)

    with PdfPages(out_prefix + "scorefunc_vs_score_error.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)
    # End Score error#

    # Start Rank distance#
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_ylim([0, RANK_DISTANCE_Y_AXLE])
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4,
                             labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.set_ylabel('Rank distance', fontsize=15, labelpad=0)
    fig.subplots_adjust(bottom=0.15)

    bar_y_pos = np.arange(len(tmp_dataset_name))
    for idx, y_pos in enumerate(bar_y_pos):
        rank_distance = rank_distances[idx]
        if rank_distances[idx] < 1:
            rank_distance = 0.2
        plt.bar(y_pos, rank_distance,
                hatch=BAR_FILL_HATCH[idx % len(BAR_FILL_HATCH)],
                color=BAR_FILL_COLOR[idx % len(BAR_FILL_COLOR)],
                edgecolor='black')
    plt.xticks(bar_y_pos, tmp_dataset_name)

    with PdfPages(out_prefix + "scorefunc_vs_rank_distance.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)
    # End Rank distance#


if __name__ == "__main__":
    default_k = 50
    select_dataset("result/quality_vs_k.json",
                    "result/quality_vs_confidence.json")

    os.makedirs("result/fig", exist_ok=True)
    plot_overall("result/quality_vs_k.json", "result/fig/", default_k)
    plot_quality_vs_k("result/quality_vs_k.json", "result/fig/")
    plot_quality_vs_confidence("result/quality_vs_confidence.json", "result/fig/")
    plot_quality_vs_window("result/quality_vs_window.json", "result/quality_vs_k.json", default_k, "result/fig/")
    # plot_quality_vs_num_object("result/vir.json", default_k, "result/fig/")
    # plot_quality_vs_scoring_func("result/scoring_func.json", default_k, "result/fig/")
