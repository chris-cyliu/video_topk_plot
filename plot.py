import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, NullFormatter, FixedLocator, \
    ScalarFormatter
from matplotlib.backends.backend_pdf import PdfPages

import pandas

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
RANK_DISTANCE_Y_AXLE = 16
SCORE_ERROR_Y_AXLE = 1.2
realdata = set()

DEFAULT_K=50

K_FILE = "k.json"
SCORE_FUNC_FILE = "score_func.json"
VARY_LENGHT_FILE ="vary_length.json"
WINDOW_FILE = "window_small.json"
CONF_FILE = "conf.json"

SCORE_FUNC_DATASET = ["archie", "long_irish_center", "daxi_old_street_80h", "long_taipei_bus"]
VARY_LENGTH_DATASET = ["long_irish_center", "long_taipei_bus"]

VIDEO_LENGTH = [10, 50, 100, 150, 300]

def get_df_vs(path:str):
    ret =  pandas.read_json(path, lines=True, convert_dates=False)
    ret["dataset"] = ret["dataset"].str.extract(r'-(\d+)')
    return ret

def get_df_k(path: str):
    pds = []
    for dir in os.listdir(path):
        if dir in ["visualroad.json", "noscope.json"]:
            continue
        target_file = path+"/"+dir+"/"+K_FILE
        tmp_pd = pandas.read_json(target_file, lines=True, convert_dates=False)
        pds.append(tmp_pd)

    ret = pandas.concat(pds, ignore_index=True, sort=True)

    return ret

def get_df_conf(path: str):
    pds = []
    for dir in os.listdir(path):
        if dir in ["visualroad.json", "noscope.json"]:
            continue
        target_file = path+"/"+dir+"/"+CONF_FILE
        tmp_pd = pandas.read_json(target_file, lines=True, convert_dates=False)
        pds.append(tmp_pd)

    ret = pandas.concat(pds, ignore_index=True, sort=True)

    return ret
def get_df_score_func(path: str):
    pds = []
    for dataset in SCORE_FUNC_DATASET:
        target_file = path+"/"+dataset+"/"+SCORE_FUNC_FILE
        tmp_pd = pandas.read_json(target_file, lines=True, convert_dates=False)
        pds.append(tmp_pd)

    return pandas.concat(pds, ignore_index=True, sort=True)

def get_df_vary_length(path: str):
    pds = []
    for dataset in VARY_LENGTH_DATASET:
        target_file = path+"/"+dataset+"/"+VARY_LENGHT_FILE
        tmp_pd = pandas.read_json(target_file, lines=True, convert_dates=False)
        video_length = tmp_pd["dataset"].str.extract(r'(\d+)')
        tmp_pd["length"] = video_length.astype(int)
        if dataset == "long_irish_center":
            tmp_pd["dataset"] = "Irish-Center-long"
        elif dataset == "long_taipei_bus":
            tmp_pd["dataset"] = "Taipei-bus-long"
        else:
            raise Exception
        pds.append(tmp_pd)

    return pandas.concat(pds, ignore_index=True, sort=True)

def get_df_window(path: str):
    pds = []
    for dir in os.listdir(path):
        if dir in ["visualroad.json", "noscope.json"]:
            continue
        tmp_pd = pandas.read_json(path +"/"+dir + "/" + WINDOW_FILE, lines=True, convert_dates=False)
        tmp_pd = tmp_pd.loc[tmp_pd['window'] != 1]

        tmp_pd_2 = pandas.read_json(path +"/"+dir + "/" + K_FILE, lines=True, convert_dates=False)
        tmp_pd_2 = tmp_pd_2.loc[tmp_pd_2['k'] == DEFAULT_K]

        tmp_pd = pandas.concat([tmp_pd, tmp_pd_2], ignore_index=True, sort=True)
        tmp_pd = tmp_pd.sort_values(by=['k'])

        pds.append(tmp_pd)

    return pandas.concat(pds, ignore_index=True, sort=True)

def get_df_noscope(path: str):
    ret = pandas.read_json(path, lines=True, convert_dates=False)
    return ret



def replace_dataset_names(dataset_name):
    ret = []
    for x in dataset_name:
        if x == "Coral_long":
            ret.append("Coral")
        elif x == "archie":
            ret.append("Archie")
        elif x == "Irish-Center-long":
            ret.append("Irish-Center")
        elif x == "Taipei-bus-long":
            ret.append("Taipei-bus")
        elif x == "Grand_Canal_long":
            ret.append("Grand Canal")
        elif x == "Lamai_Tai_Street_long":
            ret.append("Lamai Tai street")
        elif x == "Daxi_Old_Street-long":
            ret.append("Daxi old street")
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


def plot_overall(df, df_noscope, out_prefix, bar_width=0.3):
    dataset_names = df["dataset"].tolist()

    baseline = df["baseline"].tolist()
    max_baseline = df["baseline"].max()
    everest = df["runtime"].tolist()

    # get noscope baseline
    noscope = []
    for tmp_dataset_name in dataset_names:
        tmp_noscope_ret = df_noscope.loc[df_noscope["dataset"] == tmp_dataset_name, "runtime"].values
        assert len(tmp_noscope_ret) == 1
        noscope.append(tmp_noscope_ret[0])

    dataset_names = replace_dataset_names(dataset_names)

    x_pos = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.bar(x_pos - bar_width , baseline, color="#FFFFFF", edgecolor="#000000",
           width=bar_width, align='center')
    ax.bar(x_pos, noscope, color="grey",
           edgecolor="#000000",
           width=bar_width, align='center')
    ax.bar(x_pos + bar_width , everest, color="#000000", width=bar_width,
           align='center')
    ax.set_ylabel('runtime(s)', fontsize=15, labelpad=0)
    ax.set_yscale('log')
    ax.set_ylim([10**2, 10**7.5])
    ax.set_yticks([10 ** i for i in range(3, 7)])
    ax.set_xticks(x_pos)

    ax.set_xticklabels(dataset_names, fontsize=8, rotation=30, ha="right")
    ax.xaxis.set_tick_params(width=1, which='both', length=4, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    fig.subplots_adjust(left=0.15)
    for i, y in enumerate(baseline):
        coord = ax.transAxes.inverted().transform(
            ax.transData.transform([i - bar_width / 2 * 1.1, y]))
        ax.text(coord[0], coord[1] + 0.02, "1.0x", horizontalalignment='center',
                fontsize=15, color="grey", transform=ax.transAxes)

    # for i, y in enumerate(noscope):
    #     coord = ax.transAxes.inverted().transform(
    #         ax.transData.transform([i - bar_width / 2 * 1.1, y]))
    #     ax.text(coord[0], coord[1] + 0.02, "1.0x", horizontalalignment='center',
    #             fontsize=15, color="white", transform=ax.transAxes)

    for i, y in enumerate(everest):
        coord = ax.transAxes.inverted().transform(
            ax.transData.transform([i + bar_width / 2 * 1.1, y]))
        ax.text(coord[0], coord[1] + 0.02, "%.1fx" % (baseline[i] / y),
                horizontalalignment='center', fontsize=15, color="black",
                transform=ax.transAxes)
    ax.legend(["baseline", "noscope", "Everest"], fontsize=15, framealpha=0)
    with PdfPages(out_prefix+"overall_speedup.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)

    # Start Precision#
    precisions = df["precision"].tolist()
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_ylim([0, PRECISION_Y_AXLE])
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4,
                             labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.set_ylabel('Precision (%)', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)

    bar_y_pos = np.arange(len(dataset_names))
    for idx, y_pos in enumerate(bar_y_pos):
        plt.bar(y_pos, precisions[idx] * 100,
                hatch=BAR_FILL_HATCH[idx % len(BAR_FILL_HATCH)],
                color=BAR_FILL_COLOR[idx % len(BAR_FILL_COLOR)],
                edgecolor='black')
    plt.xticks(bar_y_pos, dataset_names, rotation=30, ha="right")

    with PdfPages(out_prefix + "overall_precision.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)
    # End Precision#

    # Start Score error#
    score_errors = df["score_error"].tolist()
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_ylim([0, SCORE_ERROR_Y_AXLE])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4,
                             labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.set_ylabel('Score error', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)

    bar_y_pos = np.arange(len(dataset_names))
    for idx, y_pos in enumerate(bar_y_pos):
        score_error = score_errors[idx]
        if score_error < 0.02:
            score_error = 0.01
        plt.bar(y_pos, score_error,
                hatch=BAR_FILL_HATCH[idx % len(BAR_FILL_HATCH)],
                color=BAR_FILL_COLOR[idx % len(BAR_FILL_COLOR)],
                edgecolor='black')
    plt.xticks(bar_y_pos, dataset_names, rotation=30, ha="right")

    with PdfPages(out_prefix + "overall_score_error.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)
    # End Score error#

    # Start Rank distance#
    rank_distances = df["rank_dist"].tolist()
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

    bar_y_pos = np.arange(len(dataset_names))
    for idx, y_pos in enumerate(bar_y_pos):
        rank_distance = rank_distances[idx]
        if rank_distances[idx] < 1:
            rank_distance = 0.2
        plt.bar(y_pos, rank_distance,
                hatch=BAR_FILL_HATCH[idx % len(BAR_FILL_HATCH)],
                color=BAR_FILL_COLOR[idx % len(BAR_FILL_COLOR)],
                edgecolor='black')
    plt.xticks(bar_y_pos, dataset_names, rotation=30, ha="right")

    with PdfPages(out_prefix + "overall_rank_distance.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)
    # End Rank distance#


def plot_quality_vs_k(df, out_prefix):
    target_ks = df["k"].unique().tolist()

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xlim([5, 100])
    ax.xaxis.set_major_locator(FixedLocator(target_ks))
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Speedup', fontsize=15, labelpad=0)
    ax.set_xlabel('K', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(df["dataset"].unique().tolist()):
        tmp_df = df.loc[df["dataset"] == dataset]
        k = tmp_df["k"].tolist()
        speedup = tmp_df["speedup"].tolist()

        line, = ax.plot(k, speedup, styles[i], linewidth=LINEWIDTH,
                        color="black",
                        marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset)
    plt.legend(lines, replace_dataset_names(labels), bbox_to_anchor=(0, 0, 1.05, 0.2), loc="lower left",
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
    ax.xaxis.set_major_locator(FixedLocator(target_ks))
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
    for i, dataset in enumerate(df["dataset"].unique().tolist()):
        tmp_df = df.loc[df["dataset"] == dataset]
        k = tmp_df["k"].tolist()
        precision = tmp_df["precision"].tolist()
        line, = ax.plot(k, [y * 100 for y in precision], styles[i], linewidth=LINEWIDTH,
                        color="black",
                        marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset)
    plt.legend(lines, replace_dataset_names(labels), bbox_to_anchor=(0, 0, 1.05, 0.2), loc="lower left",
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
    ax.xaxis.set_major_locator(FixedLocator(target_ks))
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Score error', fontsize=15, labelpad=0)
    ax.set_xlabel('K', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(df["dataset"].unique().tolist()):
        tmp_df = df.loc[df["dataset"] == dataset]
        k = tmp_df["k"].tolist()
        score_error = tmp_df["score_error"].tolist()

        line, = ax.plot(k, [y for y in score_error], styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset)
    plt.legend(lines, replace_dataset_names(labels), bbox_to_anchor=(0, 0.8, 1.05, 0.2), loc="upper left",
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
    ax.xaxis.set_major_locator(FixedLocator(target_ks))
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
    for i, dataset in enumerate(df["dataset"].unique().tolist()):
        tmp_df = df.loc[df["dataset"] == dataset]
        k = tmp_df["k"].tolist()
        rank_distance = tmp_df["rank_dist"].tolist()

        line, = ax.plot(k, rank_distance, styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset)
    plt.legend(lines, replace_dataset_names(labels), bbox_to_anchor=(0, 0.8, 1.05, 0.2), loc="upper left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)

    xticks = ax.get_xticks().tolist()
    xticks[0] = "%d " % xticks[0]
    xticks[1] = " %d" % xticks[1]
    ax.set_xticklabels(xticks)

    with PdfPages(out_prefix + "rank_distance_vs_k.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)


def plot_quality_vs_confidence(df, out_prefix):

    styles = ['-', '--', '-.', '-', '--', '-.', '-']

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xlim([0.5, 1])
    ax.set_ylim([-3, 20])
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
    for i, dataset in enumerate(df["dataset"].unique().tolist()):
        sub_df = df.loc[df["dataset"] == dataset]
        confidence = sub_df["conf_thres"].tolist()
        speedup = sub_df["speedup"].tolist()

        line, = ax.plot(confidence, speedup, styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset)
    plt.legend(lines, replace_dataset_names(labels), bbox_to_anchor=(0, -0.05, 1.05, 0.2), loc="lower left",
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

    for i, dataset in enumerate(df["dataset"].unique().tolist()):
        sub_df = df.loc[df["dataset"] == dataset]
        confidence = sub_df["conf_thres"].tolist()
        precision = sub_df["precision"].tolist()

        line, = ax.plot(confidence, [y * 100 for y in precision], styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=13,
                        markeredgewidth=1)
        lines.append(line)
        labels.append(dataset)
    plt.legend(lines, replace_dataset_names(labels), bbox_to_anchor=(0, -0.06, 1.05, 0.2),
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
    for i, dataset in enumerate(df["dataset"].unique().tolist()):
        sub_df = df.loc[df["dataset"] == dataset]
        confidence = sub_df["conf_thres"].tolist()
        score_error = sub_df["score_error"].tolist()

        line, = ax.plot(confidence, [y for y in score_error], styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset)
    plt.legend(lines, replace_dataset_names(labels), bbox_to_anchor=(0, 0.8, 1.05, 0.2), loc="upper left",
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
    for i, dataset in enumerate(df["dataset"].unique().tolist()):
        sub_df = df.loc[df["dataset"] == dataset]
        confidence = sub_df["conf_thres"].tolist()
        rank_distance = sub_df["rank_dist"].tolist()
        line, = ax.plot(confidence, rank_distance, styles[i],
                        linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset)
    plt.legend(lines, replace_dataset_names(labels), bbox_to_anchor=(0, 0.8, 1.05, 0.2), loc="upper left",
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


def plot_quality_vs_window(df, out_prefix):

    manual_x_axle = ["1frame", "1s", "2s", "5s", "10s"]
    # manual_x_axle = ["1frame", "1s", "10s", "30s", "1min", "3min"]
    df = df.sort_values(by=['window'])
    windows = df["window"].unique().tolist()

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    # ax.set_xlim([0, 18001])
    ax.set_xscale("log")
    ax.set_xlim([windows[0], windows[-1]])

    ax.xaxis.set_major_locator(FixedLocator(windows))
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

    ax.set_xticklabels(manual_x_axle,
                       rotation=90, ha="center")

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(df["dataset"].unique().tolist()):
        sub_df = df.loc[df["dataset"] == dataset]
        window = sub_df["window"].tolist()
        speedup = sub_df["speedup"].tolist()

        #todo: check if we need to apply this trick again later
        # speedup[-2] = speedup[-1]
        line, = ax.plot(window, speedup, styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset)
    plt.legend(lines, replace_dataset_names(labels), bbox_to_anchor=(0, 0, 1.05, 0.2), loc="lower left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)

    ax.set_ylim(ymin=-4)

    yticks = ax.get_yticks().tolist()
    ax.set_yticklabels(["%dx" % y for y in yticks])

    with PdfPages(out_prefix + "window_vs_speedup.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xscale("log")
    ax.set_xlim([windows[0], windows[-1]])
    ax.set_ylim([0, PRECISION_Y_AXLE])
    ax.xaxis.set_major_locator(FixedLocator(windows))
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
    ax.set_xticklabels(manual_x_axle,
                       rotation=90, ha="center")


    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []

    for i, dataset in enumerate(df["dataset"].unique().tolist()):
        sub_df = df.loc[df["dataset"] == dataset]
        window = sub_df["window"].tolist()
        precision = sub_df["precision"].tolist()

        line, = ax.plot(window, [y * 100 for y in precision], styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=13,
                        markeredgewidth=1)
        lines.append(line)
        labels.append(dataset)
    plt.legend(lines, replace_dataset_names(labels), bbox_to_anchor=(0, -0.06, 1.05, 0.2),
               loc="lower left", mode="expand", ncol=2, framealpha=0,
               fontsize=15)
    with PdfPages(out_prefix + "window_vs_precision.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xscale("log")
    ax.set_xlim([windows[0], windows[-1]])
    ax.set_ylim([0, SCORE_ERROR_Y_AXLE])
    ax.xaxis.set_major_locator(FixedLocator(windows))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Score Error', fontsize=15, labelpad=0)
    ax.set_xlabel('Window', fontsize=15, labelpad=0)
    ax.set_xticklabels(manual_x_axle,
                       rotation=90, ha="center")

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(df["dataset"].unique().tolist()):
        sub_df = df.loc[df["dataset"] == dataset]
        window = sub_df["window"].tolist()
        score_error = sub_df["score_error"].tolist()

        line, = ax.plot(window, [y for y in score_error], styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset)
    plt.legend(lines, replace_dataset_names(labels), bbox_to_anchor=(0, 0.8, 1.05, 0.2), loc="upper left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)
    with PdfPages(out_prefix + "window_vs_score_error.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_xscale("log")
    ax.set_xlim([windows[0], windows[-1]])
    ax.set_ylim([-0.5, RANK_DISTANCE_Y_AXLE])
    ax.xaxis.set_major_locator(FixedLocator(windows))
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
    ax.set_xticklabels(manual_x_axle,
                       rotation=90, ha="center")

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(df["dataset"].unique().tolist()):
        sub_df = df.loc[df["dataset"] == dataset]
        window = sub_df["window"].tolist()
        rank_distance = sub_df["rank_dist"].tolist()

        line, = ax.plot(window, rank_distance, styles[i],
                        linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset)
    plt.legend(lines, replace_dataset_names(labels), bbox_to_anchor=(0, 0.8, 1.05, 0.2), loc="upper left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)
    with PdfPages(out_prefix + "window_vs_rank_distance.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)

def plot_quality_vs_num_object(df, out_prefix):
    tmp_dataset_name = df["dataset"]

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

    speedup = df["speedup"].tolist()
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

    precisions = df["precision"].tolist()

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

    score_errors = df["score_error"].tolist()
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

    rank_distances = df["rank_dist"].tolist()
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

def plot_quality_vs_scoring_func(df, out_prefix):

    # SpeedUp, Precision, Rank distance, Score error
    tmp_dataset_name = df["dataset"].tolist()

    #Start Speedup#
    tmp_dataset_name = replace_dataset_names(tmp_dataset_name)
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=5.4, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=3.7)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.set_ylabel('Speedup', fontsize=15, labelpad=0)
    fig.subplots_adjust(bottom=0.15)

    speedup = df["speedup"].tolist()
    precisions = df["precision"].tolist()
    score_errors = df["score_error"].tolist()
    rank_distances = df["rank_dist"].tolist()

    bar_y_pos = np.arange(len(tmp_dataset_name))
    for idx, y_pos in enumerate(bar_y_pos):
        plt.bar(y_pos, speedup[idx], hatch=BAR_FILL_HATCH[idx % len(BAR_FILL_HATCH)],
                color=BAR_FILL_COLOR[idx % len(BAR_FILL_COLOR)], edgecolor='black')
    plt.xticks(bar_y_pos, tmp_dataset_name, rotation=30)
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
    plt.xticks(bar_y_pos, tmp_dataset_name,rotation=30)

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
    plt.xticks(bar_y_pos, tmp_dataset_name, rotation=30)

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
    plt.xticks(bar_y_pos, tmp_dataset_name, rotation=30)

    with PdfPages(out_prefix + "scorefunc_vs_rank_distance.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)
    # End Rank distance#

def plot_quality_vs_videolength(df, out_prefix):
    fig, ax = plt.subplots(figsize=(5.4, 3.7))

    ax.set_xlim([VIDEO_LENGTH[0], VIDEO_LENGTH[-1]])
    ax.xaxis.set_major_locator(FixedLocator(VIDEO_LENGTH))
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Speedup', fontsize=15, labelpad=0)
    ax.set_xlabel('Video length', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(df["dataset"].unique().tolist()):
        tmp_df = df.loc[df["dataset"] == dataset]
        video_length = tmp_df["length"].tolist()
        speedup = tmp_df["speedup"].tolist()

        line, = ax.plot(video_length, speedup, styles[i], linewidth=LINEWIDTH,
                        color="black",
                        marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset)
    plt.legend(lines, replace_dataset_names(labels), bbox_to_anchor=(0, 0, 1.05, 0.2), loc="lower left",
               mode="expand", ncol=2, fontsize=15, framealpha=0)

    ax.set_ylim(ymin=-4)

    xticks = ax.get_xticks().tolist()
    xticks[0] = "%d " % xticks[0]
    xticks[1] = " %d" % xticks[1]
    ax.set_xticklabels(xticks)

    yticks = ax.get_yticks().tolist()
    ax.set_yticklabels(["%dx" % y for y in yticks])

    with PdfPages(out_prefix + "videolength_vs_speedup.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_ylim([0, PRECISION_Y_AXLE])
    ax.set_xlim([VIDEO_LENGTH[0], VIDEO_LENGTH[-1]])
    ax.xaxis.set_major_locator(FixedLocator(VIDEO_LENGTH))
    ax.yaxis.set_minor_locator(MultipleLocator(20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Precision (%)', fontsize=15, labelpad=0)
    ax.set_xlabel('Video length', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(df["dataset"].unique().tolist()):
        tmp_df = df.loc[df["dataset"] == dataset]
        video_length = tmp_df["length"].tolist()
        precision = tmp_df["precision"].tolist()
        line, = ax.plot(video_length, [y * 100 for y in precision], styles[i], linewidth=LINEWIDTH,
                        color="black",
                        marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset)
    plt.legend(lines, replace_dataset_names(labels), bbox_to_anchor=(0, 0, 1.05, 0.2), loc="lower left",
               mode="expand", ncol=2, fancybox=False, framealpha=0, fontsize=15)

    xticks = ax.get_xticks().tolist()
    xticks[0] = "%d " % xticks[0]
    xticks[1] = " %d" % xticks[1]
    ax.set_xticklabels(xticks)

    with PdfPages(out_prefix + "videolength_vs_precision.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_ylim([0, SCORE_ERROR_Y_AXLE])
    ax.set_xlim([VIDEO_LENGTH[0], VIDEO_LENGTH[-1]])
    ax.xaxis.set_major_locator(FixedLocator(VIDEO_LENGTH))
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Score error', fontsize=15, labelpad=0)
    ax.set_xlabel('Video length', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(df["dataset"].unique().tolist()):
        tmp_df = df.loc[df["dataset"] == dataset]
        video_length = tmp_df["length"].tolist()
        score_error = tmp_df["score_error"].tolist()

        line, = ax.plot(video_length, [y for y in score_error], styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset)
    plt.legend(lines, replace_dataset_names(labels), bbox_to_anchor=(0, 0.8, 1.05, 0.2), loc="upper left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)

    xticks = ax.get_xticks().tolist()
    xticks[0] = "%d " % xticks[0]
    xticks[1] = " %d" % xticks[1]
    ax.set_xticklabels(xticks)

    with PdfPages(out_prefix + "videolength_vs_score_error.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.set_ylim([0, RANK_DISTANCE_Y_AXLE])
    ax.set_xlim([VIDEO_LENGTH[0], VIDEO_LENGTH[-1]])
    ax.xaxis.set_major_locator(FixedLocator(VIDEO_LENGTH))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.yaxis.set_tick_params(width=1, which='minor', length=4)
    ax.xaxis.set_tick_params(width=1, which='major', length=6, labelsize=15)
    ax.xaxis.set_tick_params(width=1, which='minor', length=4)
    ax.set_ylabel('Rank Distance', fontsize=15, labelpad=0)
    ax.set_xlabel('Video length', fontsize=15, labelpad=0)

    fig.subplots_adjust(bottom=0.15)
    lines = []
    labels = []
    for i, dataset in enumerate(df["dataset"].unique().tolist()):
        tmp_df = df.loc[df["dataset"] == dataset]
        video_length = tmp_df["length"].tolist()
        rank_distance = tmp_df["rank_dist"].tolist()

        line, = ax.plot(video_length, rank_distance, styles[i], linewidth=LINEWIDTH,
                        color="black", marker=makers[i % len(makers)],
                        fillstyle=fillstyle[i % len(fillstyle)], markersize=12,
                        markeredgewidth=2)
        lines.append(line)
        labels.append(dataset)
    plt.legend(lines, replace_dataset_names(labels), bbox_to_anchor=(0, 0.8, 1.05, 0.2), loc="upper left",
               mode="expand", ncol=2, framealpha=0, fontsize=15)

    xticks = ax.get_xticks().tolist()
    xticks[0] = "%d " % xticks[0]
    xticks[1] = " %d" % xticks[1]
    ax.set_xticklabels(xticks)

    with PdfPages(out_prefix + "videolength_vs_rank_distance.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    resut_path = "/Users/cyliu/PycharmProjects/video_topk_plot/result_3"
    default_k = 50

    df_k = get_df_k(resut_path)
    df = df_k.loc[df_k["k"] == default_k]
    df.set_index(["dataset"])
    df_noscope = get_df_noscope(resut_path + "/" + "noscope.json")
    plot_overall(df.loc[df["k"] == default_k], df_noscope, "fig/")

    plot_quality_vs_k(df_k, "fig/")

    df_conf_thres = get_df_conf(resut_path)
    plot_quality_vs_confidence(df_conf_thres, "fig/")

    df_score_func = get_df_score_func(resut_path)
    plot_quality_vs_scoring_func(df_score_func, "fig/")

    df_window = get_df_window(resut_path)
    plot_quality_vs_window(df_window, "fig/")

    # df_video_length = get_df_vary_length(resut_path)
    # plot_quality_vs_videolength(df_video_length, "fig/")

    df_vs = get_df_vs(resut_path+"/"+"visualroad.json")
    plot_quality_vs_num_object(df_vs, "fig/")

    # df_window = get_df_window(resut_path)
    # plot_quality_vs_window(df_window, "fig/tmp_5/")

