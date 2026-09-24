"""
Gera boxplots de WER e CER por modelo TTS, a partir do arquivo
wer_cer_results.json (que contém a lista "details" com um WER/CER
por arquivo de áudio).

Uso:
    python plot_wer_cer_boxplot.py wer_cer_results.json
"""

import json
import sys
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt


def carregar_dados(caminho_json: str):
    with open(caminho_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    wer_por_modelo = defaultdict(list)
    cer_por_modelo = defaultdict(list)
    for item in data["details"]:
        wer_por_modelo[item["model"]].append(item["wer"])
        cer_por_modelo[item["model"]].append(item["cer"])

    return wer_por_modelo, cer_por_modelo


def plotar_boxplots(wer_por_modelo, cer_por_modelo, caminho_saida="wer_cer_boxplot.png"):
    label_kw = "tick_labels" if tuple(map(int, matplotlib.__version__.split(".")[:2])) >= (3, 9) else "labels"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    cores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    for ax, (titulo, grupos, ylabel) in zip(
        axes,
        [
            ("WER por modelo", wer_por_modelo, "WER"),
            ("CER por modelo", cer_por_modelo, "CER"),
        ],
    ):
        labels = sorted(grupos.keys())
        valores = [grupos[k] for k in labels]

        bp = ax.boxplot(
            valores,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6),
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker="o", markersize=4, markerfacecolor="gray", alpha=0.6),
            **{label_kw: labels},
        )

        for patch, cor in zip(bp["boxes"], cores):
            patch.set_facecolor(cor)
            patch.set_alpha(0.7)

        ax.set_title(titulo)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=15)

        for i, k in enumerate(labels, start=1):
            n = len(grupos[k])
            ax.annotate(
                f"n={n}",
                xy=(i, 0),
                xytext=(0, -22),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="dimgray",
            )

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150)
    print(f"Boxplot salvo em: {caminho_saida}")
    plt.show()


if __name__ == "__main__":
    caminho_json = sys.argv[1] if len(sys.argv) > 1 else "wer_results.json"
    wer_por_modelo, cer_por_modelo = carregar_dados(caminho_json)
    plotar_boxplots(wer_por_modelo, cer_por_modelo)