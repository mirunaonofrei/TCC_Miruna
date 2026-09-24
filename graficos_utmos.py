"""
Gera um boxplot dos escores UTMOS por modelo TTS, a partir do
arquivo utmos_results.json (que contém a lista "details" com um
escore por arquivo de áudio).

Uso:
    python plot_utmos_boxplot.py utmos_results.json
"""

import json
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


def carregar_dados(caminho_json: str):
    with open(caminho_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # agrupa os escores UTMOS por (modelo, speaker)
    grupos = defaultdict(list)
    for item in data["details"]:
        chave = f'{item["model"]}\n({item["speaker"]})'
        grupos[chave].append(item["utmos"])

    return grupos


def plotar_boxplot(grupos: dict, caminho_saida: str = "utmos_boxplot.png"):
    # ordena os grupos pela mediana (do maior para o menor), fica mais legível
    labels = sorted(grupos, key=lambda k: -sorted(grupos[k])[len(grupos[k]) // 2])
    valores = [grupos[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 6))

    import matplotlib
    label_kw = "tick_labels" if tuple(map(int, matplotlib.__version__.split(".")[:2])) >= (3, 9) else "labels"

    bp = ax.boxplot(
        valores,
        patch_artist=True,
        **{label_kw: labels},
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6),
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, markerfacecolor="gray", alpha=0.6),
    )

    cores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for patch, cor in zip(bp["boxes"], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.7)

    ax.set_ylabel("UTMOS")
    ax.set_title("Distribuição dos escores UTMOS por modelo")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # anota o número de amostras (n) em cada caixa
    for i, k in enumerate(labels, start=1):
        n = len(grupos[k])
        ax.annotate(
            f"n={n}",
            xy=(i, min(grupos[k])),
            xytext=(0, -18),
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
    caminho_json = sys.argv[1] if len(sys.argv) > 1 else "utmos_results.json"
    grupos = carregar_dados(caminho_json)
    plotar_boxplot(grupos)