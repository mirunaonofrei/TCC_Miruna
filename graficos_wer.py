
import os
import json
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURAÇÕES
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULT_DIR = os.path.join(BASE_DIR, "Resultados_WER")
OUT_DIR = os.path.join(BASE_DIR, "graficos_wer")

os.makedirs(OUT_DIR, exist_ok=True)

INPUT_JSON = os.path.join(RESULT_DIR, "wer_results.json")

CORES = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#8172B2",
    "#64B5CD"
]

# =========================================================
# CARREGAMENTO DOS RESULTADOS
# =========================================================

def carregar_dados(caminho_json):
    with open(caminho_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    wer_por_modelo = defaultdict(list)
    cer_por_modelo = defaultdict(list)

    for item in data["details"]:
        modelo = item["model"]

        wer_por_modelo[modelo].append(float(item["wer"]))
        cer_por_modelo[modelo].append(float(item["cer"]))

    return wer_por_modelo, cer_por_modelo


# =========================================================
# BOXPLOT COMPARATIVO WER E CER
# =========================================================

def plotar_boxplots(wer_por_modelo, cer_por_modelo):

    label_kw = (
        "tick_labels"
        if tuple(map(int, matplotlib.__version__.split(".")[:2]))
        >= (3, 9)
        else "labels"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    modelos = sorted(wer_por_modelo.keys())

    for ax, titulo, grupos, ylabel in [
        (axes[0], "Distribuição do WER por modelo TTS",
         wer_por_modelo, "WER"),
        (axes[1], "Distribuição do CER por modelo TTS",
         cer_por_modelo, "CER")
    ]:

        valores = [grupos[modelo] for modelo in modelos]

        bp = ax.boxplot(
            valores,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(
                marker="D",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=6
            ),
            medianprops=dict(
                color="black",
                linewidth=1.5
            ),
            flierprops=dict(
                marker="o",
                markersize=4,
                markerfacecolor="gray",
                alpha=0.6
            ),
            **{label_kw: modelos}
        )

        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(CORES[i % len(CORES)])
            patch.set_alpha(0.7)

        ax.set_title(titulo)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()

    caminho_saida = os.path.join(
        OUT_DIR, "wer_cer_boxplot.png"
    )

    plt.savefig(
        caminho_saida,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[OK] Gráfico salvo: {caminho_saida}")


# =========================================================
# GRÁFICO DE BARRAS — WER E CER MÉDIOS
# =========================================================

def plotar_barras(wer_por_modelo, cer_por_modelo):

    modelos = sorted(wer_por_modelo.keys())

    medias_wer = [
        np.mean(wer_por_modelo[m]) for m in modelos
    ]

    medias_cer = [
        np.mean(cer_por_modelo[m]) for m in modelos
    ]

    x = np.arange(len(modelos))
    largura = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        x - largura / 2,
        medias_wer,
        largura,
        label="WER",
        color="#4C72B0"
    )

    ax.bar(
        x + largura / 2,
        medias_cer,
        largura,
        label="CER",
        color="#DD8452"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        modelos,
        rotation=15,
        ha="right"
    )

    ax.set_ylabel("Taxa de erro")
    ax.set_title("WER e CER médios por modelo TTS")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()

    caminho_saida = os.path.join(
        OUT_DIR, "wer_cer_barras.png"
    )

    plt.savefig(
        caminho_saida,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[OK] Gráfico salvo: {caminho_saida}")


# =========================================================
# FUNÇÃO PRINCIPAL
# =========================================================

def main():

    if not os.path.isfile(INPUT_JSON):
        print(f"[ERRO] Arquivo não encontrado: {INPUT_JSON}")
        return

    wer_por_modelo, cer_por_modelo = carregar_dados(INPUT_JSON)

    if not wer_por_modelo:
        print("[AVISO] Nenhum resultado encontrado.")
        return

    plotar_boxplots(wer_por_modelo, cer_por_modelo)
    plotar_barras(wer_por_modelo, cer_por_modelo)

    print(f"\nGráficos salvos em: {OUT_DIR}")


if __name__ == "__main__":
    main()
