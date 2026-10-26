
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

RESULT_DIR = os.path.join(BASE_DIR, "Resultados_UTMOS")
OUT_DIR = os.path.join(BASE_DIR, "graficos_utmos")

os.makedirs(OUT_DIR, exist_ok=True)

INPUT_JSON = os.path.join(RESULT_DIR, "utmos_results.json")

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

    grupos = defaultdict(list)

    for item in data["details"]:
        chave = f'{item["model"]}\n({item["speaker"]})'
        grupos[chave].append(float(item["utmos"]))

    return grupos


# =========================================================
# BOXPLOT UTMOS
# =========================================================

def plotar_boxplot(grupos):

    label_kw = (
        "tick_labels"
        if tuple(map(int, matplotlib.__version__.split(".")[:2])) >= (3, 9)
        else "labels"
    )

    labels = sorted(
        grupos,
        key=lambda k: -np.median(grupos[k])
    )

    valores = [grupos[k] for k in labels]

    fig, ax = plt.subplots(figsize=(11, 6))

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
        **{label_kw: labels}
    )

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(CORES[i % len(CORES)])
        patch.set_alpha(0.7)

    ax.set_ylabel("Pontuação UTMOS")
    ax.set_title("Distribuição dos escores UTMOS por modelo TTS")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()

    caminho_saida = os.path.join(
        OUT_DIR, "utmos_boxplot.png"
    )

    plt.savefig(
        caminho_saida,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[OK] Boxplot salvo: {caminho_saida}")


# =========================================================
# GRÁFICO DE BARRAS — MÉDIA UTMOS
# =========================================================

def plotar_barras(grupos):

    labels = sorted(
        grupos,
        key=lambda k: -np.mean(grupos[k])
    )

    medias = [np.mean(grupos[k]) for k in labels]
    desvios = [np.std(grupos[k], ddof=1) if len(grupos[k]) > 1 else 0 for k in labels]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        x,
        medias,
        yerr=desvios,
        capsize=5,
        color=[CORES[i % len(CORES)] for i in range(len(labels))],
        alpha=0.8
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")

    ax.set_ylabel("UTMOS médio")
    ax.set_title("Comparação das médias UTMOS por modelo TTS")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    plt.tight_layout()

    caminho_saida = os.path.join(
        OUT_DIR, "utmos_medias.png"
    )

    plt.savefig(
        caminho_saida,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[OK] Gráfico de médias salvo: {caminho_saida}")


# =========================================================
# FUNÇÃO PRINCIPAL
# =========================================================

def main():

    if not os.path.isfile(INPUT_JSON):
        print(f"[ERRO] Arquivo não encontrado: {INPUT_JSON}")
        return

    grupos = carregar_dados(INPUT_JSON)

    if not grupos:
        print("[AVISO] Nenhum resultado UTMOS encontrado.")
        return

    plotar_boxplot(grupos)
    plotar_barras(grupos)

    print(f"\nGráficos salvos em: {OUT_DIR}")


if __name__ == "__main__":
    main()
