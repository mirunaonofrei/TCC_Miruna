
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULT_DIR = os.path.join(BASE_DIR, "Resultados_NISQA")
OUT_DIR = os.path.join(BASE_DIR, "graficos_nisqa")

os.makedirs(OUT_DIR, exist_ok=True)

METRICAS = {
    "mos_pred": "MOS",
    "noi_pred": "NOI",
    "dis_pred": "DIS",
    "col_pred": "COL",
    "loud_pred": "LOUD"
}

CORES = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#8172B2",
    "#64B5CD"
]

# ============================================================
# CARREGAR OS DADOS
# ============================================================

dados = {}

for modelo in sorted(os.listdir(RESULT_DIR)):

    model_dir = os.path.join(RESULT_DIR, modelo)

    if not os.path.isdir(model_dir):
        continue

    for voz in sorted(os.listdir(model_dir)):

        voice_dir = os.path.join(model_dir, voz)

        if not os.path.isdir(voice_dir):
            continue

        csv_path = os.path.join(
            voice_dir, "NISQA_results.csv"
        )

        if not os.path.isfile(csv_path):
            print(f"[AVISO] CSV não encontrado: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        if not all(col in df.columns for col in METRICAS):
            print(f"[AVISO] Métricas ausentes em: {csv_path}")
            continue

        nome = f"{modelo} / {voz}"
        dados[nome] = df

if not dados:
    raise RuntimeError("Nenhum resultado NISQA encontrado.")

modelos = list(dados.keys())

# ============================================================
# BOXPLOT DAS CINCO MÉTRICAS
# ============================================================

label_kw = (
    "tick_labels"
    if tuple(map(int, matplotlib.__version__.split(".")[:2])) >= (3, 9)
    else "labels"
)

fig, axes = plt.subplots(1, 5, figsize=(20, 6))

for ax, (coluna, nome_metrica) in zip(axes, METRICAS.items()):

    valores = [
        dados[modelo][coluna].dropna()
        for modelo in modelos
    ]

    bp = ax.boxplot(
        valores,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(
            marker="D",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5
        ),
        medianprops=dict(color="black", linewidth=1.5),
        **{label_kw: modelos}
    )

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(CORES[i % len(CORES)])
        patch.set_alpha(0.7)

    ax.set_title(nome_metrica, fontsize=13)
    ax.set_ylim(0, 5)
    ax.set_ylabel("Pontuação" if nome_metrica == "MOS" else "")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=35)

fig.suptitle(
    "Distribuição das métricas NISQA por modelo TTS",
    fontsize=16
)

plt.tight_layout()

saida = os.path.join(
    OUT_DIR, "nisqa_boxplot.png"
)

plt.savefig(saida, dpi=300, bbox_inches="tight")
plt.close()

print(f"[OK] Boxplot salvo: {saida}")

# ============================================================
# GRÁFICO COMPARATIVO DAS MÉDIAS
# ============================================================

medias = pd.DataFrame({
    modelo: {
        nome: dados[modelo][coluna].mean()
        for coluna, nome in METRICAS.items()
    }
    for modelo in modelos
}).T

x = np.arange(len(modelos))
largura = 0.8 / len(METRICAS)

fig, ax = plt.subplots(figsize=(12, 6))

for i, metrica in enumerate(METRICAS.values()):

    ax.bar(
        x + (i - 2) * largura,
        medias[metrica],
        largura,
        label=metrica,
        color=CORES[i]
    )

ax.set_xticks(x)
ax.set_xticklabels(modelos, rotation=15, ha="right")
ax.set_ylabel("Pontuação média")
ax.set_ylim(0, 5)
ax.set_title("Comparação das médias das métricas NISQA")
ax.legend()
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

plt.tight_layout()

saida = os.path.join(
    OUT_DIR, "nisqa_medias.png"
)

plt.savefig(saida, dpi=300, bbox_inches="tight")
plt.close()

print(f"[OK] Gráfico de médias salvo: {saida}")
