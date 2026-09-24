import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# ARQUIVOS
# ============================================================

arquivos = {
     "GlowTTS + HiFi-GAN": "results/glowtts-hifigan/edresson/NISQA_results.csv",
    "GradTTS + BigVGAN": "results/gradtts-bigvgan/polly-camila/NISQA_results.csv",
    "GradTTS + HiFi-GAN": "results/gradtts-hifigan/polly-camila/NISQA_results.csv"
}

# ============================================================
# MÉTRICAS DO NISQA
# ============================================================

metricas = {
    "mos_pred": "MOS",
    "noi_pred": "NOI",
    "dis_pred": "DIS",
    "col_pred": "COL",
    "loud_pred": "LOUD"
}

# ============================================================
# CARREGAR OS DADOS
# ============================================================

dados = {}

for modelo, arquivo in arquivos.items():

    df = pd.read_csv(arquivo)

    # Corrige o cabeçalho caso o primeiro campo esteja
    # como "deg" em vez de "filename"
    if df.columns[0] == "deg":
        df.columns = [
            "filename",
            "mos_pred",
            "noi_pred",
            "dis_pred",
            "col_pred",
            "loud_pred",
            "model"
        ]

    dados[modelo] = df


# ============================================================
# BOX PLOT
# ============================================================

fig, axes = plt.subplots(
    1,
    5,
    figsize=(18, 6)
)

for ax, (coluna, nome_metrica) in zip(axes, metricas.items()):

    valores = [
        dados["GlowTTS + HiFi-GAN"][coluna],
        dados["GradTTS + BigVGAN"][coluna],
        dados["GradTTS + HiFi-GAN"][coluna]
    ]

    ax.boxplot(
        valores,
        labels=[
            "GlowTTS\n+ HiFi-GAN",
            "GradTTS\n+ BigVGAN",
            "GradTTS\n+ HiFi-GAN"
        ]
    )

    ax.set_title(nome_metrica, fontsize=13)

    ax.set_ylim(0, 5)

    ax.set_ylabel("Pontuação" if nome_metrica == "MOS" else "")

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )


# ============================================================
# TÍTULO
# ============================================================

fig.suptitle(
    "Distribuição das métricas NISQA nos modelos TTS",
    fontsize=16
)

plt.tight_layout()

# ============================================================
# SALVAR
# ============================================================

plt.savefig(
    "boxplot_nisqa_modelos_tts.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()