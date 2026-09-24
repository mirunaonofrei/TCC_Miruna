import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONFIG - ajuste os caminhos conforme onde os CSVs foram salvos
GLOWTTS_CSV = "audios_utmos_2/glowtts-hifigan/rtf_results.csv"
GRADTTS_CSV = "rtf_results_gradtts.csv"

OUT_DIR = "graficos_rtf"
os.makedirs(OUT_DIR, exist_ok=True)


def load_glowtts(path):
    """Lê o CSV do GlowTTS (colunas: id, text, rtf), ignorando a linha de média no final."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["id"])
    df["rtf"] = pd.to_numeric(df["rtf"], errors="coerce")
    df = df.dropna(subset=["rtf"])
    return df


def load_gradtts(path):
    """Lê o CSV do Grad-TTS (colunas: voice, id, text, rtf_grad, rtf_hifi, rtf_big, rtf_grad_hifi, rtf_grad_big)."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["id"])
    for col in ["rtf_grad", "rtf_hifi", "rtf_big", "rtf_grad_hifi", "rtf_grad_big"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["rtf_grad_hifi", "rtf_grad_big"])
    return df


def main():

    glow_df = load_glowtts(GLOWTTS_CSV)
    grad_df = load_gradtts(GRADTTS_CSV)

    # Monta um dicionário: nome do modelo -> array de RTFs (sistema completo)
    modelos = {
        "GlowTTS + HiFi-GAN": glow_df["rtf"].values,
        "Grad-TTS + HiFi-GAN": grad_df["rtf_grad_hifi"].values,
        "Grad-TTS + BigVGAN": grad_df["rtf_grad_big"].values,
    }

    # =========================================================
    # GRÁFICO 1 — Barras com média ± desvio padrão
    # =========================================================
    nomes = list(modelos.keys())
    medias = [np.mean(v) for v in modelos.values()]
    desvios = [np.std(v) for v in modelos.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    cores = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(nomes, medias, yerr=desvios, capsize=6, color=cores)

    for bar, media in zip(bars, medias):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(desvios) * 0.15,
            f"{media:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="RTF = 1 (tempo real)")
    ax.set_ylabel("RTF (Real-Time Factor)")
    ax.set_title("RTF médio por modelo (sistema completo)")
    ax.legend()
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rtf_barras_media.png"), dpi=150)
    plt.close()

    # =========================================================
    # GRÁFICO 2 — Boxplot (distribuição do RTF por modelo)
    # =========================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    dados = list(modelos.values())
    box = ax.boxplot(dados, labels=nomes, patch_artist=True)

    for patch, cor in zip(box["boxes"], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.6)

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="RTF = 1 (tempo real)")
    ax.set_ylabel("RTF (Real-Time Factor)")
    ax.set_title("Distribuição do RTF por modelo")
    ax.legend()
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rtf_boxplot.png"), dpi=150)
    plt.close()

    # =========================================================
    # GRÁFICO 3 — RTF por frase (linha), útil para ver variação/outliers
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(glow_df["id"], glow_df["rtf"], marker="o", label="GlowTTS + HiFi-GAN", color=cores[0])
    ax.plot(grad_df["id"], grad_df["rtf_grad_hifi"], marker="o", label="Grad-TTS + HiFi-GAN", color=cores[1])
    ax.plot(grad_df["id"], grad_df["rtf_grad_big"], marker="o", label="Grad-TTS + BigVGAN", color=cores[2])

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="RTF = 1 (tempo real)")
    ax.set_xlabel("Frase (id)")
    ax.set_ylabel("RTF (Real-Time Factor)")
    ax.set_title("RTF por frase sintetizada")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rtf_por_frase.png"), dpi=150)
    plt.close()

    # =========================================================
    # GRÁFICO 4 — Decomposição do Grad-TTS (modelo acústico vs vocoder)
    # =========================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    componentes = {
        "Grad-TTS\n(acústico)": grad_df["rtf_grad"].values,
        "HiFi-GAN\n(vocoder)": grad_df["rtf_hifi"].values,
        "BigVGAN\n(vocoder)": grad_df["rtf_big"].values,
    }
    nomes_c = list(componentes.keys())
    medias_c = [np.mean(v) for v in componentes.values()]
    desvios_c = [np.std(v) for v in componentes.values()]

    ax.bar(nomes_c, medias_c, yerr=desvios_c, capsize=6, color=["#8172B2", "#DD8452", "#55A868"])
    ax.set_ylabel("RTF (Real-Time Factor)")
    ax.set_title("RTF médio por componente (Grad-TTS)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rtf_componentes_gradtts.png"), dpi=150)
    plt.close()

    # =========================================================
    # RESUMO NO TERMINAL
    # =========================================================
    print("=" * 60)
    print("RESUMO RTF")
    print("=" * 60)
    for nome, valores in modelos.items():
        print(f"{nome:25s} | média={np.mean(valores):.4f} | desvio={np.std(valores):.4f} | "
              f"min={np.min(valores):.4f} | max={np.max(valores):.4f}")
    print("=" * 60)
    print(f"Gráficos salvos em: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()