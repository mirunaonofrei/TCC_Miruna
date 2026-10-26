
import os
import wave
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURAÇÕES
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RTF_DIR = os.path.join(BASE_DIR, "Resultados_RTF")
AUDIOS_DIR = os.path.join(BASE_DIR, "audios_rtf")
OUT_DIR = os.path.join(BASE_DIR, "graficos_rtf")

os.makedirs(OUT_DIR, exist_ok=True)

# CSVs padronizados
ARQUIVOS = {
    "GlowTTS + HiFi-GAN": "rtf_glowtts_hifigan_edresson.csv",
    "GradTTS + HiFi-GAN": "rtf_gradtts_hifigan_polly-camila.csv",
    "GradTTS + BigVGAN": "rtf_gradtts_bigvgan_polly-camila.csv",
    "Piper (Cadu)": "rtf_piper_cadu.csv",
    "Azure (Antonio)": "rtf_azure_Antonio.csv",
}

# Pastas e nomes dos arquivos WAV
CONFIG_AUDIO = {
    "GlowTTS + HiFi-GAN": {
        "pasta": "glowtts-hifigan/edresson",
        "prefixo": "glowtts_edresson",
    },
    "GradTTS + HiFi-GAN": {
        "pasta": "gradtts-hifigan/polly-camila",
        "prefixo": "gradtts_hifigan_polly-camila",
    },
    "GradTTS + BigVGAN": {
        "pasta": "gradtts-bigvgan/polly-camila",
        "prefixo": "gradtts_bigvgan_polly-camila",
    },
    "Piper (Cadu)": {
        "pasta": "piper/cadu",
        "prefixo": "piper_cadu",
    },
    "Azure (Antonio)": {
        "pasta": "azure/Antonio",
        "prefixo": "azure_Antonio",
    },
}

GRADTTS_DETALHADO = os.path.join(
    RTF_DIR, "rtf_gradtts_detalhado.csv"
)

CORES = {
    "GlowTTS + HiFi-GAN": "#4C72B0",
    "GradTTS + HiFi-GAN": "#DD8452",
    "GradTTS + BigVGAN": "#55A868",
    "Piper (Cadu)": "#8172B2",
    "Azure (Antonio)": "#64B5CD",
}

# =========================================================
# LEITURA DOS RESULTADOS
# =========================================================

def carregar_rtf(path):
    """
    Lê um CSV com colunas:
    id, text, rtf

    Remove linhas vazias e a linha de média.
    """
    df = pd.read_csv(path)

    colunas = ["id", "text", "rtf"]

    for coluna in colunas:
        if coluna not in df.columns:
            raise ValueError(
                f"Coluna '{coluna}' não encontrada em {path}"
            )

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["rtf"] = pd.to_numeric(df["rtf"], errors="coerce")

    df = df.dropna(subset=["id", "rtf"]).copy()
    df = df[np.isfinite(df["rtf"]) & (df["rtf"] >= 0)]

    df["id"] = df["id"].astype(int)
    df["text"] = df["text"].fillna("").astype(str)

    return df.sort_values("id").reset_index(drop=True)


def carregar_gradtts_detalhado(path):
    """
    Lê o CSV detalhado do GradTTS.
    """
    if not os.path.isfile(path):
        return None

    df = pd.read_csv(path)

    colunas = [
        "rtf_grad",
        "rtf_hifi",
        "rtf_big",
        "rtf_grad_hifi",
        "rtf_grad_big",
    ]

    if not all(col in df.columns for col in colunas):
        print("[AVISO] Colunas incompletas no CSV do GradTTS.")
        return None

    for coluna in colunas:
        df[coluna] = pd.to_numeric(
            df[coluna], errors="coerce"
        )

    df = df.dropna(subset=colunas).copy()

    for coluna in colunas:
        df = df[np.isfinite(df[coluna])]

    return df


# =========================================================
# DURAÇÃO DOS ARQUIVOS WAV
# =========================================================

def obter_duracao_wav(caminho):
    """
    Retorna a duração de um WAV em segundos.
    """
    with wave.open(caminho, "rb") as audio:
        frames = audio.getnframes()
        sample_rate = audio.getframerate()

    if sample_rate <= 0:
        raise ValueError("Taxa de amostragem inválida.")

    return frames / sample_rate


def carregar_duracoes(nome_modelo, df):
    """
    Associa o RTF de cada frase à duração
    do respectivo arquivo WAV.
    """
    config = CONFIG_AUDIO[nome_modelo]

    pasta = os.path.join(
        AUDIOS_DIR, config["pasta"]
    )

    resultados = []

    for _, linha in df.iterrows():
        idx = int(linha["id"])

        arquivo = (
            f'{config["prefixo"]}_{idx:03d}.wav'
        )

        caminho = os.path.join(pasta, arquivo)

        if not os.path.isfile(caminho):
            print(f"[AVISO] WAV não encontrado: {caminho}")
            continue

        try:
            duracao = obter_duracao_wav(caminho)
        except (wave.Error, OSError, ValueError) as erro:
            print(f"[AVISO] Erro ao ler {arquivo}: {erro}")
            continue

        if duracao <= 0:
            continue

        texto = str(linha["text"])

        resultados.append({
            "id": idx,
            "text": texto,
            "rtf": float(linha["rtf"]),
            "duracao": duracao,
            "num_palavras": len(texto.split()),
        })

    return pd.DataFrame(
        resultados,
        columns=[
            "id", "text", "rtf",
            "duracao", "num_palavras"
        ]
    )


# =========================================================
# FUNÇÕES AUXILIARES DOS GRÁFICOS
# =========================================================

def adicionar_linha_tempo_real(ax):
    ax.axhline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1,
        label="RTF = 1 (tempo real)"
    )


def salvar_grafico(nome_arquivo):
    caminho = os.path.join(OUT_DIR, nome_arquivo)

    plt.tight_layout()
    plt.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Gráfico salvo: {nome_arquivo}")


# =========================================================
# GRÁFICO 1 — RTF MÉDIO ± DESVIO PADRÃO
# =========================================================

def grafico_barras(modelos):
    nomes = list(modelos.keys())

    medias = [
        np.mean(modelos[nome]) for nome in nomes
    ]

    desvios = [
        np.std(modelos[nome]) for nome in nomes
    ]

    cores = [CORES[nome] for nome in nomes]

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(nomes))

    bars = ax.bar(
        x,
        medias,
        yerr=desvios,
        capsize=6,
        color=cores,
        alpha=0.85
    )

    deslocamento = max(
        max(medias),
        max(desvios),
        0.01
    ) * 0.03

    for bar, media, desvio in zip(bars, medias, desvios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            media + desvio + deslocamento,
            f"{media:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    adicionar_linha_tempo_real(ax)

    ax.set_xticks(x)
    ax.set_xticklabels(nomes, rotation=15, ha="right")

    ax.set_ylabel("RTF (Real-Time Factor)")
    ax.set_title("RTF médio por modelo TTS")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()

    salvar_grafico("rtf_barras_media.png")


# =========================================================
# GRÁFICO 2 — BOXPLOT
# =========================================================

def grafico_boxplot(modelos):
    nomes = list(modelos.keys())
    dados = [modelos[nome] for nome in nomes]

    fig, ax = plt.subplots(figsize=(11, 6))

    box = ax.boxplot(
        dados,
        tick_labels=nomes,
        patch_artist=True,
        showmeans=True
    )

    for patch, nome in zip(box["boxes"], nomes):
        patch.set_facecolor(CORES[nome])
        patch.set_alpha(0.65)

    adicionar_linha_tempo_real(ax)

    ax.set_ylabel("RTF (Real-Time Factor)")
    ax.set_title("Distribuição do RTF por modelo TTS")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()

    plt.xticks(rotation=15, ha="right")

    salvar_grafico("rtf_boxplot.png")


# =========================================================
# GRÁFICO 3 — RTF POR FRASE
# =========================================================

def grafico_por_frase(dados_frases):
    fig, ax = plt.subplots(figsize=(12, 6))

    for nome, df in dados_frases.items():
        ax.plot(
            df["id"],
            df["rtf"],
            marker="o",
            markersize=2.5,
            linewidth=1,
            color=CORES[nome],
            label=nome
        )

    adicionar_linha_tempo_real(ax)

    ax.set_xlabel("Identificador da frase")
    ax.set_ylabel("RTF (Real-Time Factor)")
    ax.set_title("RTF por frase sintetizada")
    ax.grid(alpha=0.2)
    ax.legend()

    salvar_grafico("rtf_por_frase.png")


# =========================================================
# GRÁFICO 4 — COMPONENTES DO GRADTTS
# =========================================================

def grafico_componentes_gradtts(grad_df):
    componentes = {
        "GradTTS\n(acústico)": grad_df["rtf_grad"].values,
        "HiFi-GAN\n(vocoder)": grad_df["rtf_hifi"].values,
        "BigVGAN\n(vocoder)": grad_df["rtf_big"].values,
    }

    nomes = list(componentes.keys())

    medias = [
        np.mean(v) for v in componentes.values()
    ]

    desvios = [
        np.std(v) for v in componentes.values()
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        nomes,
        medias,
        yerr=desvios,
        capsize=6,
        color=["#8172B2", "#DD8452", "#55A868"]
    )

    ax.set_ylabel("RTF (Real-Time Factor)")
    ax.set_title("RTF médio por componente do GradTTS")
    ax.grid(axis="y", alpha=0.2)

    salvar_grafico("rtf_componentes_gradtts.png")


# =========================================================
# GRÁFICO 5 — RTF X DURAÇÃO DO ÁUDIO
# =========================================================

def grafico_rtf_vs_duracao(dados_duracao):
    """
    Eixo X: duração do áudio em segundos.
    Eixo Y: RTF.

    Cada ponto representa uma frase sintetizada.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for nome, df in dados_duracao.items():
        if df.empty:
            continue

        ax.scatter(
            df["duracao"],
            df["rtf"],
            color=CORES[nome],
            label=nome,
            alpha=0.65,
            s=35
        )

    adicionar_linha_tempo_real(ax)

    ax.set_xlabel("Duração do áudio sintetizado (s)")
    ax.set_ylabel("RTF (Real-Time Factor)")
    ax.set_title("Relação entre duração do áudio e RTF")
    ax.grid(alpha=0.2)
    ax.legend()

    salvar_grafico("rtf_vs_duracao_audio.png")


# =========================================================
# GRÁFICO 6 — RTF X COMPRIMENTO DA FRASE
# =========================================================

def grafico_rtf_vs_palavras(dados_duracao):
    """
    Eixo X: número de palavras da frase.
    Eixo Y: RTF.

    Permite avaliar a influência do comprimento
    textual sobre o desempenho da síntese.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for nome, df in dados_duracao.items():
        if df.empty:
            continue

        ax.scatter(
            df["num_palavras"],
            df["rtf"],
            color=CORES[nome],
            label=nome,
            alpha=0.65,
            s=35
        )

    adicionar_linha_tempo_real(ax)

    ax.set_xlabel("Quantidade de palavras da frase")
    ax.set_ylabel("RTF (Real-Time Factor)")
    ax.set_title("Relação entre comprimento da frase e RTF")
    ax.grid(alpha=0.2)
    ax.legend()

    salvar_grafico("rtf_vs_numero_palavras.png")


# =========================================================
# FUNÇÃO PRINCIPAL
# =========================================================

def main():
    modelos = {}
    dados_frases = {}
    dados_duracao = {}

    print("\nCarregando resultados RTF...\n")

    # Carrega os CSVs disponíveis
    for nome, arquivo in ARQUIVOS.items():
        caminho = os.path.join(RTF_DIR, arquivo)

        if not os.path.isfile(caminho):
            print(f"[AVISO] CSV não encontrado: {arquivo}")
            continue

        try:
            df = carregar_rtf(caminho)
        except (OSError, ValueError, pd.errors.ParserError) as erro:
            print(f"[AVISO] Erro no CSV {arquivo}: {erro}")
            continue

        if df.empty:
            print(f"[AVISO] CSV sem resultados: {arquivo}")
            continue

        modelos[nome] = df["rtf"].to_numpy()
        dados_frases[nome] = df

        print(f"[OK] {nome}: {len(df)} frases")

    if not modelos:
        print("\nNenhum resultado RTF encontrado.")
        return

    # Carrega as durações dos áudios
    print("\nCarregando durações dos arquivos WAV...\n")

    for nome, df in dados_frases.items():
        df_duracao = carregar_duracoes(nome, df)

        if df_duracao.empty:
            print(f"[AVISO] Sem WAVs correspondentes para {nome}")
            continue

        dados_duracao[nome] = df_duracao

        print(
            f"[OK] {nome}: "
            f"{len(df_duracao)} áudios encontrados"
        )

    # Gera os gráficos principais
    grafico_barras(modelos)
    grafico_boxplot(modelos)
    grafico_por_frase(dados_frases)

    # Gráfico dos componentes do GradTTS
    grad_df = carregar_gradtts_detalhado(
        GRADTTS_DETALHADO
    )

    if grad_df is not None and not grad_df.empty:
        grafico_componentes_gradtts(grad_df)
    else:
        print(
            "[AVISO] CSV detalhado do GradTTS indisponível."
        )

    # Gráficos que dependem dos arquivos WAV
    if dados_duracao:
        grafico_rtf_vs_duracao(dados_duracao)
        grafico_rtf_vs_palavras(dados_duracao)
    else:
        print(
            "[AVISO] Gráficos de duração e comprimento "
            "não gerados: nenhum WAV encontrado."
        )

    # Resumo no terminal
    print("\n" + "=" * 85)
    print("RESUMO RTF")
    print("=" * 85)

    for nome, valores in modelos.items():
        print(
            f"{nome:28s} | "
            f"média={np.mean(valores):.4f} | "
            f"desvio={np.std(valores):.4f} | "
            f"mín={np.min(valores):.4f} | "
            f"máx={np.max(valores):.4f}"
        )

    print("=" * 85)
    print(f"\nGráficos salvos em: {OUT_DIR}")


if __name__ == "__main__":
    main()
