from phonemizer.backend import EspeakBackend
from symbols import BRPT_list


# Inicializa o fonemizador PT-BR com a mesma configuração
# utilizada no meldataset.py
phonemizer = EspeakBackend(
    language='pt-br',
    preserve_punctuation=True,
    with_stress=True
)


def verificar_arquivo(caminho):
    simbolos_encontrados = set()
    simbolos_ausentes = set()
    frases_com_problema = []

    total_frases = 0

    with open(caminho, "r", encoding="utf-8") as arquivo:
        for numero_linha, linha in enumerate(arquivo, start=1):

            linha = linha.strip()

            if not linha:
                continue

            partes = linha.split("|")

            if len(partes) != 3:
                print(
                    f"[AVISO] Linha {numero_linha} com formato inválido: "
                    f"{linha}"
                )
                continue

            wave_path, texto, speaker_id = partes

            # Fonemização PT-BR.
            # strip() remove espaços apenas das extremidades.
            fonemas = phonemizer.phonemize([texto])[0].strip()

            total_frases += 1

            ausentes_na_frase = set()

            for simbolo in fonemas:
                simbolos_encontrados.add(simbolo)

                if simbolo not in BRPT_list:
                    simbolos_ausentes.add(simbolo)
                    ausentes_na_frase.add(simbolo)

            if ausentes_na_frase:
                frases_com_problema.append({
                    "linha": numero_linha,
                    "texto": texto,
                    "fonemas": fonemas,
                    "ausentes": ausentes_na_frase
                })

    return {
        "total": total_frases,
        "encontrados": simbolos_encontrados,
        "ausentes": simbolos_ausentes,
        "problemas": frases_com_problema
    }


def mostrar_resultado(nome, resultado):
    print("\n" + "=" * 70)
    print(f"RESULTADO: {nome}")
    print("=" * 70)

    print(f"\nQuantidade de frases verificadas: {resultado['total']}")

    print("\nSímbolos produzidos pelo eSpeak:")
    for simbolo in sorted(resultado["encontrados"]):
        print(repr(simbolo), end=" ")

    print()

    if not resultado["ausentes"]:
        print("\nOK: Nenhum símbolo ausente no BRPT_list.")

    else:
        print("\nATENÇÃO: Símbolos ausentes no BRPT_list:")

        for simbolo in sorted(resultado["ausentes"]):
            print(repr(simbolo))

        print("\nFrases com símbolos ausentes:")

        for problema in resultado["problemas"]:
            print("\n" + "-" * 70)

            print(f"Linha: {problema['linha']}")
            print(f"Texto: {problema['texto']}")
            print(f"Fonemas: {repr(problema['fonemas'])}")

            print(
                "Símbolos ausentes:",
                [repr(s) for s in problema["ausentes"]]
            )


if __name__ == "__main__":

    print("Tamanho atual do BRPT_list:", len(BRPT_list))

    print("\nTokens especiais:")
    print("PAD   :", BRPT_list.index("_"))
    print("SOS   :", BRPT_list.index("*"))
    print("EOS   :", BRPT_list.index("&"))
    print("UNK   :", BRPT_list.index("@"))
    print("BLANK :", BRPT_list.index(" "))

    resultado_train = verificar_arquivo(
        "./Data/train_list.txt"
    )

    resultado_val = verificar_arquivo(
        "./Data/val_list.txt"
    )

    mostrar_resultado(
        "TRAIN",
        resultado_train
    )

    mostrar_resultado(
        "VALIDATION",
        resultado_val
    )

    # Resultado geral
    ausentes_total = (
        resultado_train["ausentes"]
        | resultado_val["ausentes"]
    )

    print("\n" + "=" * 70)
    print("RESULTADO GERAL")
    print("=" * 70)

    print(f"\nTamanho do BRPT_list: {len(BRPT_list)}")

    if not ausentes_total:
        print("\nOK: O BRPT_list cobre todos os símbolos do corpus.")
    else:
        print("\nATENÇÃO: Existem símbolos que não estão no BRPT_list:")

        for simbolo in sorted(ausentes_total):
            print(repr(simbolo))