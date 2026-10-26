import os
import subprocess
import pandas as pd
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_ROOT = os.path.join(BASE_DIR, "audios_nisqa")

OUTPUT_ROOT = os.path.join(BASE_DIR, "Resultados_NISQA")

MODEL_PATH = os.path.join(
    BASE_DIR, "NISQA", "weights", "nisqa.tar"
)

NISQA_SCRIPT = os.path.join(
    BASE_DIR, "NISQA", "run_predict.py"
)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

resumo = []

for model_name in sorted(os.listdir(AUDIO_ROOT)):

    model_root = os.path.join(AUDIO_ROOT, model_name)

    if not os.path.isdir(model_root):
        continue

    for speaker in sorted(os.listdir(model_root)):

        model_dir = os.path.join(model_root, speaker)

        if not os.path.isdir(model_dir):
            continue

        print(f"\n{'='*60}")
        print(f"Processando: {model_name} / {speaker}")
        print(f"{'='*60}")

        output_dir = os.path.join(
            OUTPUT_ROOT, model_name, speaker
        )
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            sys.executable,
            NISQA_SCRIPT,
            "--mode", "predict_dir",
            "--pretrained_model", MODEL_PATH,
            "--data_dir", model_dir,
            "--output_dir", output_dir,
        ]

        subprocess.run(cmd, check=True, cwd=BASE_DIR)

        csv_file = None

        for f in os.listdir(output_dir):
            if f.endswith(".csv"):
                csv_file = os.path.join(output_dir, f)
                break

        if csv_file is None:
            print(f"CSV não encontrado para {model_name}/{speaker}")
            continue

        df = pd.read_csv(csv_file)

        if "mos_pred" not in df.columns:
            print(f"Coluna mos_pred ausente em {csv_file}")
            continue

        mos_medio = df["mos_pred"].mean()

        resumo.append({
            "modelo": model_name,
            "voz": speaker,
            "nisqa": mos_medio
        })

        print(f"NISQA médio: {mos_medio:.4f}")

# =========================================================
# RESUMO FINAL
# =========================================================

resumo = sorted(
    resumo,
    key=lambda x: x["nisqa"],
    reverse=True
)

print("\n" + "=" * 60)
print("RANKING NISQA")
print("=" * 60)

for i, r in enumerate(resumo, start=1):
    print(
        f"{i:02d}. "
        f"{r['modelo']:<22} "
        f"{r['voz']:<18} "
        f"NISQA = {r['nisqa']:.4f}"
    )

print("=" * 60)

if resumo:
    melhor = resumo[0]

    print(
        f"\nMelhor modelo: "
        f"{melhor['modelo']} / {melhor['voz']} "
        f"({melhor['nisqa']:.4f})"
    )
    
# =========================================================
# SALVAR RESUMO
# =========================================================

if resumo:
    df_resumo = pd.DataFrame(resumo)

    output_csv = os.path.join(
        OUTPUT_ROOT, "nisqa_summary.csv"
    )

    df_resumo.to_csv(
        output_csv,
        index=False,
        encoding="utf-8"
    )

    print(f"\nResumo salvo em: {output_csv}")