import os
import subprocess
import pandas as pd

AUDIO_ROOT = "audios_utmos"
OUTPUT_ROOT = "results"
#MODEL_PATH = "NISQA/weights/nisqa_tts.tar"
MODEL_PATH = "NISQA/weights/nisqa.tar"

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
    print(f"Processando: {model_name}")
    print(f"{'='*60}")

    #output_dir = os.path.join(OUTPUT_ROOT, model_name)
    output_dir = os.path.join(OUTPUT_ROOT, model_name, speaker)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python",
        "NISQA/run_predict.py",
        "--mode", "predict_dir",
        "--pretrained_model", MODEL_PATH,
        "--data_dir", model_dir,
        "--output_dir", output_dir,
    ]

    subprocess.run(cmd, check=True)

    csv_file = None

    for f in os.listdir(output_dir):
        if f.endswith(".csv"):
            csv_file = os.path.join(output_dir, f)
            break

    if csv_file is None:
        print(f"CSV não encontrado para {model_name}")
        continue

    df = pd.read_csv(csv_file)

    mos_medio = df["mos_pred"].mean()

    resumo.append(
        {
            "modelo": model_name,
            "nisqa_tts": mos_medio
        }
    )

    print(f"NISQA-TTS médio: {mos_medio:.4f}")

# RESUMO FINAL
resumo = sorted(
    resumo,
    key=lambda x: x["nisqa_tts"],
    reverse=True
)

print("\n" + "=" * 60)
print("RANKING NISQA-TTS")
print("=" * 60)

for i, r in enumerate(resumo, start=1):
    print(
        f"{i:02d}. "
        f"{r['modelo']:15s} "
        f"{r['nisqa_tts']:.4f}"
    )

print("=" * 60)

if resumo:
    melhor = resumo[0]

    print(
        f"\nMelhor modelo: "
        f"{melhor['modelo']} "
        f"({melhor['nisqa_tts']:.4f})"
    )