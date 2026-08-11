import os
import json
import csv
from statistics import mean, median, stdev
import utmosv2

# CONFIG
AUDIO_ROOT = "audios_utmos"
OUTPUT_JSON = "utmos_results.json"
OUTPUT_CSV = "utmos_summary.csv"

# LOAD MODEL
print("Loading UTMOS...")
model = utmosv2.create_model(pretrained=True)

# MAIN
summary = []
details = []

print("\n")
print("=" * 80)
print(f"{'Modelo':<20}{'Arquivos':>10}{'UTMOS':>12}")
print("=" * 80)

model_dirs = sorted([
    d for d in os.listdir(AUDIO_ROOT)
    if os.path.isdir(os.path.join(AUDIO_ROOT, d))
])

for model_name in model_dirs:

    model_path = os.path.join(AUDIO_ROOT, model_name)
    speaker_dirs = sorted([
        d for d in os.listdir(model_path)
        if os.path.isdir(os.path.join(model_path, d))
    ])

    if len(speaker_dirs) != 1:
        print(f"ERRO: {model_name} possui {len(speaker_dirs)} speakers.")
        continue

    speaker_name = speaker_dirs[0]
    speaker_path = os.path.join(model_path, speaker_name)

    print(f"Avaliando {model_name} ({speaker_name})...")

    # Lista os áudios
    wav_files = sorted([
        f for f in os.listdir(speaker_path)
        if f.lower().endswith(".wav")
    ])

    # Predição
    resultados = model.predict(input_dir=speaker_path)

    if len(resultados) != len(wav_files):
        print(
            f"WARNING: {model_name} possui "
            f"{len(wav_files)} wavs mas "
            f"UTMOS retornou {len(resultados)} scores."
        )

    scores = []

    for wav_file, resultado in zip(wav_files, resultados):

        score = resultado["predicted_mos"]

        scores.append(score)

        details.append({
            "model": model_name,
            "speaker": speaker_name,
            "audio": wav_file,
            "utmos": score
        })

    if len(scores) == 0:
        continue

    resumo = {
        "model": model_name,
        "speaker": speaker_name,
        "num_files": len(scores),
        "mean": mean(scores),
        "std": stdev(scores) if len(scores) > 1 else 0.0,
        "median": median(scores),
        "min": min(scores),
        "max": max(scores)
    }

    summary.append(resumo)

    print(
        f"{model_name:<20}"
        f"{len(scores):>10}"
        f"{resumo['mean']:>12.3f}"
    )

# SORT
summary = sorted(
    summary,
    key=lambda x: x["mean"],
    reverse=True
)

# SAVE JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:

    json.dump(
        {
            "summary": summary,
            "details": details
        },
        f,
        indent=2,
        ensure_ascii=False
    )

# SAVE CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "model",
        "speaker",
        "num_files",
        "utmos_mean",
        "utmos_std",
        "utmos_median",
        "utmos_min",
        "utmos_max"
    ])

    for r in summary:

        writer.writerow([
            r["model"],
            r["speaker"],
            r["num_files"],
            f"{r['mean']:.4f}",
            f"{r['std']:.4f}",
            f"{r['median']:.4f}",
            f"{r['min']:.4f}",
            f"{r['max']:.4f}"
        ])

# RANKING
print("\n")
print("=" * 80)
print("FINAL UTMOS RANKING")
print("=" * 80)

for pos, r in enumerate(summary, start=1):

    print(
        f"{pos:02d} | "
        f"{r['model']:<20}"
        f"{r['mean']:.3f} ± {r['std']:.3f}"
    )

print("\nArquivos salvos:")
print(f"  {OUTPUT_JSON}")
print(f"  {OUTPUT_CSV}")