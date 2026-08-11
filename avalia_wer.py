import os
import json
import torch
import jiwer
import whisper
from tqdm import tqdm

# CONFIG
AUDIO_ROOT = "audios_wer"
CORPUS_PATH = "corpus_wer.txt"
#CORPUS_PATH = "corpus_benchmark.txt"
OUTPUT_JSON = "wer_results.json"
OUTPUT_CSV = "wer_summary.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#WHISPER_MODEL_PATH = "/mnt/efs/wilmer_tts/checkpoints/small.pt"
WHISPER_MODEL_PATH = "/mnt/efs/wilmer_tts/checkpoints/large-v3.pt"

# LOAD CORPUS
def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# TEXT NORMALIZATION
transform = jiwer.Compose([ jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.Strip(), jiwer.RemoveMultipleSpaces() ])

def normalize_text(text):
    return transform(text)

# LOAD WHISPER
print("Loading Whisper...")
whisper_model = whisper.load_model(WHISPER_MODEL_PATH, device=DEVICE)

# TRANSCRIPTION
def transcribe_whisper(audio_file):
    result = whisper_model.transcribe(audio_file, language="pt")
    return result["text"]

# MAIN
corpus = load_corpus(CORPUS_PATH)

model_dirs = sorted([
    d for d in os.listdir(AUDIO_ROOT)
    if os.path.isdir(os.path.join(AUDIO_ROOT, d))
])

all_details = []
summary = []

print(f"\nFound {len(model_dirs)} models")

for model_name in model_dirs:

    print("\n" + "=" * 60)
    print(f"Evaluating: {model_name}")
    print("=" * 60)

    model_path = os.path.join(AUDIO_ROOT, model_name)

    #wav_files = sorted([
    #    f for f in os.listdir(model_path)
    #    if f.lower().endswith(".wav")
    #])

    wav_files = sorted([
        os.path.join(root, file)
        for root, _, files in os.walk(model_path)
        for file in files
        if file.lower().endswith(".wav")
    ])

    if len(wav_files) != len(corpus):

        print(
            f"WARNING: {model_name} has "
            f"{len(wav_files)} audios but corpus has "
            f"{len(corpus)} sentences."
        )

    wers = []
    cers = []

    for i in tqdm(range(min(len(corpus), len(wav_files)))):

        #audio_file = os.path.join(
        #    model_path,
        #    wav_files[i]
        #)
        audio_file = wav_files[i]

        reference = corpus[i]

        hypothesis = transcribe_whisper(audio_file)

        ref_norm = normalize_text(reference)
        hyp_norm = normalize_text(hypothesis)

        wer = jiwer.wer(ref_norm, hyp_norm)
        cer = jiwer.cer(ref_norm, hyp_norm)

        wers.append(wer)
        cers.append(cer)

        all_details.append({
            "model": model_name,
            "id": i + 1,
            "audio": os.path.basename(wav_files[i]),
            "reference": reference,
            "transcription": hypothesis,
            "wer": wer,
            "cer": cer
        })
    
    avg_wer = sum(wers) / len(wers)
    avg_cer = sum(cers) / len(cers)

    summary.append({
        "model": model_name,
        "wer": avg_wer,
        "cer": avg_cer
    })

    #print(f"\n{model_name}: WER = {avg_wer:.4f}")
    print(f"\n{model_name}:")
    print(f"WER = {avg_wer:.4f}")
    print(f"CER = {avg_cer:.4f}")

# SORT RANKING
summary = sorted(
    summary,
    key=lambda x: x["wer"]
)

# SAVE JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {
            "summary": summary,
            "details": all_details
        },
        f,
        indent=2,
        ensure_ascii=False
    )

# SAVE CSV
with open(OUTPUT_CSV, "w", encoding="utf-8") as f:

    f.write("model,wer,cer\n")

    for row in summary:
        f.write(
            f"{row['model']},{row['wer']:.6f},{row['cer']:.6f}\n"
        )

# PRINT RANKING
print("\n")
print("=" * 60)
print("FINAL WER RANKING")
print("=" * 60)

for pos, row in enumerate(summary, start=1):
    print(
        f"{pos:02d} | "
        f"{row['model']:<20} "
        f"WER = {row['wer']:.4f} | "
        f"CER = {row['cer']:.4f}"
    )

print("\nResults saved:")
print(f"  {OUTPUT_JSON}")
print(f"  {OUTPUT_CSV}")