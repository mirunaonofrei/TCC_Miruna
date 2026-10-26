import piper
import wave
import time
import os
import csv

MODELS_DIR = "/mnt/c/Users/rainh/TCC_Miruna/piper"
"""
voices = {
            "cadu": "cadu/pt_BR-cadu-medium.onnx", 
            "edresson": "edresson/pt_BR-edresson-low.onnx",
            "faber": "faber/pt_BR-faber-medium.onnx", 
            "jeff": "jeff/pt_BR-jeff-medium.onnx"
        }
"""
voices = { "cadu": "cadu/pt_BR-cadu-medium.onnx" }
    

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

METRICA = "wer" # rtf, wer, nisqa, utmos

CORPUS_PATH = os.path.join(
    BASE_DIR, "Corpus_Avaliacao", f"corpus_{METRICA}.txt"
)

OUT_PATH = os.path.join(
    BASE_DIR, f"audios_{METRICA}", "piper"
)

RTF_DIR = os.path.join(BASE_DIR, "Resultados_RTF")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

results = {}
for voice_name, model_rel_path in voices.items():
    
    model_path = os.path.join(MODELS_DIR, model_rel_path)
    voice = piper.PiperVoice.load( model_path)#, use_cuda=True)
    out_dir = os.path.join(OUT_PATH, voice_name)
    os.makedirs(out_dir, exist_ok=True) 
    rtf_list = []
    csv_results = []
    
    # Voz
    print("=" * 60)
    print(f"Voz: {voice_name}")
    print("=" * 60)

    # Warm-up
    warm_text = "Esse teste é apenas um warm-up."
    print(f"WARM-UP: {warm_text}")
    print("=" * 60)
    _ = list(voice.synthesize(warm_text))
    
    for i, text in enumerate(texts, start=1):

        start = time.perf_counter()
        chunks = list(voice.synthesize(text))
        synth_time = time.perf_counter() - start

        pcm = b"".join(c.audio_int16_bytes for c in chunks)
        sample_rate = chunks[0].sample_rate
        duration = len(pcm) / (sample_rate * 2)
        rtf = synth_time / duration
        rtf_list.append(rtf)
        csv_results.append((i, text, rtf))

        # nome do arquivo gerado
        filename = f"piper_{voice_name}_{i:03d}.wav"
        path = os.path.join(out_dir, filename)

        with wave.open(os.path.join(out_dir, filename), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)

        print(f"[{i}/{len(texts)}] {text}")   
        print(f"RTF={rtf:.4f}")

    avg_rtf = sum(rtf_list) / len(rtf_list)
    results[voice_name] = avg_rtf
    
    if METRICA == "rtf":

        os.makedirs(RTF_DIR, exist_ok=True)

        rtf_csv_path = os.path.join(
            RTF_DIR, f"rtf_piper_{voice_name}.csv"
        )

        with open(rtf_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(["id", "text", "rtf"])

            for idx, text, rtf in csv_results:
                writer.writerow([idx, text, f"{rtf:.6f}"])

            writer.writerow([])
            writer.writerow(["media", "", f"{avg_rtf:.6f}"])

        print(f"RTFs salvos em: {rtf_csv_path}")

print("\n================ RESULTADOS ================")

for voice, rtf in results.items():
    print(f"{voice:10s}: {rtf:.4f}")

print("============================================")