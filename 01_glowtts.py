import os
import time
import torch
import csv

from TTS_glow.tts.utils.generic_utils import setup_model
from TTS_glow.tts.utils.synthesis import synthesis
from TTS_glow.tts.utils.text.symbols import make_symbols, phonemes, symbols
from TTS_glow.utils.audio import AudioProcessor
from TTS_glow.utils.io_edresson import load_config
from TTS_glow.vocoder.models import setup_generator


# CONFIG
MODEL_PATH = "/mnt/c/Users/rainh/TCC_Miruna/edresson-glowtts/edresson-glowtts/glowtts/best_model.pth.tar"
CONFIG_PATH = "/mnt/c/Users/rainh/TCC_Miruna/edresson-glowtts/edresson-glowtts/glowtts/config.json"
VOCODER_MODEL_PATH = "/mnt/c/Users/rainh/TCC_Miruna/edresson-glowtts/edresson-glowtts/hifigan/best_model.pth.tar"
VOCODER_CONFIG_PATH = "/mnt/c/Users/rainh/TCC_Miruna/edresson-glowtts/edresson-glowtts/hifigan/config.json"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

METRICA = "wer"  # rtf, wer, nisqa, utmos

CORPUS_PATH = os.path.join(
    BASE_DIR, "Corpus_Avaliacao", f"corpus_{METRICA}.txt"
)

OUT_PATH = os.path.join(
    BASE_DIR, f"audios_{METRICA}", "glowtts-hifigan"
)

RTF_DIR = os.path.join(BASE_DIR, "Resultados_RTF")

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOICES = ["edresson"]

# LOAD MODELS
def load_models():

    print("Loading GlowTTS...")

    C = load_config(CONFIG_PATH)
    ap = AudioProcessor(**C.audio)

    if "characters" in C.keys():
        global symbols, phonemes
        symbols, phonemes = make_symbols(**C.characters)

    speaker_embedding = None
    speaker_embedding_dim = None

    num_chars = len(phonemes) if C.use_phonemes else len(symbols)
    model = setup_model(num_chars, 1, C, speaker_embedding_dim)

    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint["model"])
    model.eval()
    model = model.to(DEVICE)

    model.length_scale = 0.65
    model.noise_scale = 0

    print("Loading HiFi-GAN...")

    vocoder_config = load_config(VOCODER_CONFIG_PATH)
    vocoder_model = setup_generator(vocoder_config)
    vocoder_model.load_state_dict(torch.load(VOCODER_MODEL_PATH, map_location="cpu")["model"])
    vocoder_model.eval()
    vocoder_model = vocoder_model.to(DEVICE)

    return model, vocoder_model, ap, C, speaker_embedding


# SYNTHESIS
def synthesize(text, model, vocoder_model, ap, C, speaker_embedding=None):

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    waveform, _, _, mel_postnet_spec, _, _ = synthesis(
        model,
        text,
        C,
        USE_CUDA,
        ap,
        None,
        None,
        False,
        C.enable_eos_bos_chars,
        True
    )


    device = next(vocoder_model.parameters()).device

    vocoder_input = torch.from_numpy(mel_postnet_spec).float()
    vocoder_input = vocoder_input.unsqueeze(0).transpose(1, 2)
    vocoder_input = vocoder_input.to(device)

    with torch.no_grad():
        waveform_voc = (
            vocoder_model(vocoder_input)
            .squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
        )

    duration = len(waveform_voc) / ap.sample_rate

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    synthesis_time = time.perf_counter() - start_time
    rtf = synthesis_time / duration

    return waveform_voc, rtf


# CORPUS
def load_corpus(path):

    with open(path, "r", encoding="utf-8") as f:
        sentences = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return sentences


# MAIN
if __name__ == "__main__":

    for voice in VOICES:
        os.makedirs(os.path.join(OUT_PATH, voice), exist_ok=True)

    model, vocoder_model, ap, C, speaker_embedding = load_models()
    sentences = load_corpus(CORPUS_PATH)
    rtfs = []
    results = []

    # Voz
    print("=" * 60)
    print(f"Voz: {VOICES[0]}")
    print("=" * 60)
    
    # Warm-up
    warm_text = "Esse teste é apenas um warm-up."
    print(f"WARM-UP: {warm_text}")
    print("=" * 60)
    wav0, rtf0 = synthesize(warm_text, model, vocoder_model, ap, C, speaker_embedding)
    
    for id, text in enumerate(sentences, start=1):
        wav, rtf = synthesize(text, model, vocoder_model, ap, C, speaker_embedding)
        speaker = VOICES[0]
        output_file = os.path.join(OUT_PATH, speaker, f"glowtts_{speaker}_{id:03d}.wav")
        ap.save_wav(wav, output_file)
        rtfs.append(rtf)
        results.append((id, text, rtf))

        print(f"[{id}/{len(sentences)}] {text}")
        print(f"RTF: {rtf:.4f}")

    media_rtf = sum(rtfs) / len(rtfs)
    print("\n" + "=" * 60)
    print(f"Arquivos gerados : {len(sentences)}")
    # print(f"RTF médio        : {sum(rtfs)/len(rtfs):.4f}")
    print(f"RTF médio        : {media_rtf:.4f}")
    print("=" * 60)

    if METRICA == "rtf":

        os.makedirs(RTF_DIR, exist_ok=True)

        rtf_csv_path = os.path.join(
            RTF_DIR, f"rtf_glowtts_hifigan_{VOICES[0]}.csv"
        )

        with open(rtf_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(["id", "text", "rtf"])

            for id, text, rtf in results:
                writer.writerow([id, text, f"{rtf:.6f}"])

            writer.writerow([])
            writer.writerow(["media", "", f"{media_rtf:.6f}"])

        print(f"RTFs salvos em: {rtf_csv_path}")