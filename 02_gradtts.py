import os
import sys
import time
import json
import numpy as np
from scipy.io.wavfile import write
import torch

# PATHS BASE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "prototype-final-gradtts")
CKPT_DIR = os.path.join("/mnt/c/Users/rainh/TCC_Miruna/prototype-final-gradtts", "ckpts")

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "hifigan"))
sys.path.append(os.path.join(PROJECT_ROOT, "bigvgan"))

import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols_pt_br import symbols
from utils import intersperse

from hifigan.env import AttrDict as HiFiAttrDict
from hifigan.models import Generator as HiFiGAN
from bigvgan.env import AttrDict as BigAttrDict
from bigvgan.models import BigVGAN

# CONFIG FIXA
#CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "ckpts/gradtts-old/grad_1500.pt")
#HIFIGAN_CHECKPT = os.path.join(PROJECT_ROOT, "ckpt/hifigan-old/hifigan.pt")
VOICES = ["polly-camila"]

#CORPUS_PATH = os.path.join(BASE_DIR, "corpus_benchmark.txt")
#CORPUS_PATH = os.path.join(BASE_DIR, "corpus_rtf.txt")
#CORPUS_PATH = os.path.join(BASE_DIR, "corpus_wer.txt")
CORPUS_PATH = os.path.join(BASE_DIR, "corpus_utmos.txt")

CHECKPOINT_PATH = os.path.join(CKPT_DIR, "gradtts/grad_799.pt")
HIFIGAN_CONFIG = os.path.join(CKPT_DIR, "hifigan/hifigan-config.json")
HIFIGAN_CHECKPT = os.path.join(CKPT_DIR, "hifigan/hifigan-awspolly-v4-1480k")
BIGVGAN_CONFIG = os.path.join(CKPT_DIR, "bigvgan/config.json")
BIGVGAN_CHECKPT = os.path.join(CKPT_DIR, "bigvgan/g_01670000")

#OUTPUT_DIR1 = os.path.join(BASE_DIR, "audios_rtf", "gradtts-hifigan")
#OUTPUT_DIR2 = os.path.join(BASE_DIR, "audios_rtf", "gradtts-bigvgan")
#OUTPUT_DIR1 = os.path.join(BASE_DIR, "audios_wer", "gradtts-hifigan")
#OUTPUT_DIR2 = os.path.join(BASE_DIR, "audios_wer", "gradtts-bigvgan")
OUTPUT_DIR1 = os.path.join(BASE_DIR, "audios_utmos", "gradtts-hifigan")
OUTPUT_DIR2 = os.path.join(BASE_DIR, "audios_utmos", "gradtts-bigvgan")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTEPS = 10
SAMPLE_RATE = 22050

os.makedirs(OUTPUT_DIR1, exist_ok=True)
os.makedirs(OUTPUT_DIR2, exist_ok=True)
# CARREGA MODELOS
print("Carregando Grad-TTS...")

generator = GradTTS( len(symbols) + 1, params.n_spks, params.spk_emb_dim, params.n_enc_channels, params.filter_channels,
                     params.filter_channels_dp, params.n_heads, params.n_enc_layers, params.enc_kernel, params.enc_dropout,
                     params.window_size, params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale,
                    )
generator.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=lambda loc, storage: loc))
_ = generator.cuda().eval()


print(f"Grad-TTS OK ({generator.nparams} params)")


print("Carregando HiFiGAN...")
with open(HIFIGAN_CONFIG) as f:
    h_hifi = HiFiAttrDict(json.load(f))

hifigan = HiFiGAN(h_hifi)
hifigan.load_state_dict(torch.load(HIFIGAN_CHECKPT)["generator"])
hifigan.cuda().eval()
hifigan.remove_weight_norm()

print("Carregando BigVGAN...")
with open(BIGVGAN_CONFIG) as f:
    h_big = BigAttrDict(json.load(f))

bigvgan = BigVGAN(h_big)
bigvgan.load_state_dict(torch.load(BIGVGAN_CHECKPT)["generator"])
bigvgan.cuda().eval()
bigvgan.remove_weight_norm()

# CORPUS
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    texts = [t.strip() for t in f if t.strip()]
cmu = cmudict.CMUDict(os.path.join(PROJECT_ROOT, "resources/cmu_dictionary"))

# MÉTRICAS
rtf_grad_list = []
rtf_hifi_list = []
rtf_big_list = []
rtf_grad_hifi_list = []
rtf_grad_big_list = []

def normalize(audio):
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (0.98 / peak)
    return (audio * 32767).astype(np.int16)

# INFERÊNCIA
for voice in VOICES:

    out_dir_hifi = os.path.join(OUTPUT_DIR1, voice)
    out_dir_big = os.path.join(OUTPUT_DIR2, voice)
    os.makedirs(out_dir_hifi, exist_ok=True)
    os.makedirs(out_dir_big, exist_ok=True)
    
    # Voz
    print("=" * 60)
    print(f"Voz: {voice}")
    print("=" * 60)
    
    with torch.no_grad():

        # Warm-up
        warm_text = "Esse teste é apenas um warm-up."
        print(f"WARM-UP: {warm_text}")
        print("=" * 60)
        x = torch.LongTensor(intersperse(text_to_sequence(warm_text, dictionary=cmu), len(symbols))).unsqueeze(0).to(DEVICE)
        x_lengths = torch.LongTensor([x.shape[-1]]).to(DEVICE)
        _, y_dec, _ = generator.forward( x, x_lengths, n_timesteps=TIMESTEPS, temperature=1.5, stoc=False, spk=None, length_scale=0.91)
        audio_hifi = (hifigan(y_dec).cpu().squeeze().clamp(-1,1).numpy())
        audio_big  = (bigvgan(y_dec).cpu().squeeze().clamp(-1,1).numpy())

        for i, text in enumerate(texts, start=1):

            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).unsqueeze(0).to(DEVICE)
            x_lengths = torch.LongTensor([x.shape[-1]]).to(DEVICE)
            
            # tempo inferência
            t0 = time.perf_counter()
            _, y_dec, _ = generator.forward( x, x_lengths, n_timesteps=TIMESTEPS, temperature=1.5, stoc=False, spk=None, length_scale=0.91)
            t_grad = time.perf_counter() - t0

            # vocoder
            t0 = time.perf_counter()
            audio_hifi = (hifigan(y_dec).cpu().squeeze().clamp(-1,1).numpy())
            t_hifi = time.perf_counter() - t0

            t0 = time.perf_counter()
            audio_big  = (bigvgan(y_dec).cpu().squeeze().clamp(-1,1).numpy())
            t_big = time.perf_counter() - t0

            audio_hifi_int16 = normalize(audio_hifi)
            audio_big_int16  = normalize(audio_big)

            # salva wav
            write(os.path.join(out_dir_hifi, f"gradtts_hifigan_{voice}_{i:03d}.wav"), SAMPLE_RATE, audio_hifi_int16)
            write(os.path.join(out_dir_big, f"gradtts_bigvgan_{voice}_{i:03d}.wav"), SAMPLE_RATE, audio_big_int16)

            # métricas
            duration = len(audio_hifi) / SAMPLE_RATE
            rtf_grad = t_grad / duration
            rtf_hifi = t_hifi / duration
            rtf_big = t_big / duration
            rtf_grad_hifi = (t_grad + t_hifi) / duration
            rtf_grad_big = (t_grad + t_big) / duration

            rtf_grad_list.append(rtf_grad)
            rtf_hifi_list.append(rtf_hifi)
            rtf_big_list.append(rtf_big)
            rtf_grad_hifi_list.append(rtf_grad_hifi)
            rtf_grad_big_list.append(rtf_grad_big)

            print(f"[{i}/{len(texts)}] {text}")
            print(
                f"Grad={t_grad:.3f}s | "
                f"HiFi={t_hifi:.3f}s | "
                f"BigVGAN={t_big:.3f}s | "
                f"RTF(HiFi)={rtf_grad_hifi:.4f} | "
                f"RTF(BigVGAN)={rtf_grad_big:.4f}"
            )

print("\n==============================================")
print("RESUMO FINAL GRAD-TTS")
print("==============================================")

print(f"Frases sintetizadas: {len(rtf_grad_list)}")

print("\nRTF apenas do modelo acústico")
print(f"Grad-TTS             : {np.mean(rtf_grad_list):.4f} ± {np.std(rtf_grad_list):.4f}")

print("\nRTF apenas do vocoder")
print(f"HiFi-GAN             : {np.mean(rtf_hifi_list):.4f} ± {np.std(rtf_hifi_list):.4f}")
print(f"BigVGAN              : {np.mean(rtf_big_list):.4f} ± {np.std(rtf_big_list):.4f}")

print("\nRTF do sistema completo")
print(f"Grad-TTS + HiFi-GAN  : {np.mean(rtf_grad_hifi_list):.4f} ± {np.std(rtf_grad_hifi_list):.4f}")
print(f"Grad-TTS + BigVGAN   : {np.mean(rtf_grad_big_list):.4f} ± {np.std(rtf_grad_big_list):.4f}")