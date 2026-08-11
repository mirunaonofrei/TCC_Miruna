# Avaliação de modelos TTS para PT-BR

Este guia fornece uma avaliação de modelos TTS baseado em métricas objetivas: UTMOSv2, NISQA e WER/CER. Previamente, sugiro instalar os seguintes pacotes com SUDO:
```python
suto apt-get update
sudo apt install ffmpeg
sudo apt install -y espeak
```
## Modelos avaliados (PT-BR)
Os seguintes modelos foram avaliados:

### 1. Glowtts+hifigan (2020)
```python
gdown 18GL9CFjCkf9OPjg67hoC1kOE2dLT-_Tc 
conda create -n env_glowtts python=3.10 -y
conda activate env_glowtts
pip install torch torchvision torchaudio
pip install librosa soundfile phonemizer unidecode inflect tensorboardX matplotlib tqdm scipy
pip install --force-reinstall setuptools==68.2.2
pip install pypinyin jieba pyyaml coqpit
python 01_glowtts.py
```
- Vozes disponíveis: 1M: Edresson (TTS-Portuguese) 
- Descrição do modelo: O Glow-TTS atua como modelo acústico, convertendo a sequência de texto em uma representação mel-espectral por meio de uma arquitetura baseada em normalizing flows, com alinhamento texto-fala obtido pelo mecanismo de alinhamento monotônico. O espectrograma mel gerado pelo Glow-TTS é então convertido em waveform pelo HiFi-GAN, um vocoder generativo adversarial baseado em redes convolucionais, capaz de reconstruir áudio de alta fidelidade com baixa latência.
- Treinamento/afinamento: A dataset possui 10hrs de voz (10k frases) obtidos do dataset TTS-Portuguese, treinado do scratch.
- Custo: NVIDIA GeForce Gtx 1080 TI de 32GB.

### 2. GradTTS (2022) 
```python
conda create -n env_gradtts python=3.10
pip install torch torchvision torchaudio
pip install Cython==0.29.23
pip install numpy==1.26.4
pip install matplotlib einops inflect Unidecode librosa scipy tqdm gdown
pip install num2words nltk phonemizer
gdown 1X0jWeNXgxuTGvTPLEmJWhL37ZQTtHNmP && unzip prototype-final-gradtts.zip
cd prototype-final-gradtts/model/monotonic_align
python setup.py build_ext --inplace
python 02_gradtts.py
```
- Vozes disponíveis: 1F: Camila (Polly) 
- Descrição do modelo: Grad-TTS + HiFi-GAN e Grad-TTS + BigVGAN, utilizam o mesmo modelo acústico Grad-TTS, composto por um encoder e um preditor de duração baseados na arquitetura do Glow-TTS, além de um decoder por difusão baseado em U-Net para gerar espectrogramas Mel a partir do texto. A diferença entre as abordagens está no vocoder empregado para reconstruir a forma de onda: o HiFi-GAN prioriza síntese em tempo real com baixa latência, enquanto o BigVGAN utiliza ativações periódicas (AMP) para melhorar a fidelidade e a qualidade perceptual do áudio, especialmente em altas frequências.
- Treinamento/afinamento: Deve ter umas 10hrs de voz (10k frases) obtidos da Camila da AWS Polly, treinado do scratch.
- Custo: 1 GPU V100 para o hifigan e 2xA100 GPU para o bigvgan (12 dias).


### 9. StyleTTS2 (2026) [https://github.com/DW-Deepwin/s2s-demo]
```python
conda create -n env_styletts2 python=3.10
conda activate env_styletts2
sudo apt install -y espeak
pip install soundfile phonemizer pandas matplotlib jiwer tqdm tensorboard pyworld click nvitop
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pyyaml einops einops_exts munch scipy librosa transformers
python 09_styletts2.py
```

### UTMOSv2 [https://github.com/sarulab-speech/UTMOSv2]
```python
conda create -n env_utmos python=3.11 -y
conda activate env_utmos
pip install git+https://github.com/sarulab-speech/UTMOSv2.git
python avalia_utmosv2.py
```

### WER [https://github.com/jitsi/jiwer]
```python
conda create -n env_wer python=3.10 -y
conda activate env_wer
pip install jiwer soundfile tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install openai-whisper
python avalia_wer.py
```

### NISQA [https://github.com/gabrielmittag/NISQA]
```python
conda create -n env_nisqa python=3.10 -y
conda activate env_nisqa
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy pandas scikit-learn matplotlib seaborn tqdm pyyaml librosa soundfile
git clone https://github.com/gabrielmittag/NISQA.git
python avalia_nisqa.py
```