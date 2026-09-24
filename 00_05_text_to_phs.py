#from phonemizer import phonemize
import phonemizer
from phonemizer.phonemize import _phonemize
from tqdm import tqdm
import re

_punctuations = '!\'(),-.:;? '
pat = r'['+_punctuations[:-1]+']+'
def text2phoneme(text, language, backend):
    # Convert graphemes to phonemes.
    seperator = phonemizer.separator.Separator(' |', '', '|',)
    #seperator = phonemizer.separator.Separator('',)
    punctuations = re.findall(pat, text)
    ph_br = _phonemize(backend, text, seperator, False, 1, False, False)
    return ph_br

def load_backend():
    from phonemizer.punctuation import Punctuation
    from logging import Logger
    from phonemizer.logger import get_logger
    from phonemizer.backend import BACKENDS
    language = 'pt-br'
    backend = 'espeak'
    #seperator = phonemizer.separator.Separator(' |', '', '|',)
    phonemizer_back = BACKENDS[backend](
            language,
            punctuation_marks = Punctuation.default_marks(),
            preserve_punctuation=True,
            with_stress=True,
            tie=False,
            language_switch='keep-flags',
            words_mismatch='ignore',
            logger=get_logger())
    return phonemizer_back    

backend = load_backend()

#spk_file = "./corpus_text/sippulse_corpus_v3.txt"
spk_file = "./corpus_text/sippulse_corpus_correcoes.txt"
meta = [i.strip().replace('\n', '') for i in open(spk_file, 'r', encoding='utf-8').readlines()]
#print(meta[:100])
#out = open('all.txt', 'w', encoding="utf-8")
out = open('all_correcoes.txt', 'w', encoding="utf-8")
#speakers = ['Antonio','Humberto','Donato','Francisca','Manuela','Giovanna']
speakers = ['Humberto']
for speaker in speakers:
    spk = f'spk={speaker}'
    wav_prefix = f'../wavs_16k_3db_24k/{speaker}'
    for id, line in enumerate(tqdm(meta)):
        txt = line
        #print(txt)
        id_real = id + 10079
        #id = "{0:05d}".format(i)
        ph = text2phoneme(txt, 'pt-br', backend)
        ph = ph.replace('|', '')
        ph_str = f'ph=*{ph}&'
        file_id = f'id={speaker}_{id_real}'
        wav_path = f'wav={wav_prefix}/{speaker}_{id_real}.wav'
        # AQUI VA 
        #if ('?' in txt) or ('!' in txt):
        out.write(f'{file_id}|{ph_str}|{wav_path}|{spk}\n')
out.close()

"""
spk_file = "frases_otavio.csv"
meta = [i.strip().replace('\n', '') for i in open(spk_file, 'r', encoding='utf-8').readlines()]
print(len(meta))

out = open('all_keila.txt', 'w', encoding="utf-8")
speakers = ['Keila']
speaker = speakers[0]
wav_prefix = f'./wavs_16k_3db_24k/{speaker}'
spk = f'spk={speaker}'
for id, line in enumerate(tqdm(meta)):
    txt = line
    ph = text2phoneme(txt, 'pt-br', backend)
    ph = ph.replace('|', '')
    ph_str = f'ph=*{ph}&'
    file_id = f'id={speaker}_{id}'
    wav_path = f'wav={wav_prefix}/{speaker}_{id}.wav'
    out.write(f'{file_id}|{ph_str}|{wav_path}|{spk}\n')
"""
