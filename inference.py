import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile
from nltk.tokenize import word_tokenize

from models import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i


class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(char)
        return indexes


textclenaer = TextCleaner()

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


def compute_style(ref_dicts):
    reference_embeddings = {}
    for key, path in ref_dicts.items():
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = preprocess(audio).to(device)

        with torch.no_grad():
            ref = model.style_encoder(mel_tensor.unsqueeze(1))
        reference_embeddings[key] = (ref.squeeze(1), audio)

    return reference_embeddings


# load phonemizer
import phonemizer

lid = 1
languages = ["en", "us"]
mode = ["en-us", "en-gb-x-rp"]
global_phonemizer = phonemizer.backend.EspeakBackend(language=mode[lid], preserve_punctuation=True, with_stress=True)

# load hifi-gan

import sys

sys.path.insert(0, "./Demo/hifi-gan")

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from attrdict import AttrDict
from vocoder import Generator
import librosa
import numpy as np
import torchaudio

h = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


cp_g = scan_checkpoint("Vocoder/", 'g_')

config_file = os.path.join(os.path.split(cp_g)[0], 'config.json')
with open(config_file) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)

device = torch.device(device)
generator = Generator(h).to(device)

state_dict_g = load_checkpoint(cp_g, device)
generator.load_state_dict(state_dict_g['generator'])
generator.eval()
generator.remove_weight_norm()

# load StyleTTS
model_path = "./Models/StyleTTS_pretrained_LibriTTS/epoch_2nd_00050.pth"
model_config_path = "./Models/StyleTTS_pretrained_LibriTTS/config.yml"

config = yaml.safe_load(open(model_config_path))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

model = build_model(Munch(config['model_params']), text_aligner, pitch_extractor)

params = torch.load(model_path, map_location='cpu')
params = params['net']
for key in model:
    if key in params:
        if not "discriminator" in key:
            print('%s loaded' % key)
            model[key].load_state_dict(params[key])
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

# get first 3 training sample as references

train_path = config.get('train_data', None)
val_path = config.get('val_data', None)
val_path = "Data/val_list_libritts_seen.txt"
train_list, val_list = get_data_path_list(train_path, val_path)
import time

t0 = time.time()
ref_dicts = {}
for j in range(len(val_list)):
    filename = val_list[j].split('|')[0]
    print(filename)
    name = filename.split('/')[-1].replace('.wav', '')
    ref_dicts[name] = filename

reference_embeddings = compute_style(ref_dicts)
print("ref embed cost time: ", time.time() - t0)
# synthesize a text
text = ''' StyleTTS is a style-based generative model for parallel TTS that can synthesize diverse speech with natural prosody from a reference speech utterance. '''
# tokenize
ps = global_phonemizer.phonemize([text])
ps = word_tokenize(ps[0])
ps = ' '.join(ps)
print("phones : ", ps)
tokens = textclenaer(ps)
tokens.insert(0, 0)
tokens.append(0)
tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

converted_samples = {}

out_dir = "result"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with torch.no_grad():
    input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
    m = length_to_mask(input_lengths).to(device)
    t_en = model.text_encoder(tokens, input_lengths, m)

    for key, (ref, _) in reference_embeddings.items():

        s = ref.squeeze(1)
        style = s  # [1,128]

        d = model.predictor.text_encoder(t_en, style, input_lengths,
                                         m)  # t_en [1,512,182]  input_lengths s [1] d s[1,182,640]

        x, _ = model.predictor.lstm(d)  # x s[1,182,512]
        duration = model.predictor.duration_proj(x)  # duration s[1,182,1]
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)  # pred_dur s[182]

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))  # pred_aln_trg s[182,405]
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)
            # c_frame = 405
        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(
            device))  # d.transpose(-1, -2) [1,640,182]  pred_aln_trg.unsqueeze(0) [1,182,405]  en [1,640,405]
        style = s.expand(en.shape[0], en.shape[1], -1)  # s [1,128] style [1,640,128]

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)  # en [1,640,405]  s[1,128]  F0_pred [1,810] N_pred [1,810]

        out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)),
                            F0_pred, N_pred, ref.squeeze().unsqueeze(0))  # out [1,80,810]

        c = out.squeeze()  # c [80,810]
        y_g_hat = generator(c.unsqueeze(0))  # y_g_hat [1,1,243000]
        y_out = y_g_hat.squeeze().cpu().numpy()  # y_out s[1]

        c = out.squeeze()
        y_g_hat = generator(c.unsqueeze(0))
        y_out = y_g_hat.squeeze()

        converted_samples[key] = y_out.cpu().numpy()
for key, wave in converted_samples.items():
    print('Synthesized: %s' % key)
    soundfile.write(os.path.join(out_dir, key + "_zeroshot_{}.wav".format(languages[lid])), wave, 24000)

print("done")
print("cost time : ", time.time() - t0)
