import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import phonemizer
import soundfile
from nltk.tokenize import word_tokenize

from models import *
from utils import *
import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from attrdict import AttrDict
import librosa
import numpy as np
import torchaudio
import time


class TextCleaner:
    def __init__(self, dummy=None):
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»“” '
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

        # Export all symbols:
        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

        self.dicts = {}
        for i in range(len((symbols))):
            self.dicts[symbols[i]] = i

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.dicts[char])
            except KeyError:
                print(char)
        return indexes


class StyleTTS:
    def __init__(self, device_id=0):
        self.device = 'cuda:{}'.format(str(device_id)) if torch.cuda.is_available() else 'cpu'
        self.textcleaner = TextCleaner()
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        self.mean, self.std = -4, 4
        # load generator
        self.generator = self.load_model_generator()
        # load StyleTTS
        self.model_path = "./Models/LibriTTS/epoch_2nd_00050.pth"
        self.model_config_path = "./Models/LibriTTS/config.yml"

        self.config = yaml.safe_load(open(self.model_config_path))

        # load pretrained ASR model
        self.model = self.load_model_styletts()
        self.ref_embeddings = self.get_ref()
        self.keys = list(self.ref_embeddings.keys())
        self.out_dir = "result"
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.speaker_map = {"5123_34572_000011_000000":"man", "3224_168198_000002_000001":"woman", "7342_73961_000018_000006":"woman",
                            "3118_5909_000012_000004":"man", "1335_163935_000025_000005":"woman", "1571_141320_000039_000000": "man"}


    def load_model_generator(self):
        import sys
        sys.path.insert(0, "./hifi-gan")
        from vocoder import Generator
        h = None
        cp_g = self.scan_checkpoint("Vocoder/", 'g_')

        config_file = os.path.join(os.path.split(cp_g)[0], 'config.json')
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)

        device = torch.device(self.device)
        generator = Generator(h).to(device)

        state_dict_g = self.load_checkpoint(cp_g, device)
        generator.load_state_dict(state_dict_g['generator'])
        generator.eval()
        generator.remove_weight_norm()
        return generator

    def load_model_styletts(self):
        ASR_config = self.config.get('ASR_config', False)
        ASR_path = self.config.get('ASR_path', False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = self.config.get('F0_path', False)
        pitch_extractor = load_F0_models(F0_path)

        model = build_model(Munch(self.config['model_params']), text_aligner, pitch_extractor)

        params = torch.load(self.model_path, map_location='cpu')
        params = params['net']
        for key in model:
            if key in params:
                if not "discriminator" in key:
                    print('%s loaded' % key)
                    model[key].load_state_dict(params[key])
        _ = [model[key].eval() for key in model]
        _ = [model[key].to(self.device) for key in model]
        return model

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask

    def preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def compute_style(self, ref_dicts):
        reference_embeddings = {}
        for key, path in ref_dicts.items():
            wave, sr = librosa.load(path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                audio = librosa.resample(audio, sr, 24000)
            mel_tensor = self.preprocess(audio).to(self.device)

            with torch.no_grad():
                ref = self.model.style_encoder(mel_tensor.unsqueeze(1))
            reference_embeddings[key] = (ref.squeeze(1), audio)

        return reference_embeddings

    def load_checkpoint(self, filepath, device):
        assert os.path.isfile(filepath)
        print("Loading '{}'".format(filepath))
        checkpoint_dict = torch.load(filepath, map_location=device)
        print("Complete.")
        return checkpoint_dict

    def scan_checkpoint(self, cp_dir, prefix):
        pattern = os.path.join(cp_dir, prefix + '*')
        cp_list = glob.glob(pattern)
        if len(cp_list) == 0:
            return ''
        return sorted(cp_list)[-1]

    def get_ref(self):
        # get first 3 training sample as references

        train_path = self.config.get('train_data', None)
        val_path = "Data/val_list_libritts_seen.txt"
        train_list, val_list = get_data_path_list(train_path, val_path)
        ref_dicts = {}
        for j in range(len(val_list)):
            filename = val_list[j].split('|')[0]
            print(filename)
            name = filename.split('/')[-1].replace('.wav', '')
            ref_dicts[name] = filename
        reference_embeddings = self.compute_style(ref_dicts)
        return reference_embeddings

    def text_to_speech(self, text, lid=1, sid=0):
        # tokenize
        languages = ["en", "us"]
        mode = ["en-us", "en-gb-x-rp"]
        global_phonemizer = phonemizer.backend.EspeakBackend(language="en",
                                                                  preserve_punctuation=True,
                                                                  with_stress=True)

        ps = global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        print("phones : ", ps)
        tokens = self.textcleaner(ps)
        tokens.insert(0, 0)
        tokens.append(0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)  # [1,15]
        converted_samples = {}
        random_sid = random.randint(0, 5)  # random_sid = 3
        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)  # [15]
            m = self.length_to_mask(input_lengths).to(self.device)  # [1,15] all False
            t_en = self.model.text_encoder(tokens, input_lengths, m)  # [1,512,15]
            key = self.keys[random_sid]  # '3118_5909_000012_000004'
            ref, _ = self.ref_embeddings[key]  # [1,128]
            s = ref.squeeze(1)  # [1,128]
            style = s  # [1,128]

            d = self.model.predictor.text_encoder(t_en, style, input_lengths,
                                                  m)  # t_en [1,512,15] style s[1,128] input_lengths s[1] m s[1,15]  d s[1,15,640]

            x, _ = self.model.predictor.lstm(d)  # x s[1,15,512]
            duration = self.model.predictor.duration_proj(x)  # duration s[1,15,1]
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)  # pred_dur s[15]

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))  # pred_aln_trg s[input_lengths=15, int(pred_dur.sum()=63]
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1  # monotonous align
                c_frame += int(pred_dur[i].data)
                # c_frame = 405
            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(
                self.device))  # d.transpose(-1, -2) [1,640,15]  pred_aln_trg.unsqueeze(0) [1,15,63]  en [1,640,63]
            style = s.expand(en.shape[0], en.shape[1], -1)  # s [1,128] style [1,640,128]

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en,
                                                            s)  # en [1,640,63]  s[1,128]  F0_pred [1,126] N_pred [1,126]

            out = self.model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(self.device)),  # t_en [1, 512, 15]  pred_aln_trg [1, 15,63] > [1 ,512, 63]
                                     F0_pred, N_pred, ref.squeeze().unsqueeze(0))  # ref.squeeze().unsqueeze(0) = [1,128] out [1,80,126]

            c = out.squeeze()  # c [80,126]
            y_g_hat = self.generator(c.unsqueeze(0))  # y_g_hat [1,1,37800]
            y_out = y_g_hat.squeeze().cpu().numpy()  # y_out s[1]
            hid = str(time.time()).replace(".", "")
            int_id = str(random.randint(1000, 2000))
            soundfile.write(
                os.path.join(self.out_dir, key + "_{}_{}_{}.wav".format(languages[lid], int_id, hid)), y_out,
                24000)
            print("done")


styletts_obj = StyleTTS()
styletts_obj.text_to_speech("hello world")
