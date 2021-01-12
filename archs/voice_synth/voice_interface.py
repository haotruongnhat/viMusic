from itertools import cycle
import numpy as np
import scipy as sp
from scipy.io.wavfile import write
import pandas as pd
import librosa
import torch

from archs.voice_synth.mellotron.syllabify import syllable3
import re
import music21
import numpy as np

from archs.voice_synth.mellotron.hparams import create_hparams
from archs.voice_synth.mellotron.model import Tacotron2, load_model
from archs.voice_synth.mellotron.waveglow.denoiser import Denoiser
from archs.voice_synth.mellotron.layers import TacotronSTFT
from archs.voice_synth.mellotron.data_utils import TextMelLoader, TextMelCollate
from archs.voice_synth.mellotron.text import cmudict, text_to_sequence
from archs.voice_synth.mellotron.mellotron_utils import get_data_from_musicxml
from archs.voice_synth.mellotron.waveglow.glow import WaveGlow

from py3nvml.py3nvml import *

checkpoint_path = "archs/voice_synth/mellotron/models/mellotron_libritts.pt"
waveglow_path = 'archs/voice_synth/mellotron/models/waveglow.pt'
audio_paths = 'archs/voice_synth/mellotron/data/examples_filelist.txt'
arpabet_dict = cmudict.CMUDict('archs/voice_synth/mellotron/data/cmu_dictionary')

def panner(signal, angle):
    angle = np.radians(angle)
    left = np.sqrt(2)/2.0 * (np.cos(angle) - np.sin(angle)) * signal
    right = np.sqrt(2)/2.0 * (np.cos(angle) + np.sin(angle)) * signal
    return np.dstack((left, right))[0]

def load_mel(path, hparams, cuda_device_string):
    stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                    hparams.mel_fmax, cuda_device_string = cuda_device_string)

    audio, sampling_rate = librosa.core.load(path, sr=hparams.sampling_rate)
    audio = torch.from_numpy(audio)
    if sampling_rate != hparams.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = melspec.to(cuda_device_string)

    return melspec

def word_to_syllable_phonemes(word, add_bracket = True):
    sylls = [syl for syl in list(syllable3.generate(word.rstrip()))[0]]
    phonemes = [syl.phonemes for syl in sylls]
    stresses = [int(syl.get_stress()) for syl in sylls]

    if add_bracket:
        phonemes[0] = '{' + phonemes[0]
        phonemes[-1] += '}'
    
    return phonemes, stresses

def create_cuda_device_string(index):
    return "cuda:{}".format(index)



class VoiceDNN():
    def __init__(self, cuda_device_index=0):
        self._hparams = create_hparams()
        
        self._cuda_device_string = "cuda:{}".format(cuda_device_index)

        self._mellotron = load_model(self._hparams, self._cuda_device_string)
        self._mellotron.load_state_dict(
            torch.load(checkpoint_path, 
                        map_location=self._cuda_device_string)['state_dict'])

        WN_config = dict(n_layers=8, n_channels=256, kernel_size=3)
        self._waveglow = WaveGlow(n_mel_channels=80, n_flows=12, 
                                n_group=8, n_early_every=4, n_early_size=2, WN_config=WN_config,
                                cuda_device_string = self._cuda_device_string).to(self._cuda_device_string).eval()
        self._waveglow.load_state_dict(torch.load(waveglow_path, 
                        map_location=self._cuda_device_string))

        self._denoiser = Denoiser(self._waveglow, cuda_device_string=self._cuda_device_string).to(self._cuda_device_string).eval()

        self._dataloader = TextMelLoader(audio_paths, self._hparams, cuda_device_string=self._cuda_device_string)
        self._datacollate = TextMelCollate(1)

        file_idx = 0
        audio_path, text, sid = self._dataloader.audiopaths_and_text[file_idx]
        self._mel, self._sampling_rate = load_mel(audio_path, self._hparams, self._cuda_device_string), self._hparams.sampling_rate

        self._speaker_id = torch.LongTensor([70]).to(self._cuda_device_string)

        nvmlInit()

    def generate(self, score, bpm=120, frequency_scaling=0.4, speaker_id=70, method1 = False):
        data = get_data_from_musicxml(score, bpm, convert_stress=True, method1=method1)
 
        for i, (part, v) in enumerate(data.items()):

            tensor_array = [None]*3
            for i,name in enumerate(['rhythm','pitch_contour','text_encoded']):
                tensor_array[i] = data[part][name].to(self._cuda_device_string)

            rhythm, pitch_contour, text_encoded = tensor_array

            with torch.no_grad(): 
                mel_outputs, mel_outputs_postnet, gate_outputs, alignments_transfer = self._mellotron.inference_noattention(
                    (text_encoded, self._mel, self._speaker_id, pitch_contour*frequency_scaling, rhythm))

                audio = self._denoiser(self._waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[0, 0]
                audio = audio.cpu().numpy()
                audio_stereo = audio

            del tensor_array

        # audio_stereo = audio_stereo / np.max(np.abs(audio_stereo))
        return audio_stereo, self._sampling_rate

def create_test_score():
    def create_score_test_3(string=''):
        score2 = music21.stream.Score()
        part = music21.stream.Part()

        print("Generated String: {}".format(string))
        for word in string.split():

            sylls, stresses = word_to_syllable_phonemes(word, True)
        
            for idx, syl in enumerate(sylls):
                pitch = np.random.randint(40+12, 40 + 24 +12)
                note = music21.note.Note(pitch)
                note.quarterLength = 0.5
                note.lyric = ('-' if idx != 0 else '') +  syl + ('-' if idx != (len(sylls) -1) else '')
                part.append(note)
                
        score2.insert(part)
        return score2

    score = create_score_test_3('Saw you there Magic madness heaven sins Saw you there Magic madness heaven sins Saw you there Magic madness heaven sins Saw you there Magic madness heaven sins Saw you there Magic madness heaven sins Saw you there Magic madness heaven sins Saw you there Magic madness heaven sins')
    return score

if __name__ == "__main__":
    model = VoiceDNN(5)
    model.generate(create_test_score())