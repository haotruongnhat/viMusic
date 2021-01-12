import os, random, subprocess, copy
from pathlib import Path
import numpy as np
from archs.instruments.synthesis import Synthesis
import librosa
from IPython.display import Audio
import os
import scipy.io.wavfile

from IPython.display import Audio
from common.object import Configs
from archs.instruments.synthesis import Synthesis
from archs.instruments.track import Instrument
from archs.instruments.templates import Template
from scipy import signal

try:
    sinsyNG_exec_path = os.environ['SINSYNG_EXEC']
except:
    raise ValueError("No sinsyNG executable's environment path found")

class OUTPUT_TYPE:
    WAV_ARRAY = 0
    WAV_FILE = 1
    SCORE_OBJ = 2
    MIDI_FILE = 3
    JUPYTER_AUDIO = 4
    MIDI_OBJ = 5

class MODE:
    JUST_MELODY = 0
    MELODY_WITH_HUMMING = 1
    MELODY_WITH_TTS_VOICE = 2
    MELODY_WITH_DL_VOICE = 3

def remove_none_lyrics_note(score):
    notes = score.flat.getElementsByClass(['Note'])
    
    none_lyric_note_indices = np.where(np.array(notes.stream().lyrics()[1]) == None)[0]

    for idx in none_lyric_note_indices:
        score.remove(notes[idx], recurse=True)
        
    elems = list(score.flat.getElementsByClass(['Rest', 'ChordSymbol']))
    score.remove(elems, recurse=True)

def mastering(lead_score, config_name, voice_synth= 0, output_type=OUTPUT_TYPE.WAV_ARRAY, file_path=None, voice_model_instance=None, tempo=120):
    """
    Mastering the lead_score with config_name (genre)
    
    Args:
        lead_score (music21::Score)
        config_name: based on `config` object
        output_type (OUTPUT_TYPE)
        file_path (str): output's file path
    """

    list_of_matching_configs = list(Path('archs/instruments/genre_templates').glob('**/{}*.json'.format(config_name)))

    assert len(list_of_matching_configs) != 0, "No config matched"

    config_dir = random.choice(list_of_matching_configs)
    cfg = Configs()
    print(str(config_dir))
    cfg.read_from_file(str(config_dir))

    templ = Template.create_from_config(cfg)

    templ.assign_lead_melody(lead_score)
    
    if voice_synth == MODE.JUST_MELODY:
        if output_type == OUTPUT_TYPE.WAV_ARRAY:
            return templ.to_wav_array(tempo, True)
        elif output_type == OUTPUT_TYPE.WAV_FILE:
            assert file_path is not None, 'Should input file_path'
            return templ.to_wav_file(file_path, tempo, True)
        elif output_type == OUTPUT_TYPE.SCORE_OBJ:
            return templ.to_dynamic_instrument_score() 
        elif output_type == OUTPUT_TYPE.MIDI_FILE:
            assert file_path is not None, 'Should input file_path'
            templ.to_midi_file(file_path, tempo, True)   
        elif output_type == OUTPUT_TYPE.JUPYTER_AUDIO:
            return templ.play(tempo, True)
        elif output_type == OUTPUT_TYPE.MIDI_OBJ:
            return templ.to_midi(tempo, True)
        else:
            raise ValueError('Not valid output_type')

    if voice_synth != MODE.JUST_MELODY:
        print('Voice synthesis')
        voice_score = copy.deepcopy(lead_score)
        remove_none_lyrics_note(voice_score)

        offsets = [1, 1]

        if voice_synth == MODE.MELODY_WITH_HUMMING:
            voice_ins = Instrument('Vocals', 'Laah')
            midi = Synthesis.convert_score_to_pretty_mid(voice_score)
            bank_id, program_id = voice_ins.get_soundfont()
            midi.instruments[0].bank = bank_id
            midi.instruments[0].program = program_id
            midi.instruments[0].amplitude_offset = 1
            voice_arr = Synthesis.to_wav_array(midi)

            # Mix two wav arr
            offsets = [0.5, 1]

        elif voice_synth == MODE.MELODY_WITH_TTS_VOICE:
            fp = lead_score.write('xml')

            wav_fp = Path(fp)
            wav_fp = wav_fp.with_name(wav_fp.stem + '_voice.wav')

            args = (sinsyNG_exec_path, "-o", str(wav_fp), "-m", "Gene", fp)
            popen = subprocess.Popen(args, stdout=subprocess.PIPE)
            popen.wait()

            """Merge two wav files""" 
            # print('Read instrument wav from: {}'.format(file_path))
            print('Read voice wav from: {}'.format(str(wav_fp)))

            voice_arr, _ = librosa.load(str(wav_fp), 44100)

            # Mix two wav arr
            offsets = [0.5, 1]

            #Try to remove the written file
            os.remove(str(fp)) 
            os.remove(str(wav_fp))

        elif voice_synth == MODE.MELODY_WITH_DL_VOICE:
            """JR's Model"""
            assert voice_model_instance, 'No instance of VoiceDNN model'

            voice_arr, voice_sampling_rate = voice_model_instance.generate(lead_score, tempo, frequency_scaling=0.5)
            
            voice_arr = signal.resample(voice_arr, int(voice_arr.shape[0]*2)) #Resampling the voice array
            # voice_arr = np.concatenate( ( np.zeros((2, )), voice_arr)  ) #Add some offset to match the notes

            voice_arr[np.where(voice_arr > 0.6)] = 0.6
            voice_arr[np.where(voice_arr < -0.6)] = -0.6

            offsets = [0.5, 1]
            
        else:
            raise ValueError('Not valid voice_synth arg')

        print('Instrument Synthesis')
        music_arr = templ.to_wav_array(tempo, True)

        print('Mix Ins+Voice together')
        waveforms = [music_arr, voice_arr]
        synthesized = np.zeros(np.max([w.shape[0] for w in waveforms]))

        # Sum all waveforms in
        for offset, waveform in zip(offsets, waveforms):
            synthesized[:waveform.shape[0]] += (waveform / np.max(np.abs(waveform))) * offset

        # Normalize
        synthesized /= 2
        synthesized = synthesized.astype(np.float32)

        if output_type == OUTPUT_TYPE.WAV_ARRAY:
            return synthesized
        elif output_type == OUTPUT_TYPE.WAV_FILE:
            scipy.io.wavfile.write(file_path, 44100, synthesized)
            return float(synthesized.shape[0]) / 44100
        elif output_type == OUTPUT_TYPE.JUPYTER_AUDIO:
            return Audio(synthesized, rate=44100, normalize=False)