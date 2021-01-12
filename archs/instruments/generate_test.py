from archs.instruments.generate import *

def get_score_from_file(path):
    import music21
    score = music21.converter.parse(path)
    return score

@pytest.mark.parametrize("score", [get_score_from_file('archs/instruments/test_data/song_en_short.xml')])
def test_remove_none_lyrics_note(score):
    remove_none_lyrics_note(score)

    #Check whether there's non-lyric notes
    notes = score.flat.getElementsByClass(['Note'])
    none_lyric_note_indices = np.where(np.array(notes.stream().lyrics()[1]) == None)[0]

    assert len(none_lyric_note_indices) == 0  

def mastering_lyric_data(name):
    if name == 'score':
        return [get_score_from_file('archs/instruments/test_data/song_en_short.xml')]
    elif name == 'config_name':
        return [tmpl.stem for tmpl in Path('archs/instruments/genre_templates').glob('**/*.json')]
    elif name == 'mode':
        return [MODE.JUST_MELODY, MODE.MELODY_WITH_HUMMING, MODE.MELODY_WITH_TTS_VOICE, MODE.MELODY_WITH_DL_VOICE]
    elif name == 'output_type':
        return [[OUTPUT_TYPE.WAV_ARRAY, ''], 
                        [OUTPUT_TYPE.WAV_FILE, 'archs/instruments/test_data/wav_file.wav'], 
                        [OUTPUT_TYPE.SCORE_OBJ, ''], 
                        [OUTPUT_TYPE.MIDI_FILE, 'archs/instruments/test_data/midi_file.mid'], 
                        [OUTPUT_TYPE.JUPYTER_AUDIO, ''], 
                        [OUTPUT_TYPE.MIDI_OBJ, '']]

def mastering_nonelyric_data(name):
    if name == 'score':
        return [get_score_from_file('archs/instruments/test_data/numb.mxl')]
    elif name == 'config_name':
        return [tmpl.stem for tmpl in Path('archs/instruments/genre_templates').glob('**/*.json')]
    elif name == 'mode':
        return [MODE.JUST_MELODY]
    elif name == 'output_type':
        return [[OUTPUT_TYPE.WAV_ARRAY, ''], 
                        [OUTPUT_TYPE.WAV_FILE, 'archs/instruments/test_data/wav_file.wav'], 
                        [OUTPUT_TYPE.SCORE_OBJ, ''], 
                        [OUTPUT_TYPE.MIDI_FILE, 'archs/instruments/test_data/midi_file.mid'], 
                        [OUTPUT_TYPE.JUPYTER_AUDIO, ''], 
                        [OUTPUT_TYPE.MIDI_OBJ, '']]


@pytest.mark.parametrize('score', mastering_lyric_data('score'))
@pytest.mark.parametrize('config_name', mastering_lyric_data('config_name'))
@pytest.mark.parametrize('mode', mastering_lyric_data('mode'))
@pytest.mark.parametrize('output_type', mastering_lyric_data('output_type'))
def test_mastering_with_lyric(score, config_name, mode, output_type):
    ret = mastering(score, config_name, mode, output_type=output_type[0], file_path=output_type[1])

# @pytest.mark.parametrize('score', mastering_nonelyric_data('score'))
# @pytest.mark.parametrize('config_name', mastering_nonelyric_data('config_name'))
# @pytest.mark.parametrize('mode', mastering_nonelyric_data('mode'))
# @pytest.mark.parametrize('output_type', mastering_nonelyric_data('output_type'))
# def test_mastering_none_lyric(score, config_name, mode, output_type):
#     ret = mastering(score, config_name, mode, output_type=output_type[0], file_path=output_type[1])