from sf2utils.sf2parse import Sf2File
from common.object import Dict
from pathlib import Path

soundfonts_path = Path('archs/instruments/viMusic.sf2')
full_name = ['Keys', 'Vocals', 'Orchestral', 
            'Synth', 'Clean Guitar', 
            'Distortion Guitar', 'Bass Guitar', 
            'Acapella', 'SGM', 'Beat Box',
            'Keys 2', 'Clean Guitar 2', 'Bass Guitar 2', 
            'Distortion Guitar 2', 'Orchestral 2', 'Synth 2',
            'Saxophone']
short_name = ['K', 'V', 'O', 'S', 'CG', 'DG', 'BG', 'A', 'SGM', 'BB',
              'K2', 'CG2', 'BG2', 'DG2', 'O2', 'S2', 'Sa']

with open(str(soundfonts_path), 'rb') as sf2_file:
    sf2 = Sf2File(sf2_file)
    
soundfont_info = Dict()

soundfont_name = str(soundfonts_path)
soundfont_info['package'] = soundfont_name
soundfont_info['instruments'] = {}

#Initialize package word
for name in full_name:
    soundfont_info['instruments'][name] = {}

for pre in sf2.presets:
    name = pre.name
    if name == 'EOP': break
    bank, program = pre.bank, pre.preset
    
    type_name = name.split('-')[0]
    last_name = '-'.join(name.split('-')[1:])
    if type_name in short_name:
        soundfont_info['instruments'][full_name[short_name.index(type_name)]][last_name] = dict(bank=bank, program=program)
        
### Parse patterns

map1 = {}
parent = Path('archs/instruments/patterns/data')

for path in parent.glob('*'):
    if path.name[0] == '.':
        continue
        
    map1[path.name] = {}
    
    for fp in path.glob('**/*.mid'):
        if not (fp.parents[0].name in map1[path.name].keys()):
            map1[path.name][fp.parents[0].name] = {}
    
        if not (fp.stem in map1[path.name][fp.parents[0].name].keys()):
            map1[path.name][fp.parents[0].name][fp.stem] = {}
            
        map1[path.name][fp.parents[0].name][fp.stem]['midiPath'] = str(fp)
        
soundfont_info['patterns'] = map1

soundfont_info.write_to_file(str(soundfonts_path.parent / 'viMusic_instruments_meta.json'))
