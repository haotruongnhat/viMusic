import os, shutil

import copy
import numpy as np
import music21
import miditoolkit
import common.pretty_midi.pretty_midi as pretty_midi
from pipelines.converter.hooktheory.hkt_generator import HKTObject
from common.constants import DEF, SIZE, CONVERSION, SEQType
# import common.vocab as vocab_lib
from common.vocab import RemiVocabItem, MusicAutobotVocabItem
from common.vocab_definition import REMI, MUSIC_AUTOBOT
from common.object import Note, Chord
from common.chord_recognition import MIDIChord
from abc import ABCMeta, abstractmethod, ABC
import lyricwikia
import re

"""
   - Music21's quarterLength = 4 MusicItem's duration (d4)
	- Minimum note length is d1 (which is sixteenth note)
	- Always proccess as multi-track score.
	- MusicItem: used for processing score
	- ItemProcess: normal conversion between events, barsequence and tokens
	- MusicItem(abstract), ItemProcessor (no function these 2 classes are of same name)
	- XMLItem : for XML
	- MidiItem : for Midi
	- RemiItem(object)
	- MusicAutoBotItem(Music)
"""

class ItemProcessor:
	"""
	containing functions used for conversion for different type of item
	"""
	@staticmethod
	def quantize_items(items, ticks=120):
		# grid
		grids = np.arange(0, items[-1].start, ticks, dtype=int)
		# process
		for item in items:
			index = np.argmin(abs(grids - item.start))
			shift = grids[index] - item.start
			item.start += shift
			item.end += shift
		return items  

	@staticmethod
	def pad_seq(seq, padding_len, vocab):
		"""
		Adding pad tokens or trim the sequence to normalize the sequece shape
		
		Args:
		
			seq (numpy array): tokens sequence
			padding_len (int): Length
			vocab: VocabItem
			
		Returns:
		
			padded_array (numpy array)
		"""
		pad_len = max(padding_len-seq.shape[0], 0)
		if pad_len == 0:
			return seq
			print('Should extend the `padding_len`. Current `padding_len` = {}, Sequence length: {}'.format(padding_len, seq.shape[0]))
		
		return np.pad(seq, (0, pad_len), 'constant', constant_values=vocab.pad_idx)[:padding_len]
	#events of notesequence and chord sequence to every other things
	@staticmethod
	def events_to_barsequence(notesequence, chordsequence,time_signature):
		"""
		Generate bar-relative sequence
		- Input: System's list of Note and system's list of Chord
		Returns:
		
			list of seq: Each element in list is a bar sequence
						 Each bar sequence has sorted note and chord inside
		"""


		# Note and chord's position by bar
		note_positional_indices = [n.at_measure(time_signature) for n in notesequence]
		chord_positional_indices = [c.at_measure(time_signature) for c in chordsequence]
		
		# Trim the sequence of chord and note to be same length
		max_measure = min(max(note_positional_indices), max(chord_positional_indices)) + 1
		
		bars = []

		for bar_idx in range(max_measure):

			# Extract note in bar_idx-th bar
			indices = [i for i,value in enumerate(note_positional_indices) if value==bar_idx]
			notes_in_bar = [notesequence[idx] for idx in indices]
			relative_notes = [n.to_relative_note_to_bar(time_signature) for n in notes_in_bar]
			
			# Extract chord in bar_idx-th bar
			indices = [i for i,value in enumerate(chord_positional_indices) if value==bar_idx]
			chords_in_bar = [chordsequence[idx] for idx in indices]
			relative_chords = [c.to_relative_note_to_bar(time_signature) for c in chords_in_bar]
			
			# Sort notes and chords by: priority, start_time, pitch
			# Chord's priority is 0 and Note's priority is 1, Chord object will come first
			seq = sorted(relative_notes + relative_chords, key = lambda x: (x.start, x.priority, x.pitch))
			
			bars.append(seq)
		
		return bars

	@staticmethod
	def events_to_tokens(notesequence,chordsequence,time_signature,vocab,padding_len=512, as_text=False):
		"""
		Generate tokens
		
		Args:
		
			as_text (bool): This will return the tokens as number or readable string
			
		Returns:
		
			tokens or string: int or string of tokens
		"""
		bars = ItemProcessor.events_to_barsequence(notesequence,chordsequence,time_signature)
		tokens = ItemProcessor.barsequence_to_tokens(bars,vocab,padding_len)
		if as_text == False:
			return tokens
		else:
			return vocab.textify(tokens.tolist())
	######################################################################

	#barsequence to every other things
	@staticmethod
	def barsequence_to_tokens(bars,vocab,padding_len):
		tokens = []

		for bar in bars:    
			# Add bar tokens at the beginning of bar
			tokens += [vocab.bar_idx]
			
			for event in bar:
				if isinstance(event, Chord):
					tokens += vocab.chord_to_tokens(event)
				elif isinstance(event, Note):
					tokens += vocab.note_to_tokens(event)

		tokens = np.array(tokens)
		tokens = ItemProcessor.pad_seq(tokens, padding_len, vocab)

		return tokens
	
	@staticmethod
	def barsequence_to_notesequence(sequence,time_signature):
		"""
		Extract notesequence from barsequence
		"""
		notesequence = []
		for bar_idx, bar in enumerate(sequence):
			for element in bar:
				if isinstance(element, Note):
					notesequence.append(element.to_absolute_note(bar_idx, time_signature))
		
		return notesequence
	
	@staticmethod
	def barsequence_to_chordsequence(sequence,time_signature):
		"""
		Extract chordsequence from barsequence
		"""  
		chordsequence = []
		
		num_bar = len(sequence)
		
		for bar_idx, bar in enumerate(sequence):
			for element in bar:
				if isinstance(element, Chord):
					chordsequence.append(element.to_absolute_chord(bar_idx, time_signature)) 
		
		# Fill in the chord's duration
		for elem_idx in range(len(chordsequence)):
			if elem_idx == len(chordsequence) -1:
				chordsequence[elem_idx].duration = num_bar*CONVERSION.bar_to_duration_ratio() - chordsequence[elem_idx].start
				continue
			
			chordsequence[elem_idx].duration = chordsequence[elem_idx + 1].start - chordsequence[elem_idx].start
			 
		return chordsequence
	######################################################################

	#tokens to every other things
	@staticmethod
	def tokens_to_barsequence(tokens,vocab):
		"""
		Parse tokens into barsequence
		"""
		# print(tokens)
		# Find indices of bar tokens
		bar_tks_indices = np.where(tokens == vocab.bar_idx)[0]
		# print(bar_tks_indices)
		# print(vocab.bar_idx)
		# Get position start and end tokens in vocab
		position_start_idx, position_end_idx = vocab.position_idx_endpoint

		bars = []
		for idx in range(len(bar_tks_indices)):
			if idx == (len(bar_tks_indices) -1):
				#Process last bar token. From bar token to first pad
				# import pdb; pdb.set_trace()
				try: 
					first_pad_token_index = np.where(tokens == vocab.pad_idx)[0][0]
				except:
					first_pad_token_index = len(tokens) - 1
				bar_tokens = tokens[bar_tks_indices[idx]:first_pad_token_index]
			else:
				bar_tokens = tokens[(bar_tks_indices[idx]+1):bar_tks_indices[idx+1]]
				
			pos_token_indices = np.where((bar_tokens >= position_start_idx) & (bar_tokens <= position_end_idx))[0]
			
			events = []
			for pos_idx in range(len(pos_token_indices)):
				if pos_idx == (len(pos_token_indices) -1):
					#Process last pos token. From last pos token to end
					event_tokens = bar_tokens[pos_token_indices[pos_idx]:]
				else:
					event_tokens = bar_tokens[pos_token_indices[pos_idx]:pos_token_indices[pos_idx+1]]
				
				events.append(list(event_tokens))

			bars.append(events)

		# Decode the tokens into events
		barsequence = []
		
		for bar in bars:
			seq = []
			for event_tokens in bar:
				try:	#Fix
					seq.append(ItemProcessor.tokens_to_events(event_tokens,vocab))
				except:
					pass
			if len(seq) > 0:
				barsequence.append(seq)
						
		return barsequence

	@staticmethod
	def tokens_to_notesequence(tokens,vocab,time_signature):
		return ItemProcessor.barsequence_to_notesequence(
			ItemProcessor.tokens_to_barsequence(
				tokens,vocab
			),
			time_signature
		)
	
	@staticmethod
	def tokens_to_chordsequuence(tokens,vocab,time_signature):
		chord_seq = ItemProcessor.barsequence_to_chordsequence(
			ItemProcessor.tokens_to_barsequence(tokens,vocab), time_signature)

		#Calculate duration
		for idx, chord in enumerate(chord_seq):
			if idx == (len(chord_seq) -1):
				chord[idx].duration = chord[idx].at_measure()*4.0 - chord[idx].start
				continue

			chord[idx].duration = chord[idx+1].start - chord[idx].start

		return chord_seq
	
	@staticmethod
	def tokens_to_events(tokens,vocab):
		"""
		Decode multiple tokens into related event
		"""

		# Classify the event type
		attribute_tokens = tokens[1:]

		def check_token_in_one_of_multiple_ranges(token, ranges):
			return any([(token in att_range) for att_range in ranges])

		# Chord
		if all([check_token_in_one_of_multiple_ranges(token, vocab.chord_idx_range) for token in attribute_tokens]):
			return vocab.tokens_to_chord(tokens)
		# Note
		elif all([check_token_in_one_of_multiple_ranges(token, vocab.note_idx_range) for token in attribute_tokens]):
			return vocab.tokens_to_note(tokens)
		else:
			raise ValueError('Can classify the event class')

	@staticmethod
	def barseq_to_score(seq, play_chord = True, tempo = 120, time_sig=music21.meter.TimeSignature('4/4')):
		noteseq = ItemProcessor.barsequence_to_notesequence(seq, time_signature=time_sig)

		chordseq = ItemProcessor.barsequence_to_chordsequence(seq, time_signature=time_sig)
		score = music21.stream.Score()
  
		part = music21.stream.Part()
		part.insert(music21.key.Key('C', 'major'))
		part.insert(time_sig)
		part.insert(music21.tempo.MetronomeMark(number=tempo))
		for elem in noteseq:
			n = elem.to_music21_note()
			part.insert(n)

		if not play_chord:
			for idx, elem in enumerate(chordseq):
				chords = elem.to_music21_chord(playable=False)
				if isinstance(chords, list):
					for c in chords:
						part.insert(c)
				else:
					part.insert(chords)

		score.insert(part)

		if play_chord:
			part = music21.stream.Part()
			part.insert(music21.key.Key('C', 'major'))
			part.insert(time_sig)
			part.insert(music21.tempo.MetronomeMark(number=tempo))
			for elem in chordseq:
				chords = elem.to_music21_chord(playable=True)
				if isinstance(chords, list):
					for c in chords:
						part.insert(c)
				else:
					part.insert(chords)
     
			score.insert(part)

		return score

	######################################################################

class MusicItem(ABC):
	def __init__(self,data=None):
		self.data = data
		super().__init__()

	@abstractmethod
	def from_file(self, fp):
		pass

class XMLItem(MusicItem): #only for XML

	def __init__(self, data=None):
		super().__init__(data)
		self.type = 'xml'
	######################
	### Those functions to process the music21 score
	######################
	def from_music21_score(self, score):
		self.data = score

	def from_file(self, fp):
		self.data = music21.converter.parse(fp)
		self.score_sanity_check()

	def score_sanity_check(self):
		# Check where the measure is calculated correct (being wrong sometime)
		bar_quarterLength = CONVERSION.bar_to_quarterLength_ratio(self.time_signature)
		
		uncertain_flag = False
		
		for part in self.data.parts:
			measures = part.getElementsByClass('Measure')
			for idx in range(len(measures)-1):
				if (measures[idx + 1].offset - measures[idx].offset) != bar_quarterLength:
					uncertain_flag = True
					# print("There is uncertain duration between two bars's offset.\nCorrect value: {} quarterLength. Actual value: {} quarterLength".format(
					#     bar_quarterLength, measures[idx + 1].offset - measures[idx].offset))
				
		quarterLength_per_bar = CONVERSION.bar_to_quarterLength_ratio(self.time_signature)        
		
		for part in self.data.parts:
			measures = part.getElementsByClass('Measure')
			for idx, measure in enumerate(measures):
				measure.offset = quarterLength_per_bar * idx
				measure.quarterLength = quarterLength_per_bar
	
	def parse_from_hkt_file(self, hkt_fp, chord_to_note=False):
		"""
		Parse the information from Hooktheory's Json file

		Args: 

			`hkt_fp` (str): Hooktheory's JSON file path 
			`chord_to_note` (bool): If True, turn the chord annotation to notes and add to new parts
		"""
		def hkt_note_length_to_m21_quarterLength(length):
			"""
			Convert Hooktheory's note length into music21's quarter length
			"""
			return length

		def hkt_pitch_to_m21_pitch(key, mode, scale_degree, octave=4):
			"""
			### Convert Hooktheory's pitch into music21's pitch
			
			Args:

				- key: Song's key ('C')
				- mode: Song's mode ('major', 'minor')
				- scale_degree: Note's degree in specific scale ( like in Cmaj: [C, D, E, F, G, A, B, C])
			"""
			return music21.key.Key(key, mode).pitches[int(scale_degree[-1])-1]

		def hkt_note_to_m21_note(key, mode, hkt_note):
			"""
			Convert Hooktheory's note into music21's note
			Args:
				key: Song's key ('C')
				mode: Song's mode ('major', 'minor')
				hkt_note: Hooktheory's note
			""" 
			pitch = hkt_pitch_to_m21_pitch(key, mode, hkt_note.scale_degree)
			duration = hkt_note_length_to_m21_quarterLength(hkt_note.note_length)
			offset = hkt_note_length_to_m21_quarterLength(hkt_note.start_beat_abs-1)

			isRest = hkt_note.isRest

			if not isRest:
				note = music21.note.Note(pitch, quarterLength=duration)
				note.octave = note.octave + hkt_note.octave
			else:
				note = music21.note.Rest(quarterLength=duration)

			note.offset = offset
			
			return note

		def hkt_chord_to_m21_chord(key, mode, hkt_chord, write_as_chord= False):
			"""
			Convert Hooktheory's chord into music21's chord
			Args:
				key: Song's key ('C')
				mode: Song's mode ('major', 'minor')
				hkt_chord: Hooktheory's chord
				write_as_chord: If False, the chord is used for annotation, else for insert notes
			""" 
			rm = music21.roman.RomanNumeral(hkt_chord.roman_basic, music21.key.Key(key, mode))
			# rm.inversion(hkt_chord.fb)
			chord_symbol = music21.harmony.chordSymbolFromChord(rm)
			chord_symbol.offset = hkt_note_length_to_m21_quarterLength(hkt_chord.start_beat_abs-1)
			
			if write_as_chord == True:
				duration = hkt_note_length_to_m21_quarterLength(hkt_chord.chord_duration)
				chord_symbol.writeAsChord = True
				chord_symbol.quarterLength = duration

			return chord_symbol

		'''Get information from HKTObj'''
		hkt_obj = HKTObject(hkt_fp)
		key_signature = hkt_obj.key
		mode = hkt_obj.mode
		time_signature = '{}/4'.format(hkt_obj.beat_per_measure)
		bpm = hkt_obj.bpm

		score = music21.stream.Score()

		for segment in hkt_obj.segments:
			part = music21.stream.Part()

			meter = music21.meter.TimeSignature(time_signature)
			key = music21.key.Key(key_signature, mode)
			tempo = music21.tempo.MetronomeMark(number=bpm)

			# Adding meta
			part.insert(0, tempo)
			part.insert(0, meter)
			part.insert(0, key)

			# Adding note to sheet
			for note in segment.melody:
				n = hkt_note_to_m21_note(key_signature, mode, note)
				part.insert(n)
			
			# Adding chord to sheet
			# (?!?) Dont know why adding this chord makes the tempo disabled (?!?)
			if not chord_to_note:
				for chord in segment.chords:
					c = hkt_chord_to_m21_chord(key_signature, mode, chord)
					part.insert(c)

			score.insert(part)

		# Adding chord part into score
		if chord_to_note:
			for segment in hkt_obj.segments:
				part = music21.stream.Part()
				meter = music21.meter.TimeSignature(time_signature)
				key = music21.key.Key(key_signature, mode)
				tempo = music21.tempo.MetronomeMark(number=bpm)

				# Adding meta
				part.insert(0, tempo)
				part.insert(0, meter)
				part.insert(0, key)

				for chord in segment.chords:
					c = hkt_chord_to_m21_chord(key_signature, mode, chord, True)
					part.insert(c)
				score.insert(part)

		self.data = score

	def transpose_to_nosharp_key(self, in_place=False):
		"""
		Remove note's semitone by transform to no-sharp key like C major or A minor
		"""
		current_degree, current_mode = self.key_degree_w_mode
		if current_mode == 'minor':
			key_name = 'a' # A minor         
		else:
			key_name = 'C' # C major
			
		return self.transpose_to(key_name, in_place=in_place)

	def transpose_octave(self, num_octave, in_place=False):
		"""
			Transpose all the notes by `num_octave` ascending or descending.
			
			Args:
				
				num_octave (int): Negative number for descending transposition
									and Positive number for ascending transposition
		"""
		
		transposed_score = self.data if in_place else copy.deepcopy(self.data)
		for part_idx in range(len(transposed_score.parts)):
			transposed_score.parts[part_idx].flat.transpose(num_octave * 12, inPlace = True)
		
		return None if in_place else self.__class__(transposed_score)

	def transpose(self, num_halftone, in_place=False):
		"""
			Transpose all the notes by `num_octave` ascending or descending.
			
			Args:
				
				num_octave (int): Negative number for descending transposition
									and Positive number for ascending transposition
		"""

		transposed_score = self.data if in_place else copy.deepcopy(self.data)
		for part_idx in range(len(transposed_score.parts)):
			transposed_score.parts[part_idx].flat.transpose(num_halftone, inPlace = True)

		return None if in_place else MusicItem(transposed_score)

	def transpose_to(self, key, mode_following=True, in_place=False):
		"""
			Transpose all the notes into input `key`.
			It will transpose as the shortest interval to destinate key
			
			Args:
				
				key: String (Case Sensitive) or music21.key.Key class
				mode_following: True: Follow the mode from key of original score
								Else it will take to mode from input `key`. Default: `major`
		"""
		dest_key = music21.key.Key(key) if not isinstance(key, music21.key.Key) else key
		dest_degree = dest_key.tonic.pitchClass

		current_degree, current_mode = self.key_degree_w_mode

		transpose_up_interval = (dest_degree + 12 - current_degree)
		transpose_down_interval = (dest_degree - current_degree)
		
		transpose_interval = transpose_up_interval if (abs(transpose_up_interval) <= abs(transpose_down_interval)) else transpose_down_interval
		transposed_score = self.data if in_place else copy.deepcopy(self.data)
		for part_idx in range(len(transposed_score.parts)):
			transposed_score.parts[part_idx].flat.transpose(transpose_interval, inPlace = True)
		
		return None if in_place else self.__class__(transposed_score)

	@property
	def score(self):
		return self.data

	@score.setter
	def score(self, obj):
		self.data = obj

	@property
	def time_signature(self):
		return self.data.flat.getElementsByClass('TimeSignature')[0] if self.data else music21.meter.TimeSignature('4/4')

	@property
	def key_obj(self):
		return self.data.flat.getElementsByClass('Key')[0] if self.data else music21.key.Key('C', 'major')

	@property
	def tempo_obj(self):
		return self.data.flat.getElementsByClass('MetronomeMark')[0] if self.data else music21.tempo.MetronomeMark(number=120)

	@property
	def key_name(self):
		current_key = self.data.flat.getElementsByClass('Key')[0] if self.data else music21.key.Key('C', 'major')
		return current_key.tonicPitchNameWithCase

	@property
	def key_name_w_mode(self):
		current_key = self.data.flat.getElementsByClass('Key')[0] if self.data else music21.key.Key('C', 'major')
		return current_key.tonicPitchNameWithCase, current_key.mode

	@property
	def key_degree(self):
		current_key = self.data.flat.getElementsByClass('Key')[0] if self.data else music21.key.Key('C', 'major')
		return current_key.tonic.pitchClass

	@property
	def key_degree_w_mode(self):
		current_key = self.data.flat.getElementsByClass('Key')[0] if self.data else music21.key.Key('C', 'major')
		return current_key.tonic.pitchClass, current_key.mode

	def show(self, as_text=False):
		self.data.show('txt') if as_text else self.data.show()
		
	def write(self, fp, file_format='xml'):
		self.data.write(file_format, fp)

	def stack(self, score, in_place=False):
		"""
		Add new parts from score into self.data 
		
		Args:
			score (music21.Score) or (MusicItem):
		"""
		# Type check
		ext_item = MusicItem(score) if isinstance(score, music21.stream.Score) else copy.deepcopy(score)

		if (self.time_signature.numerator != ext_item.time_signature.numerator) or (self.time_signature.denominator != ext_item.time_signature.denominator):
			print('WARNING: Unmatched time signature. Will change to match the source one \n \
				Soure: {}/{}. External: {}/{}'.format(self.time_signature.numerator,
													  self.time_signature.denominator,
													  ext_item.time_signature.denominator,
													  ext_item.time_signature.denominator))
			
		if (self.key_name_w_mode != ext_item.key_name_w_mode):
			print('WARNING: Unmatched key signature. Will change to match the source one \n \
				Soure: {}-{}. External: {}-{}'.format(self.key_name_w_mode[0],
													  self.key_name_w_mode[1],
													  ext_item.key_name_w_mode[0],
													  ext_item.key_name_w_mode[1]))

		if (self.data.flat.getElementsByClass(['Note', 'Chord']).highestTime != 
			ext_item.score.flat.getElementsByClass(['Note', 'Chord']).highestTime):
			print('WARNING: Unmatched score length')


		new_score = None
		if not in_place:
			new_score = copy.deepcopy(self.data)
		else:
			new_score = self.data
			
		for part in ext_item.score.parts:
			new_part = music21.stream.Part()
			new_part.append(self.time_signature)
			new_part.append(self.key_obj)
			new_part.append(self.tempo_obj)

			for elem in part.flat.getElementsByClass(['Note', 'Chord']):
				new_part.insert(elem)

			new_score.append(new_part)            

		if not in_place:
			return self.__class__(new_score) #return class according to current self

	def append(self, score, start_from_new_measure = True, in_place=False):
		"""
		Append score from current score
		
		Args:
			score (music21.Score):
			start_from_new_measure (bool):
		
		"""
		ext_item = MusicItem(score) if isinstance(score, music21.stream.Score) else copy.deepcopy(score)

		if (self.time_signature.numerator != ext_item.time_signature.numerator) or (self.time_signature.denominator != ext_item.time_signature.denominator):
			print('WARNING: Unmatched time signature. Will change to match the source one \n \
				Soure: {}/{}. External: {}/{}'.format(self.time_signature.numerator,
													  self.time_signature.denominator,
													  ext_item.time_signature.denominator,
													  ext_item.time_signature.denominator))
			
		if (self.key_name_w_mode != ext_item.key_name_w_mode):
			print('WARNING: Unmatched key signature. Will change to match the source one \n \
				Soure: {}-{}. External: {}-{}'.format(self.key_name_w_mode[0],
													  self.key_name_w_mode[1],
													  ext_item.key_name_w_mode[0],
													  ext_item.key_name_w_mode[1]))

		unmatched_parts_count = False
		if (len(self.data.parts) != len(ext_item.score.parts)):
			print('WARNING: Unmatched number of parts. \n \
				Soure: {} part(s). External: {} part(s)'.format(len(self.data.parts),
													  len(ext_item.score.parts)))
			unmatched_parts_count = True


		new_score = None
		if not in_place:
			new_score = copy.deepcopy(self.data)

		new_bar_idx = int(round(self.data.flat.getElementsByClass(['Note', 'Chord']).highestTime * 
								CONVERSION.quarterLength_to_bar_ratio(self.time_signature)))

		offset = new_bar_idx * CONVERSION.bar_to_quarterLength_ratio(self.time_signature)

		new_score = music21.stream.Score()
		
		# Insert notes and chord from source
		for idx, part in enumerate(self.data.parts):
			new_part = music21.stream.Part()
			new_part.append(self.time_signature)
			new_part.append(self.key_obj)
			new_part.append(self.tempo_obj)
		
			for elem in part.flat.getElementsByClass(['Note', 'Chord']):
				new_part.insert(elem)
				
			# If the parts is unmatched, store all elements from the external into first parts of existing score
			if unmatched_parts_count and (idx == 0):
				for elem in ext_item.score.flat.getElementsByClass(['Note', 'Chord']):
					elem.offset += offset
					new_part.insert(elem)
					
			if not unmatched_parts_count:
				for elem in ext_item.score.parts[idx].flat.getElementsByClass(['Note', 'Chord']):
					elem.offset += offset
					new_part.insert(elem)

			new_score.insert(new_part)

		if not in_place:
			return self.__class__(new_score)
		else:
			self.data = new_score

class MidiItem(MusicItem): #only for midi

	def __init__(self, data=None):
		super().__init__(data=score)
		self.type = 'midi'

	def from_file(self, fp):
		self.data = pretty_midi.PrettyMIDI(fp)
		self.pathFile = fp

class RemiItem(object):
	"""
	Being created to help encode score into REMI (REvamped MIDIderived events) and decode, by following the paper:
		[Pop Music Transformer: Generating Music with Rhythm and Harmony] [https://arxiv.org/pdf/2002.00212.pdf]
	"""

	def __init__(self, path=None,vocab=None):
		if path.endswith(".xml") or path.endswith(".mxl"):
			self.item = XMLItem()
			self.item.from_file(path)
		elif path.endswith(".midi") or path.endswith(".mid"):
			self.item = MidiItem()
			self.item.from_file(path)
		else:
			raise ValueError("Unknown file format: {}".format(path))
		self.vocab = vocab

	def to_notesequence(self,syll_encoder=None):
		"""
		Convert all the music21 notes into note sequence        
		"""
		if self.item.type == 'xml':
			sequence = []
			notes = self.item.score.flat.getElementsByClass(['Note'])
			for note in notes:
				n = Note()
				n.from_music21_note(note)
				if syll_encoder is not None:
					syll_emb = None
					if n.syllable not in syll_encoder.wv.vocab:
						syll_emb = syll_encoder.wv['I']
					else:
						syll_emb = syll_encoder.wv[n.syllable]
					n.syll_emb = copy.deepcopy(syll_emb)
				sequence.append(n)
		else:
			notes=[]
			sequence=[]
			for instrument in self.item.data.instruments:
				for note in instrument.notes: notes.append(note)

			for note in notes:
				n = Note()
				n.from_pretty_midi_note(note)
				sequence.append(n)

		return sequence        

	def to_chordsequence(self):
		"""
		Convert all music21 chords into chord sequence
		"""
		sequence = []
		if self.item.type == 'xml':
			#chord_for_melody_exist=True
			chords = self.item.score.flat.getElementsByClass('Chord')
			for chord in chords:
				c = Chord()
				c.from_music21_chord(chord)
				sequence.append(c)
		else:
			midi=miditoolkit.midi.parser.MidiFile(self.item.pathFile)
			if (any(x.key_name!=midi.key_signature_changes[0].key_name for x in midi.key_signature_changes)): return []

			#3. Check that time signature is 4/4
			if len(midi.time_signature_changes):
				time_signature = midi.time_signature_changes[0]
			else:
				time_signature = pretty_midi.TimeSignature(numerator=4,denominator=4,time=0.0)
			if time_signature.numerator != 4 or time_signature.denominator !=4:
				return []

			if len(midi.key_signature_changes):
				key = midi.key_signature_changes[0]
			else:
				key = pretty_midi.KeySignature(0,0) #default
			string_key = pretty_midi.key_number_to_key_name(key.key_number).split(" ")
			key_name = string_key[0] + ("m" if string_key[1] != 'Major' else "")
			original_key_scale = [p.pitchClass for p in music21.key.Key(key_name).pitches]
			if key.key_number <= 11: #major
				transform_key_scale = [p.pitchClass for p in music21.key.Key('C').pitches]
			else:
				transform_key_scale = [p.pitchClass for p in music21.key.Key('Am').pitches]
		
			notes = []
			accidental_notes_indices = []
			for instrument in midi.instruments:
				if not instrument.is_drum:
					for note in instrument.notes:
						#transform note
						try:
							index = original_key_scale.index(note.pitch % 12)
							correct_transformation = transform_key_scale[index]
							note.pitch = note.pitch - (note.pitch % 12) + correct_transformation
							notes.append(note)
						except:
							#skipping accidental note
							notes.append(note) 
							accidental_notes_indices.append(len(notes) - 1)
			notes = sorted(notes,key=lambda x : x.start)

			method=MIDIChord()
			chords=method.extract(notes=notes)
			melody_list=[]

			for start,end,chord_name in chords:
				#Check and split chord to make sure that it exists
				chord,other_stuff = chord_name.split(":")
				if chord == "N" or other_stuff == "N":
					continue
				if "/" in other_stuff:
					mode,bass = other_stuff.split("/")
				else:
					mode = other_stuff
					bass = "None" #easier to store in db
				split_midi = miditoolkit.midi.parser.MidiFile()
				split_notes = copy.deepcopy([(x,i) for i,x in enumerate(notes) if x.start >= start and x.end <= end])
				split_notes_indices = [x[1] for x in split_notes]
				split_notes = [x[0] for x in split_notes]
				if len(split_notes) == 0: #Seems to be no chord here
					continue
				#make sure that it begins at zero
				start_step = min(split_notes,key=lambda x : x.start).start
				piano = pretty_midi.Instrument(0)
				#restart notes, and checkk that there are no accidental notes
				is_successful = True
				for i,n in enumerate(split_notes):
					if split_notes_indices[i] in accidental_notes_indices:
						is_successful = False
						break
					#reset offset
					n.start -= start_step
					n.end -= start_step
					piano.notes.append(n)
				#add instrument, key, time signature
				split_midi.time_signature_changes = [time_signature]
				split_midi.key_signature_changes = [key]
				split_midi.instruments.append(piano)


				melody_list.append((start,end,split_midi,chord,mode,bass))

			unique_chords = []
			melodies = []
			for melody in melody_list:
				if melody[1:] not in unique_chords:
					unique_chords.append(melody[1:])
					melodies.append(melody)

			tmp_melodies=[]
			for melody in melodies: tmp_melodies.append(list(melody))

			for melody in tmp_melodies:
				c=Chord()
				melody[1] = CONVERSION.ticks_to_duration(melody[1],melody[2].ticks_per_beat)
				melody[0] = CONVERSION.ticks_to_duration(melody[0],melody[2].ticks_per_beat)

				c.from_miditoolkit_chord(melody)
				sequence.append(c)

		return sequence       

	def to_barsequence(self):
		"""
		Generate bar-relative sequence
		
		Returns:
		
			list of seq: Each element in list is a bar sequence
						 Each bar sequence has sorted note and chord inside
		"""
		notesequence = self.to_notesequence()
		chordsequence = self.to_chordsequence()

		return ItemProcessor.events_to_barsequence(notesequence,chordsequence,self.item.time_signature)

	def to_tokens(self, padding_len=512, as_text=False):
		"""
		Generate tokens
		
		Args:
		
			as_text (bool): This will return the tokens as number or readable string
			
		Returns:
		
			tokens or string: int or string of tokens
		"""
		notesequence = self.to_notesequence()
		chordsequence = self.to_chordsequence()
		return ItemProcessor.events_to_tokens(notesequence,chordsequence,self.item.time_signature,self.vocab,padding_len,as_text)
	
	def from_tokens(self, tokens, in_place=False, tempo=120, play_chord=True):
		"""
		Decode tokens to music21's score

		Args:

			tokens (array)
			in_place (bool)

		Return:

			RemiItem

		"""
		barseq = ItemProcessor.tokens_to_barsequence(tokens, vocab=self.vocab)
		score = ItemProcessor.barseq_to_score(barseq, play_chord, tempo)

		if in_place:
			self.item = XMLItem(data = score)
		else:
			if self.item is None:
				return XMLItem(data = score) 

			if self.item.type == "xml": 
				  return XMLItem(data = score) 
			else:
				return MidiItem(data = score) #Checking this, score is currently Music21 Score class

class MusicAutobotItem(MusicItem):
	"""
	Being created to help encode score into MusicAutobot sequence and decode, by following the article:
		[https://towardsdatascience.com/a-multitask-music-model-with-bert-transformer-xl-and-seq2seq-3d80bd2ea08e]
	"""
	def __init__(self, score=None, vocab=None):
		super().__init__(score=score)
		self.vocab = vocab if vocab else MusicAutobotVocabItem(MUSIC_AUTOBOT.INDEX_TOKENS)

	def to_timestep_multihot_array(self):
		"""
			Convert score's object to timestep_array.
			Timestep_array has size equivalent to number of time step. 
			At each time step will be information of notes being play (multi-hot(value) encoding)
		"""

		track_duration = CONVERSION.quarterLength_to_duration(max(self.data.flat.getElementsByClass('Note').highestTime,\
															self.data.flat.getElementsByClass('Chord').highestTime)) +1

		step_array = np.zeros((track_duration, len(self.data.parts), SIZE.NOTE))

		def note_data(pitch, note):
			return (pitch.midi, int(CONVERSION.quarterLength_to_duration(note.offset)), int(CONVERSION.quarterLength_to_duration(note.duration.quarterLength)))

		for idx, part in enumerate(self.data.parts):
			notes=[]
			for elem in part.flat:
				if isinstance(elem, music21.note.Note):
					notes.append(note_data(elem.pitch, elem))
				if isinstance(elem, music21.chord.Chord):
					for p in elem.pitches:
						notes.append(note_data(p, elem))

			''' Pass the notes into score array'''
			# sort notes by offset (1), duration (2) so that hits are not overwritten and longer notes have priority
			notes_sorted = sorted(notes, key=lambda x: (x[1], x[2])) 
			for n in notes_sorted:
				if n is None: continue
				pitch,offset,duration = n
				if duration > SIZE.DURATION: duration =  SIZE.DURATION
				step_array[offset, idx, pitch] = duration
				step_array[offset+1:offset+duration, idx, pitch] = DEF.VALTCONT      # Continue holding note
			
		return step_array
  
	def from_timestep_multihot(self, timestep_array, bpm=120):
		"""
		Convert the timestep_array into score
		"""
		from itertools import groupby
		# combining notes with different durations into a single chord may 
		# overwrite conflicting durations. Example: aylictal/still-waters-run-deep
		def group_notes_by_duration(notes):
			"separate notes into chord groups"
			keyfunc = lambda n: n.duration.quarterLength
			notes = sorted(notes, key=keyfunc)
			return [list(g) for k,g in groupby(notes, keyfunc)]

		stream = music21.stream.Score()
		stream.append(music21.key.Key('C'))
		stream.append(music21.meter.TimeSignature('4/4'))
		stream.append(music21.tempo.MetronomeMark(number=bpm))
		for inst in range(timestep_array.shape[1]):
			part = music21.stream.Part()
			part.append(music21.instrument.Piano())

			partarr = timestep_array[:,inst,:]
			for tidx,t in enumerate(partarr):
				note_idxs = np.where(t > 0)[0] # filter out any negative values (continuous mode)
				if len(note_idxs) == 0: continue
				notes = []
				for nidx in note_idxs:
					note = music21.note.Note(nidx)
					note.duration = music21.duration.Duration(CONVERSION.duration_to_quarterLength(partarr[tidx,nidx]))
					notes.append(note)
				for g in group_notes_by_duration(notes):
					if len(g) == 1:
						part.insert(CONVERSION.duration_to_quarterLength(tidx), g[0])
					else:
						chord = music21.chord.Chord(g)
						part.insert(CONVERSION.duration_to_quarterLength(tidx), chord)

			stream.append(part)

		stream = stream.transpose(0)
		self.data = stream
  
	def to_polyphony_encode(self, merge_into_one_part=False, skip_last_rest=True):
		""" 
		Encode score into polyphony encoding ( which is exactly the same as Polyphony RNN)

		Args:
			merge_into_one_part (bool): There're multiple parts. If True, all parts's notes are merged into
										one polyphony encoding array
			skip_last_rest (bool):

		Return:
			list of (array): Return list of polyphony encoding array. 
							If `merge_into_one_part` is True, list's length is 1, else it's equivelant to num_parts
		"""
		# combine instruments
		result = []
		timestep_array = self.to_timestep_multihot_array()
		
		if not merge_into_one_part:
			num_parts = timestep_array.shape[1]
			for part_idx in range(num_parts):
				array = timestep_array[:, part_idx:(part_idx+1), :] # Using `part_idx:(part_idx+1)` will keep the timestep_array shape
				result.append(self.from_timestep_array_to_poly_enc(array, skip_last_rest))
		else:
			result.append(self.from_timestep_array_to_poly_enc(timestep_array, skip_last_rest))

		return result
	
	@classmethod
	def from_timestep_array_to_poly_enc(cls, timestep_array, skip_last_rest=True):
		"""
			Convert timestep_array into polyphony encoding

			Args:
				timestep_array (array of (step, part, note))
		"""
		result = []
		wait_count = 0
		
		for idx, timestep in enumerate(timestep_array):
			flat_time = cls.timestep_to_notes(timestep)
			if len(flat_time) == 0:
				wait_count += 1
			else:
				# pitch, octave, duration, instrument
				if wait_count > 0: result.append([DEF.VALTSEP, wait_count])
				result.extend(flat_time)
				wait_count = 1
		if wait_count > 0 and not skip_last_rest: result.append([DEF.VALTSEP, wait_count])
		
		return np.array(result, dtype=int).reshape(-1, 2)
	
	def from_poly_encode(self, poly_encode, note_size = SIZE.NOTE, bpm=120):
		"""
			Convert the poly_encode array into score (in_place function type)

			Args:
				poly_encode (list): List of polyphony encoding array
		"""
		timestep_array = self.from_poly_encode_to_timestep_multihot(poly_encode, note_size)
		self.from_timestep_multihot(timestep_array, bpm)
	
	@classmethod
	def timestep_to_notes(cls, step, note_range=DEF.NORMAL_RANGE, enc_type=None):
		"""
			Each step's note descriptions. Each note will be append into array as [pitch, duration]

			Args:

				step (array): one step element in timestep_array
				note_range (minBound, maxBound): The note has pitch in this range will be included
				enc_type (None, parts, full): Three types of note's description

			Return:
				list: note-on-timestep's description Depend on `enc_type`. On Default, it's [[mote, duration],[mote, duration]]
		"""
		notes = []

		for i,n in zip(*step.nonzero()):
			d = step[i,n]
			if d < 0: continue # only supporting short duration encoding for now
			if n < note_range[0] or n >= note_range[1]: continue # must be within midi range
			notes.append([n,d,i])

		notes = sorted(notes, key=lambda x: x[0], reverse=True) # sort by note (highest to lowest)

		if enc_type is None: 
			# note, duration
			return [n[:2] for n in notes] 
		if enc_type == 'parts':
			# note, duration, part
			return [n for n in notes]
		if enc_type == 'full':
			# note_class, duration, octave, instrument
			return [[n%12, d, n//12, i] for n,d,i in notes] 
	
	@classmethod
	def from_poly_encode_to_timestep_multihot(cls, multi_poly_encode, note_size=SIZE.NOTE):
		"""
			Convert polyphony encoding array into timestep_array

			Args:
				multi_poly_encode (list): List of polyphony encoding array      

			Return:
				array: Timestep array       
		"""


		# num_instruments = 1 if len(poly_encode.shape) <= 2 else poly_encode.max(axis=0)[-1]
		num_parts = len(multi_poly_encode)

		def poly_encode_len(poly_encode):
			duration = 0

			for t in poly_encode:
				note, dur = t.tolist()
				if note == DEF.VALTSEP: duration += dur

			return duration + 1

		max_len = max([poly_encode_len(poly_encode) for poly_encode in multi_poly_encode])
		step_array = np.zeros((max_len, num_parts, note_size))

		for part_idx, poly_encode in enumerate(multi_poly_encode):
			idx = 0
			for step in poly_encode:
				note, duration = step.tolist()

				# n,d,i = (step.tolist()+[0])[:3] # or n,d,i
				if note < DEF.VALTSEP: 
					continue # special token
				if note == DEF.VALTSEP:
					idx += duration
					continue

				step_array[idx, part_idx, note] = duration

		return step_array

	######################
	### Those functions to tokenize the music21's score
	#####################
	def to_tokens(self, padding_len=128, merge_into_one_part = False, as_text=False):
		"""
			Convert score into tokens array

			Args:
				merge_into_one_part (bool): merge_into_one_part (bool): There're multiple parts. If True, all parts's notes are merged into
											one polyphony encoding array

				as_text (bool): Will return the tokens string as readable-text

			Return:                              
				list of (str): Return list of tokens string. 
							   If `merge_into_one_part` is True, list's length is 1, else it's equivelant to num_parts
		"""

		def seq_prefix(seq_type, vocab):
			if seq_type == SEQType.Empty: return np.empty(0, dtype=int)
			start_token = vocab.bos_idx
			if seq_type == SEQType.Chords: start_token = vocab.stoi[MUSIC_AUTOBOT.CSEQ.prefix]
			if seq_type == SEQType.Melody: start_token = vocab.stoi[MUSIC_AUTOBOT.MSEQ.prefix]
			return np.array([start_token, vocab.pad_idx])

		def part_to_tokens(t, vocab, seq_type=SEQType.Sentence, add_eos=False):
			if isinstance(t, (list, tuple)) and len(t) == 2: 
				return [part_to_tokens(x, seq_type) for x in t]

			t = t.copy()

			t[:, 0] = t[:, 0] + vocab.note_range[0]
			t[:, 1] = t[:, 1] + vocab.dur_range[0]

			prefix = seq_prefix(seq_type, vocab)
			suffix = np.array([vocab.stoi[MUSIC_AUTOBOT.EOS.prefix]]) if add_eos else np.empty(0, dtype=int)

			return np.concatenate([prefix, t.reshape(-1), suffix])

		poly_enc = self.to_polyphony_encode(merge_into_one_part)

		if as_text:
			texts = []

			for idx, part in enumerate(poly_enc):
				tks = part_to_tokens(part, self.vocab, add_eos=True)
				padded_tks = pad_seq(tks, padding_len, self.vocab).tolist()

				texts.append(self.vocab.textify(padded_tks))

			return texts
		else:
			tokens = np.zeros( (len(poly_enc), padding_len), dtype=np.int16)
			# Polyphony encoding to tokens
			for idx, part in enumerate(poly_enc):
				tks = part_to_tokens(part, self.vocab, add_eos=True)
				padded_tks = pad_seq(tks, padding_len, self.vocab)
				tokens[idx, :] = padded_tks
				
			return tokens

	def to_melody_chord_tokens(self, padding_len=128):
		"""
		By default, the MusicItem is parse from Hooktheory's dataset
		Hence, there're only two parts available: melody and chord

		Returns:

			list of melody and chord tokens
		"""
		tokens = self.to_tokens(padding_len)

		# Replace the first token
		tokens[DEF.MELODY_PART_INDEX][0] = self.vocab.stoi[MUSIC_AUTOBOT.MSEQ.prefix]
		tokens[DEF.CHORD_PART_INDEX][0] = self.vocab.stoi[MUSIC_AUTOBOT.CSEQ.prefix]

		return tokens

	def to_positional_melody_chord_tokens(self, padding_len=128):
		"""
		Returns:
		"""
		tokens = self.to_positional_enc_tokens(padding_len)
		# Replace the first token
		tokens[DEF.MELODY_PART_INDEX][DEF.TOKENS_INDEX][0] = self.vocab.stoi[MUSIC_AUTOBOT.MSEQ.prefix]
		tokens[DEF.CHORD_PART_INDEX][DEF.TOKENS_INDEX][0] = self.vocab.stoi[MUSIC_AUTOBOT.CSEQ.prefix]

		return tokens

	def to_positional_enc_tokens(self, padding_len=128, merge_into_one_part = False):
		def position_enc(idxenc, vocab):
			"Calculates positional beat encoding."
			sep_idxs = (idxenc == vocab.sep_idx).nonzero()[0]
			sep_idxs = sep_idxs[sep_idxs+2 < idxenc.shape[0]] # remove any indexes right before out of bounds (sep_idx+2)
			dur_vals = idxenc[sep_idxs+1]
			dur_vals[dur_vals == vocab.mask_idx] = vocab.dur_range[0] # make sure masked durations are 0
			dur_vals -= vocab.dur_range[0]

			posenc = np.zeros_like(idxenc)
			posenc[sep_idxs+2] = dur_vals

			return posenc.cumsum()

		multi_part_tokens = self.to_tokens(padding_len, merge_into_one_part)

		position_enc_tokens = np.zeros( (len(multi_part_tokens), 2, padding_len) , dtype=np.int16) # Dims: (number_of_parts, tokens_and_pos_enc, padding_len)
		position_enc_tokens[:, DEF.TOKENS_INDEX, :] = multi_part_tokens

		# Positional enc
		for idx in range(position_enc_tokens.shape[0]):
			position_enc_tokens[idx, DEF.POSITIONAL_ENC_INDEX, :] = position_enc(position_enc_tokens[idx, DEF.TOKENS_INDEX, :], self.vocab) // 4
			position_enc_tokens[idx, DEF.POSITIONAL_ENC_INDEX, :] += 1

			# Begin of sequence beat is 0
			position_enc_tokens[idx, DEF.POSITIONAL_ENC_INDEX, :][0] = 0
			position_enc_tokens[idx, DEF.POSITIONAL_ENC_INDEX, :] = np.where((position_enc_tokens[idx, DEF.TOKENS_INDEX, :] != self.vocab.pad_idx), position_enc_tokens[idx, DEF.POSITIONAL_ENC_INDEX, :], 0)

			# position_enc_tokens[idx, DEF.POSITIONAL_ENC_INDEX, :][0:2] = np.array([0, 0])
			# position_enc_tokens[idx, DEF.POSITIONAL_ENC_INDEX, :] == self.vocab.pad_idx = np.array([0, 0])

		return position_enc_tokens

	def from_tokens(self, tokens):
		"""
			Convert tokens into score (in_place function type)
		"""
		poly_enc = self.from_tokens_to_poly_enc(tokens, self.vocab)
		self.from_poly_encode(poly_enc)

	@classmethod
	def from_tokens_to_poly_enc(cls, tokens, vocab):
		"""
			Convert tokens into polyphony encoding array

			Args:
				tokens (list): Contain list of tokens string
				vocab (MusicVocab): Contain methods for tokenizing and untokenizing

			Return:
				list: List of polyphony encoding array             
		"""

		def to_valid_idxenc(t, valid_range):
			r = valid_range
			t = t[np.where((t >= r[0]) & (t < r[1]))]
			if t.shape[-1] % 2 == 1: t = t[..., :-1]
			return t

		def to_valid_npenc(t):
			is_note = (t[:, 0] < DEF.VALTSEP) | (t[:, 0] >= SIZE.NOTE)
			invalid_note_idx = is_note.argmax()
			invalid_dur_idx = (t[:, 1] < 0).argmax()

			invalid_idx = max(invalid_dur_idx, invalid_note_idx)
			if invalid_idx > 0: 
				if invalid_note_idx > 0 and invalid_dur_idx > 0: invalid_idx = min(invalid_dur_idx, invalid_note_idx)
				print('Non midi note detected. Only returning valid portion. Index, seed', invalid_idx, t.shape)
				return t[:invalid_idx]
			return t

		def tokens_to_part(t, vocab, validate=True):
			if validate: t = to_valid_idxenc(t, vocab.npenc_range)
			t = t.copy().reshape(-1, 2)

			if t.shape[0] == 0: return t

			t[:, 0] = t[:, 0] - vocab.note_range[0]
			t[:, 1] = t[:, 1] - vocab.dur_range[0]
			if validate: return to_valid_npenc(t)
			return t

		return [tokens_to_part(tks, vocab) for tks in tokens]

class LyricsItem(object):
	"""
	A Lyrics item containing a segment of the lyrics, with its label
	"""
	def __init__(self,id,artist,song,genre,ssm,segment_ssm):
		"""
		id: id for the lyrics (str)
		artist and song: I guess you guys don't need explaination right :v ?
		segment_ssm: ssm segment
		"""
		self._id = id

		###1. Find lyrics based on artist and song
		#An array of segments, each is a list of sentences
		self._lyrics = None
		self._find_lyrics(artist,song)

		###2. genre
		self._genre = genre

		###3. Convert id
		self._id = re.sub(r"ObjectId\((.*)\)",r"\1",self._id)


		###4. Save ssm
		#a normalize self-similarity ssm of sentences, with shape = (len(sentence),len(sentence))
		self._ssm = ssm


		###5. segment ssm
		#a normalize self-similarity ssm of segments, with shape = (num_seg,num_seg)
		#not really need it at the moment, but you know... just save it in case someone need it
		self._segment_ssm = segment_ssm


		###6. Added features

	def _find_lyrics(self,artist,song):
		try:
			lyrics = lyricwikia.get_lyrics(artist,song)
			if lyrics is None or not isinstance(lyrics,str) or lyrics == "":
				raise RuntimeError("Lyrics of {} is not found. Skipping...".format(self._id))
		except:
			raise RuntimeError("Lyrics of {} is not found. Skipping...".format(self._id))

		#preprocess lyrics
		paragraphs = lyrics.split("\n\n")
		self._lyrics = [x.split("\n") for x in paragraphs]

	@property
	def input(self):
		return self._ssm

	@property
	def lyrics(self):
		return self._lyrics
	
		