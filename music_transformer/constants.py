from magenta.music.constants import *
from fractions import Fraction

DEFAULT_MAX_SHIFT_STEPS = 100
MAX_VELOCITY = 127
MIN_VELOCITY = 0
MAX_PITCH = 127
MIN_PITCH = 0

DEFAULT_CHORD_BASE_PITCH = 48

C3 = 48
C2 = 36

DEFAULT_BPM = 120.0
DEFAULT_START_BEAT = Fraction(0,1)
NONE_BEAT = "0000"
MINIMUM_ERROR = 1e-5

note_duration_bin_index = {
    8 : {
        1 : "0001"
    },
    4 : {
        1 : "0010"
    },
    2 : {
        1 : "0011"
    },
    1 : {
        1 : "0010",
        2 : "0101",
        4 : "0110",
        8 : "0111",
        16 : "1000",
        32 : "1001",
        64 : "1010",
        128 : "1011",
        256 : "1100",
        512 : "1101",
        1024 : "1110"
    }
}


