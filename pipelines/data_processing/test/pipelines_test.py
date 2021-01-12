from pipelines.data_processing.pipelines import LyricsSegment
from common.music_item import LyricsItem

class TestLyricsSegment:
    def test_call_1(self):
        lyrics_segment = LyricsSegment()
