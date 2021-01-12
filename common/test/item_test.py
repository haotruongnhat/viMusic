from common.music_item import LyricsItem

class TestLyricsItem:
    def test_find_lyrics_1(self):
        item = LyricsItem(None,"Adele","Hello",None,None)
        try:
            lyrics_in_paragraph = None
            lyrics_in_paragraph = item._find_lyrics("Adele","Hello")
        except:
            assert lyrics_in_paragraph == None
        
        assert lyrics_in_paragraph != None

    def test_find_lyrics_2(self):
        item = LyricsItem(None,"Adele","Hello",None,None)
        try:
            lyrics_in_paragraph = None
            lyrics_in_paragraph = item._find_lyrics("jjjjj","Hwevvwev")
        except:
            assert lyrics_in_paragraph == None
        
        assert lyrics_in_paragraph != None