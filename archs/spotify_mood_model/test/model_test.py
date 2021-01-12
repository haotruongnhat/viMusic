from archs.spotify_mood_model.model import SpotifyMoodModel

class TestSpotifyMoodModel:
    spotify_mood_model = SpotifyMoodModel("models/spotify_mood_model/elmo","models/spotify_mood_model/checkpoints",3)
    mood = ['danceability','energy','speechiness','valence','acousticness','instrumentalness','liveness']
    """
    [lyrics_prep] TEST
    """
    def test_lyrics_prep_1(self):
        mood_lyrics = "\
Radio, video \n\
Boogie with a suitcase \n\
You're living in a disco \n\
Forget about the rat race\n\
\n\n\
Let's do the milkshake \n\
Selling like a hotcake \n\
Try some, buy some\n\
Fee-fi-fo-fum\n\
Talk about pop muzik\n\
Talk about pop muzik"
        lyrics_prep_result = TestSpotifyMoodModel.spotify_mood_model.lyrics_prep(mood_lyrics)
        assert lyrics_prep_result == 'radio video Boogie suitcase -PRON- live disco forget rat race let -PRON- milkshake selling like hotcake try buy Feefifofum Talk pop muzik Talk pop muzik'

    def test_lyrics_prep_2(self):
        mood_lyrics = "Hello everyone my name is Tyler"
        lyrics_prep_result = TestSpotifyMoodModel.spotify_mood_model.lyrics_prep(mood_lyrics)
        assert lyrics_prep_result == 'hello everyone name Tyler'

    def test_lyrics_prep_3(self):
        mood_lyrics = "Hello it's me \n\
I was wondering if after all these years you'd like to meet \n\
To go over everything \n\
They say that time's supposed to heal ya \n\
But I ain't done much healing"
        lyrics_prep_result = TestSpotifyMoodModel.spotify_mood_model.lyrics_prep(mood_lyrics)
        assert lyrics_prep_result == 'hello -PRON- wonder year would like meet to go everything -PRON- say time suppose heal ya but -PRON- not do much healing'

    def test_predict_1(self):
        mood_lyrics = "\
Radio, video \n\
Boogie with a suitcase \n\
You're living in a disco \n\
Forget about the rat race\n\
\n\n\
Let's do the milkshake \n\
Selling like a hotcake \n\
Try some, buy some\n\
Fee-fi-fo-fum\n\
Talk about pop muzik\n\
Talk about pop muzik"
        mood_result = TestSpotifyMoodModel.spotify_mood_model.predict(mood_lyrics)
        assert type(mood_result) == dict
        
        assert all([x in mood_result for x in TestSpotifyMoodModel.mood])

        assert all([mood_result[key] >= 0.0 and mood_result[key] <= 1.0 for key in mood_result])

    def test_predict_2(self):
        mood_lyrics = "Hello everyone my name is Tyler"
        mood_result = TestSpotifyMoodModel.spotify_mood_model.predict(mood_lyrics)
        assert type(mood_result) == dict
        
        assert all([x in mood_result for x in TestSpotifyMoodModel.mood])

        assert all([mood_result[key] >= 0.0 and mood_result[key] <= 1.0 for key in mood_result])

    def test_predict_3(self):
        mood_lyrics = "Hello it's me \n\
I was wondering if after all these years you'd like to meet \n\
To go over everything \n\
They say that time's supposed to heal ya \n\
But I ain't done much healing"
        mood_result = TestSpotifyMoodModel.spotify_mood_model.predict(mood_lyrics)
        assert type(mood_result) == dict
        
        assert all([x in mood_result for x in TestSpotifyMoodModel.mood])

        assert all([mood_result[key] >= 0.0 and mood_result[key] <= 1.0 for key in mood_result])
