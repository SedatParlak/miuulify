from flask import Blueprint, render_template, request, flash, redirect, url_for, session
import validators
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import joblib
import pickle



cid = '4ed8351b49ee4d53b7dbf0c9a4f5f030'
secret = 'a39aecc1ca2c4e7f943fcc1ee0d70224'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

df = pd.read_csv('dataset/language.csv')


#############################################################################

views = Blueprint('views', __name__)


@views.route('/', methods=['GET', 'POST'])
def home():

    if request.method == 'POST':
        spotift_link = request.form.get('spotify-link')
        
        if not validators.url(spotift_link):
            flash('URL is not valid')
        else:
            song_id = spotift_link.split('/')[-1:][0].split('?')[0]

            if len(df[df['id'] == song_id]) == 0:
                
                track_ids, mood = recommender_1(df, spotift_link)

                if mood == 0:
                    session['song_mood'] = ['Hmmm... Mutlu olduğunu hissediyorum. Hayatın tadını çıkarıyor gibisin. Hadi! Sana eşlik edecek olan şarkılara bir göz at.']
                elif mood == 1:
                    session['song_mood'] = ['Hmmm... Sanırım bugün biraz hüzünlüsün. Çok kafaya takmamak gerek ya. Sana bu modda önerdiklerime bir göz at.'],
                elif mood == 2:
                    session['song_mood'] = ['Hmmm...Enerjik olduğunu hissediyorum. Oturmaya mı geldik kardeşim. O zaman dans!'],
                else:
                    session['song_mood'] = ['Hmmm...Sakin bir anındayız. Hadi biraz rahatlayalım ve anın tadını çıkaralım. Bu moda uygun önerdiklerime bir göz at.']

                
            else:
                mood = df[df['id'] == song_id]['moodd'].values[0]   

                if mood == 0:
                    session['song_mood'] = 'Hmmm... Mutlu olduğunu hissediyorum. Hayatın tadını çıkarıyor gibisin. Hadi! Sana eşlik edecek olan şarkılara bir göz at.'
                elif mood == 1:
                    session['song_mood'] = 'Hmmm... Sanırım bugün biraz hüzünlüsün. Çok kafaya takmamak gerek ya. Sana bu modda önerdiklerime bir göz at.',
                elif mood == 2:
                    session['song_mood'] = 'Hmmm...Enerjik olduğunu hissediyorum. Oturmaya mı geldik kardeşim. O zaman dans!',
                else:
                    session['song_mood'] = 'Hmmm...Sakin bir anındayız. Hadi biraz rahatlayalım ve anın tadını çıkaralım. Bu moda uygun önerdiklerime bir göz at.'       
                
                track_ids = recommender_2(df, spotift_link, mood)            
            
            recommend_dict = {}
            for index, id_ in enumerate(track_ids):
                recommend_dict[index] = get_tracks_properties(id_)
            
            session['recommend_dict'] = recommend_dict
            
            return redirect(url_for('views.recommend'))

    return render_template('index.html')



@views.route('/recommend')
def recommend():

    song_mood = session.get('song_mood')
    recommend_dict = session.get('recommend_dict')
    return render_template('recommend.html', recommend_dict=recommend_dict, song_mood=song_mood)


##########################################################################################


def recommender_1(dataframe, spotify_link, scale=0.1):
    
    # get song feautures from Spotify API
    song_features = sp.audio_features(spotify_link)[0]
    
    # create a dataframe with song features
    song_df = pd.DataFrame([song_features['danceability'],
                       song_features['acousticness'],
                       song_features['energy'],
                       song_features['instrumentalness'],
                       song_features['liveness'],
                       song_features['valence'],
                       song_features['loudness'],
                       song_features['speechiness'],
                       song_features['tempo'],
                       song_features['key'],
                       song_features['time_signature']]).T
    
    # rename columns
    song_df.columns = ['danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'valence',
                  'loudness', 'speechiness', 'tempo', 'key', 'time_signature']
        
    # add new feature
    song_df['avg'] = song_df['acousticness'] * song_df['valence']
        
    # scale tempo and loudness between 0, 1
    song_df['tempo'] = song_df['tempo'].values[0] / 246
    song_df['loudness'] = song_df['loudness'].values[0] / (-60)
        
    # load ML model
    with open('website/static/mood_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

        
    # predict mood from ML model
    song_mood = loaded_model.predict(song_df)
        
    # Recommendations
    dataframe = dataframe[(dataframe['moodd'] == song_mood[0])]
    recommend_df = dataframe[(dataframe['valence'].between(song_features['valence'] - scale, song_features['valence'] + scale)) &
                                 (dataframe['acousticness'].between(song_features['acousticness'] - scale, song_features['acousticness'] + scale)) &
                                 (dataframe['energy'].between(song_features['energy'] - scale, song_features['energy'] + scale)) &
                                 (dataframe['danceability'].between(song_features['danceability'] - scale, song_features['danceability'] + scale)) &
                                 (dataframe['loudness'].between(song_features['loudness'] - 10, song_features['loudness'] + 10)) &
                                 (dataframe['instrumentalness'].between(song_features['instrumentalness'] - scale, song_features['instrumentalness'] + scale))]\
                                 .sort_values('popularity', ascending=False)
    recommend_df = recommend_df.drop_duplicates(keep='first')
    
    if len(recommend_df) < 8:
        mood_df = dataframe[dataframe['moodd'] == song_mood[0]].sort_values('popularity', ascending=False).head(10).sample(8)
        recommend_df = pd.concat([recommend_df, mood_df]).iloc[0:8]
        recommend_df = recommend_df.sample(frac = 1)
    elif (len(recommend_df) >= 8) and (len(recommend_df) < 16):
        mood_df = dataframe[dataframe['moodd'] == song_mood[0]].sort_values('popularity', ascending=False).sample(8)
        recommend_df = pd.concat([recommend_df, mood_df]).iloc[0:8]
        recommend_df = recommend_df.sample(frac = 1)
    else:
        recommend_df = recommend_df.sort_values('popularity', ascending=False).head(16).sample(8)
        
    tracks_id = recommend_df['id'].iloc[0:8]
    
    return tracks_id, song_mood[0]   
   


def recommender_2(dataframe, link, mood, scale=0.1):
    
    # get song id
    song_id = link.split('/')[-1:][0].split('?')[0]
    
    # get song features from dataframe
    song_tag = dataframe[dataframe['id'] == song_id]['tag'].values[0]
    song_valence = dataframe[dataframe['id'] == song_id]['valence'].values[0]
    song_acousticness = dataframe[dataframe['id'] == song_id]['acousticness'].values[0]
    song_energy = dataframe[dataframe['id'] == song_id]['energy'].values[0]
    song_danceability = dataframe[dataframe['id'] == song_id]['danceability'].values[0]
    song_loudness = dataframe[dataframe['id'] == song_id]['loudness'].values[0]
    song_instrumentalness = dataframe[dataframe['id'] == song_id]['instrumentalness'].values[0]
    song_artists = dataframe[dataframe['id'] == song_id]['artists'].values[0]
    song_language = dataframe[dataframe['id'] == song_id]['language'].values[0]
        
        
    # recommendations based on similiar language
    similiar_language = dataframe[(dataframe['moodd'] == mood) 
                                     & (dataframe['tag'] == song_tag)
                                     & (dataframe['language'] == song_language)
                                     & (dataframe['valence'].between(song_valence - scale, song_valence + scale))
                                     & (dataframe['energy'].between(song_energy - scale, song_energy + scale)) 
                                     & (dataframe['danceability'].between(song_danceability - scale, song_danceability + scale))
                                     & (dataframe['loudness'].between(song_loudness - 10, song_loudness + 10))].sort_values('popularity', ascending=False)
    similiar_language = similiar_language.drop_duplicates(keep='first')
        
        
    if len(similiar_language) < 4:
        similiar_language = similiar_language.iloc[0:3]
    elif (len(similiar_language) >= 4) and (len(similiar_language) < 13):
        similiar_language = similiar_language.sample(4)
    else:
        similiar_language = similiar_language.head(12).sample(4)

        
    # other recommendations
    dataframe = dataframe[(dataframe['moodd'] == mood) & (dataframe['tag'] == song_tag)]
    others_df = dataframe[(dataframe['valence'].between(song_valence - scale, song_valence + scale)) &
                                 (dataframe['acousticness'].between(song_acousticness - scale, song_acousticness + scale)) &
                                 (dataframe['energy'].between(song_energy - scale, song_energy + scale)) &
                                 (dataframe['danceability'].between(song_danceability - scale, song_danceability + scale)) &
                                 (dataframe['loudness'].between(song_loudness - 10, song_loudness + 10)) &
                                 (dataframe['instrumentalness'].between(song_instrumentalness - scale, song_instrumentalness + scale))]\
                                 .sort_values('popularity', ascending=False)
        
    recommend_df = pd.concat([similiar_language, others_df], axis=0)
        

    # drop dublicates and user's song
    recommend_df = recommend_df.drop_duplicates(keep='first')
    song_index = dataframe[dataframe['id'] == song_id].index
    recommend_df = recommend_df.drop(song_index).iloc[0:8]
    recommend_df = recommend_df.sample(frac = 1)
        

    if len(recommend_df) < 8:
            mood_df = dataframe[dataframe['moodd'] == mood].sort_values('popularity', ascending=False).head(20).sample(10)
            recommend_df = pd.concat([recommend_df, mood_df]).iloc[0:8]
            recommend_df = recommend_df.sample(frac = 1)       
        
    tracks_id = recommend_df['id'].iloc[0:8]
    
    return tracks_id



# get tracks properties like artist, image, url, etc..
def get_tracks_properties(track_id):
    
    track_dict = {}
    # get track
    track = sp.track(track_id)
    # artist names
    total_artists = len(track['artists'])
    artist_names = ''
    for artist in range(total_artists):
        if artist == (total_artists - 1):
            artist_names += track['artists'][artist]['name']
        else:
            artist_names += track['artists'][artist]['name'] + ', '
    track_dict['artist_name'] = artist_names
    # track name
    track_dict['track_name'] = track['name']
    # album name
    track_dict['album_name'] = track['album']['name']
    # image name
    track_dict['image_url'] = track['album']['images'][0]['url']
    # spotify url
    track_dict['spotify_url'] = track['external_urls']['spotify']
        
    return track_dict
