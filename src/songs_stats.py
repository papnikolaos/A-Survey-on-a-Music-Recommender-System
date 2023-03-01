import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def top_songs(song_num):
    rating_sum_list = [0]*song_num
    rating_count_list = [0]*song_num
    with open("pipeline/train.txt","r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            rating_sum_list[int(line.split()[1])]+=int(line.split()[2])
            rating_count_list[int(line.split()[1])]+=1
    rating_avgs = [i / j for i, j in zip(rating_sum_list, rating_count_list)]
    for i in range(song_num):
        rating_avgs[i] = [i,rating_avgs[i]]
    rating_avgs.sort(key=lambda x: x[1],reverse=True)
    best_songs = [str(i[0]) for i in rating_avgs[:12]]
    best_song_ratings = [i[1] for i in rating_avgs[:12]]

    return best_songs,best_song_ratings

def top_artists(song_num):
    song_df = pd.read_csv('Dataset\song-attributes.txt', sep="\t", header=None)
    song_df.columns = ['song_id', 'album_id', 'artist_id', 'genre_id']
    artist_num = song_df['artist_id'].max()
    rating_sum_list = [0]*artist_num
    rating_count_list = [0]*artist_num

    with (open("ids_label_encoder.pkl", "rb")) as pkl:
        encoder = pickle.load(pkl)
    decoded_ids = encoder.inverse_transform(range(song_num))
    
    with open("pipeline/train.txt","r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            song_real_id = decoded_ids[int(line.split()[1])]
            artist_id = song_df[song_df['song_id'] == song_real_id]['artist_id'].values[0]
            rating_sum_list[artist_id]+=int(line.split()[2])
            rating_count_list[artist_id]+=1
    rating_avgs = [0 if j == 0 else  i / j for i, j in zip(rating_sum_list, rating_count_list)]
    
    for i in range(artist_num):
        rating_avgs[i] = [i,rating_avgs[i]]
    rating_avgs.sort(key=lambda x: x[1],reverse=True)
    best_artists = [str(i[0]) for i in rating_avgs[:12]]
    best_artists_ratings = [i[1] for i in rating_avgs[:12]]

    return best_artists,best_artists_ratings

def top_albums(song_num):
    song_df = pd.read_csv('Dataset\song-attributes.txt', sep="\t", header=None)
    song_df.columns = ['song_id', 'album_id', 'artist_id', 'genre_id']
    album_num = song_df['album_id'].max()
    rating_sum_list = [0]*album_num
    rating_count_list = [0]*album_num

    with (open("ids_label_encoder.pkl", "rb")) as pkl:
        encoder = pickle.load(pkl)
    decoded_ids = encoder.inverse_transform(range(song_num))
    
    with open("pipeline/train.txt","r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            song_real_id = decoded_ids[int(line.split()[1])]
            album_id = song_df[song_df['song_id'] == song_real_id]['album_id'].values[0]
            rating_sum_list[album_id]+=int(line.split()[2])
            rating_count_list[album_id]+=1
    rating_avgs = [0 if j == 0 else  i / j for i, j in zip(rating_sum_list, rating_count_list)]
    
    for i in range(album_num):
        rating_avgs[i] = [i,rating_avgs[i]]
    rating_avgs.sort(key=lambda x: x[1],reverse=True)
    best_albums = [str(i[0]) for i in rating_avgs[:12]]
    best_albums_ratings = [i[1] for i in rating_avgs[:12]]

    return best_albums,best_albums_ratings