# pip install pandas nltk matplotlib wordcloud regex seaborn streamlitç
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import nltk
import re
from wordcloud import WordCloud
import streamlit as st

@st.cache_data
def load_data():
    return pd.read_csv("Datasets/30000 Spotify Songs/spotify_songs.csv")

@st.cache_data
def load_common_neighbors_data():
    return pd.read_csv("common_neighbor_centrality1.csv")

def clean_words(song_titles):
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    words_list = [x.lower() for i in song_titles for x in nltk.word_tokenize(i)]
    words_alpha = [i for i in words_list if not re.match('^[^a-z]+$', i)]
    word_list_final = [i for i in words_alpha if not i in nltk_stopwords]
    remove_words = ["feat","remix","edit","version","radio","'s","mix","n't","u","'m"]
    word_list_final = [i for i in word_list_final if not i in remove_words]
    return word_list_final

def generate_wordcloud(word_list_final):
    token_str = ' '.join(word_list_final)
    song_title_wordcloud = WordCloud(background_color='black', margin=2, width=800, height=400).generate(token_str)
    return song_title_wordcloud

def generate_word_freq(word_list_final):
    feq_dist = nltk.FreqDist(word_list_final)
    feq_dist_df = feq_dist.most_common(10)
    songs_word_freq = pd.DataFrame(feq_dist_df,columns = ['Word','Count'])
    songs_word_freq.to_csv("songs_word_freq.csv")
    return songs_word_freq

def main():
    try:
        spotify_songs_data = load_data()
        common_neighbors_data = load_common_neighbors_data()
        
        # Get unique genres and add "All" as the first option
        genres = spotify_songs_data["playlist_genre"].unique().tolist()
        genres.insert(0, "All")

        st.title("Spotify Songs Analysis")
        # Create a dropdown for genre selection
        selected_genre = st.selectbox("Select Genre", genres)

        # Filter songs based on the selected genre
        if selected_genre != "All":
            spotify_songs_data = spotify_songs_data[spotify_songs_data["playlist_genre"] == selected_genre]

        song_titles = spotify_songs_data["track_name"].astype('str')
        word_list_final = clean_words(song_titles)
        song_title_wordcloud = generate_wordcloud(word_list_final)
        # songs_word_freq = generate_word_freq(word_list_final)

        st.image(song_title_wordcloud.to_array())


        # New section for artist selection
        st.header("Spotify Collaborations Prediction")

        suggestions = common_neighbors_data["artist_name"].tolist()
        selected_artist = st.selectbox("Select Artist", [""] + suggestions)

        # Display top 10 collaborations for selected artist
        if selected_artist:
            artist_data = common_neighbors_data[common_neighbors_data["artist_name"].str.strip().str.lower() == selected_artist.lower()]
            top_links = artist_data.iloc[0]["top_links"].split(", ")[:10]
            st.markdown("### Top 10 Potential Collaborations:")
            for collaborator in top_links:
                st.write(f"- {collaborator}")
    except FileNotFoundError:
        st.error("The file was not found")

if __name__ == "__main__":
    main()