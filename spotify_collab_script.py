#!/usr/bin/env python
# coding: utf-8

import re
import numpy as np 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


def load_and_preprocess_data():
    nodes_df = pd.read_csv('Datasets/Spotify Artist Feature Collaboration Network/reduced_nodes-10k.csv')
    edges_df = pd.read_csv('Datasets/Spotify Artist Feature Collaboration Network/edges.csv')

    edges_df = edges_df.rename(columns={'id_0': 'source', 'id_1': 'target'})

    spotify_ids = set(nodes_df['spotify_id'])
    sources = set(edges_df['source'])
    targets = set(edges_df['target'])

    missing_sources = sources.difference(spotify_ids)
    missing_targets = targets.difference(spotify_ids)

    edges_df = edges_df[~edges_df['source'].isin(missing_sources)]
    edges_df = edges_df[~edges_df['target'].isin(missing_targets)]

    scaler = MinMaxScaler()

    nodes_df["popularity_norm"] = MinMaxScaler().fit_transform(nodes_df[["popularity"]])

    nodes_df["log_followers"] = np.log(nodes_df["followers"])
    nodes_df["log_followers"] = nodes_df["log_followers"].replace([np.inf, -np.inf], np.nan)

    finite_max = nodes_df[nodes_df["log_followers"] != np.inf]["log_followers"].max()
    nodes_df["log_followers"] = nodes_df["log_followers"].fillna(finite_max)

    nodes_df["log_followers_norm"] = scaler.fit_transform(nodes_df[["log_followers"]])

    hit_counts = []
    for index, row in nodes_df.iterrows():
        chart_hits = eval(row['chart_hits']) if pd.notnull(row['chart_hits']) else []
        hit_count_sum = 0
        if chart_hits:
            for hit in chart_hits:
                _, hit_count = hit.split(' ')
                hit_count = int(hit_count[1:-1])  
                hit_count_sum += hit_count
        hit_counts.append(hit_count_sum)

    hit_counts_norm = scaler.fit_transform(np.array(hit_counts).reshape(-1, 1))

    nodes_df['hit_count_norm'] = hit_counts_norm

    return nodes_df, edges_df

def visualize_graph(G):
    plt.figure(figsize=(20,20))
    plt.xticks([])
    plt.yticks([])

    labels = {node: data['name'] for node, data in G.nodes(data=True)}

    colors = [data['popularity'] for node, data in G.nodes(data=True)]

    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), labels=labels, node_color=colors, node_size=100, with_labels=True, cmap=plt.cm.viridis, font_size=16)
    st.pyplot(plt)

def visualize_artist_graph(G, artist_name):
    artist_node = None
    for node, data in G.nodes(data=True):
        if data['name'] == artist_name:
            artist_node = node
            break

    if artist_node is None:
        print(f"No artist named '{artist_name}' was found in the graph.")
        return

    radius = 1  
    subgraph = nx.ego_graph(G, artist_node, radius=radius)

    visualize_graph(subgraph)

def create_graph(nodes_df, edges_df):
    G = nx.Graph()

    for index, row in nodes_df.iterrows():
        G.add_node(row['spotify_id'], 
                   name=row['name'], 
                   log_followers_norm=row['log_followers_norm'],
                   popularity_norm=row['popularity_norm'],
                   hit_count_norm=row['hit_count_norm'],
                   popularity=row['popularity'])

    for index, row in edges_df.iterrows():
        source_node = G.nodes[row['source']]
        target_node = G.nodes[row['target']]

        source_weight = source_node['log_followers_norm'] + (source_node['hit_count_norm'] * source_node['popularity_norm'])
        target_weight = target_node['log_followers_norm'] + (target_node['hit_count_norm'] * target_node['popularity_norm'])

        weight = source_weight + target_weight

        G.add_edge(row['source'], row['target'], weight=weight)

    return G

@st.cache_data
def load_data():
    return pd.read_csv("Datasets/30000 Spotify Songs/spotify_songs.csv")

@st.cache_data
def load_common_neighbors_data():
    return pd.read_csv("common_neighbor_centrality.csv")

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
        nodes_df, edges_df = load_and_preprocess_data()
        G = create_graph(nodes_df, edges_df)

        # New section for artist selection
        st.header("Spotify Artists Collaborations Prediction")
        artist_names = [data['name'] for node, data in G.nodes(data=True)]
        selected_artist = st.selectbox('Select an artist:', artist_names)
        #visualize_artist_graph(G, selected_artist)

        suggestions = common_neighbors_data["artist_name"].tolist()
        # selected_artist = st.selectbox("Select Artist", [""] + suggestions)
        # Display top 10 collaborations for selected artist
        if selected_artist:
            artist_data = common_neighbors_data[common_neighbors_data["artist_name"].str.strip().str.lower() == selected_artist.lower()]
            top_links = artist_data.iloc[0]["top_links"].split(", ")[:10]

            # Display the graph
            visualize_artist_graph(G, selected_artist)

            # In the sidebar, display the top 10 collaborations
            st.sidebar.markdown("### Top 10 Potential Collaborations:")
            for collaborator in top_links:
                st.sidebar.write(f"- {collaborator}")

        
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

    except FileNotFoundError:
        st.error("The file was not found")

if __name__ == "__main__":
    main()