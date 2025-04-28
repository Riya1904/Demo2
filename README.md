import pandas as pd
1.Load the datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')
2.Preprocess genres and tags for content-based filtering
movies['genres'] = movies['genres'].fillna('')
tags['tag'] = tags['tag'].fillna('')
3.Group the tags by movie 
tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x))
movies = movies.merge(tags_grouped, on='movieId', how='left')
4.Combine genres and tags
movies['content'] = movies['genres'] + ' ' + movies['tag'].fillna('')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
5.Define the thresholds for cosine similarity
theta1 = 0.7
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import numpy as np
6.Generate similarity matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])
7.Cosine Similarity Matrix
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
8.Recommending similar movies
def recommend_content_based(title, sim_matrix):
    idx = movies.index[movies['title'] == title][0]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]
print(recommend_content_based('Toy Story (1995)', cosine_sim_matrix)

import torch
import networkx as nx
import pandas as pd
9.Encode users and movies with unique IDs
user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()
user_id_map = {id_: i for i, id_ in enumerate(user_ids)}
movie_id_map = {id_: i for i, id_ in enumerate(movie_ids)}
10.Convert user-movie interactions into edges
user_movie_edges = [(user_id_map[uid], len(user_ids) + movie_id_map[mid]) for uid, mid in zip(ratings['userId'], ratings['movieId'])]
11. Create a NetworkX graph
user_movie_graph = nx.Graph()
user_movie_graph.add_edges_from(user_movie_edges)
12.Get the number of nodes (sum of unique users and movies)
num_nodes = len(user_ids) + len(movie_ids)
13.Create node labels tensor: 0 for users, 1 for movies
labels = torch.cat([torch.zeros(len(user_ids)), torch.ones(len(movie_ids))])
14. Display the graph and labels
print("Graph edges:", user_movie_graph.edges())
print("Labels tensor:", labels)
15.Convert user-movie interactions into edges
user_movie_edges = [(user_id_map[uid], len(user_ids) + movie_id_map[mid]) for uid, mid in zip(ratings['userId'], ratings['movieId'])]
num_user_nodes = len(user_ids)
num_movie_nodes = len(movie_ids)
16.Create adjacency matrix for user-movie interactions
adj_matrix = torch.zeros((num_user_nodes + num_movie_nodes, num_user_nodes + num_movie_nodes))
for u, v in user_movie_edges:
    adj_matrix[u, v] = 1
    adj_matrix[v, u] = 1 
17. Define GraphSAGE layer w
class SAGEConv(nn.Module):
    def __init__(self, in_feats, out_feats, aggregator_type='mean'):
        super(SAGEConv, self).__init__()
        self.fc = nn.Linear(in_feats * 2, out_feats)

    def forward(self, adj_matrix, h):
        h_neigh = torch.matmul(adj_matrix, h)
        h_cat = torch.cat([h, h_neigh], dim=1)
        return self.fc(h_cat)
18. Define the GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats)
        self.conv2 = SAGEConv(hidden_feats, out_feats)

    def forward(self, adj_matrix, inputs):
        h = F.relu(self.conv1(adj_matrix, inputs))
        h = self.conv2(adj_matrix, h)
        return h
19. Initialize node embeddings and model
input_dim = 128  # Example dimension for initial embeddings
node_features = torch.rand(num_user_nodes + num_movie_nodes, input_dim)  # Random node features
model = GraphSAGE(in_feats=input_dim, hidden_feats=64, out_feats=32)
with torch.no_grad():
embeddings = model(adj_matrix, node_features)
20.Function to recommend based on user-movie embedding similarity
def recommend_collaborative(user_id, k=10):
    user_idx = user_id_map[user_id]
    user_embedding = embeddings[user_idx]
    movie_embeddings = embeddings[num_user_nodes:]
    similarities = F.cosine_similarity(user_embedding.unsqueeze(0), movie_embeddings)
    recommended_movie_indices = similarities.topk(k).indices
    return movies.iloc[recommended_movie_indices.numpy()]['title']
21.Example recommendation for user_id = 1
print(recommend_collaborative(1))  

