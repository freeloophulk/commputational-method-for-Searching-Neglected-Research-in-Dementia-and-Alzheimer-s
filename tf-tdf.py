import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sentences = [
 "Dogs are people best friends.",
 "dogs are my best friend.",
 " dogs are the best.",
 "this sentence is just for something completely different."
]
def main():
   graph_sentences_by_similarity(sentences)

def get_tfidf_matrix(sentences):
   tfidf_vectorizer = TfidfVectorizer(stop_words="english")
   return tfidf_vectorizer.fit_transform(sentences)

def get_cosine_similarity_matrix():
   tfidf_matrix = get_tfidf_matrix(sentences)
   return cosine_similarity(tfidf_matrix)

def graph_sentences_by_similarity(sentences):

   G = nx.Graph()
   G.add_nodes_from(sentences)
   cosine_similarity_matrix = get_cosine_similarity_matrix()

   for row in range(0, len(cosine_similarity_matrix)):
     for col in range(0, len(cosine_similarity_matrix[0])):
       if row != col and cosine_similarity_matrix[row][col] > 0.5:
          G.add_edge(sentences[row], sentences[col])

   G.remove_nodes_from(list(nx.isolates(G)))
   nx.draw(
     G,
     with_labels=True,
    )
   plt.show()

if __name__ == "__main__":
 main()