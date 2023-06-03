import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity 
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot_tool
from sentence_transformers import SentenceTransformer
from sentence_transformers import models

df = pd.read_csv('Dataset1.csv',usecols = ['TI'],encoding='ISO-8859-1') 

df_sample = df.sample(n=10000)


sentences = df_sample['TI'].tolist()

def main():    
 graph_sentences_by_similarity(sentences)

def get_sentences_embeddings(sentences):
     sentence_embeddings = SentenceTransformer('bert-base-nli-mean-tokens')
     return sentence_embeddings.encode(sentences)

def get_cosine_similarity_matrix(sentences):
     embedding_matrix = get_sentences_embeddings(sentences)
     return cosine_similarity(embedding_matrix)

 
def graph_sentences_by_similarity(sentences):
     
     G = nx.Graph()
     G.add_nodes_from(sentences)    
   
     cosine_similarity_matrix = get_cosine_similarity_matrix(sentences)
    
     

     for row in range(0, len(cosine_similarity_matrix)):
        for col in range(0, len(cosine_similarity_matrix[0])):
            if row != col and cosine_similarity_matrix[row][col] > 0.8:
                G.add_edge(sentences[row], sentences[col]) 

     G.remove_nodes_from(list(nx.isolates(G)))
     nx.draw(
        G,
        node_size=10, node_color='blue',
        with_labels = False,
    )
     plt.show()

if __name__ == "__main__":
 main()
    
