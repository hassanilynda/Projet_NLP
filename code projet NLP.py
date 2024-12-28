import nltk
from nltk.corpus import stopwords
import pandas as pd

data = pd.read_csv(r'BBC_News_Train.csv',sep=',',names=["ArticleId","Text","Category"])

fichier_trie = data.sort_values(by="ArticleId",ascending=True)

fichier_trie.to_csv("BBC_trie",index=False)



def filtre(df):
    
    df['Text'] = df['Text'].str.lower()
    
    df['Text'] = [nltk.word_tokenize(val) for val in df['Text']]

    stopwords = nltk.corpus.stopwords.words('english')
    
    df['Text'] = df['Text'].apply(
    lambda text: [
        token for token in text if token.isalpha() and token not in stopwords
    ]
)

    return df


filtre(data)

print(data[:1])


## dictionnaire lexicale 

## regrouper les mots dans un dictionnaire , ou chaque mot sera represente par son nombre de fois ou il apparait dans 
## les documents, id du document et sa representation vectorielle 

from gensim.models import Word2Vec

phrases = [token for token in data['Text']]


model = Word2Vec(sentences=phrases,vector_size=100,window=5,min_count=3)


dictio={}



pd.set_option('display.max_colwidth', 20)

print(data.head(0))

for ligne in data.values:  ##parcourir les lignes
    for token in ligne[1]:
        if (token not in dictio and token in model.wv) : ## si mot pas dans dictio et dans model
            dictio[token]=[1,[ligne[0]],model.wv[token]]
        elif token in dictio and token in model.wv:  ## si mot dans dictio et dans model
            dictio[token][0]+=1
            if ligne[0] not in dictio[token][1] : 
                dictio[token][1].append(ligne[0])


print(len(dictio))

from itertools import islice

# Afficher les 10 premiers éléments du dictionnaire

##for key, value in islice(dictio.items(), 10):
    ##print(key, value)


## representer chaque document en fonction de ses mots 

## parcours du dictionnaire et regrouper les mots d'un document pour choisir sa categorie 

import numpy as np
from numpy import dot
from numpy.linalg import norm



data['moy'] = [[] for _ in range(len(data))]

print(data[:2])

for idx in range(len(data)):
    id=data.iloc[idx]['ArticleId']
    tab=[]
    for mot in dictio:
        if id in  dictio[mot][1] :
            tab.append(dictio[mot][2])
    moy = np.mean(tab,axis=0)
    data.at[idx,'moy'] = moy

##print(data)


from numpy import dot
from numpy.linalg import norm

## calcul de similarite entre moy des textes et mots presents dans le texte 


## fonction cos pour calculer angle entre les 2 vecteurs 
def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))



##documents les plus similaires pour un document donne :
document = 1833


distances_document = {}

##print(data[data["ArticleId"]==document]["moy"])


def most_similar_documents(document,data,n=5):
    distances_document={}
    similaires = []

    vec_document = data[data["ArticleId"]==document]["moy"].values[0]
    
    for doc in data["ArticleId"]:
        if(doc!=document) :
            vec_autre_doc = data[data["ArticleId"]==doc]["moy"].values[0]
            distances_document[doc] = cosine_similarity(vec_document,vec_autre_doc)
    

    similaires = sorted(distances_document.items(),key = lambda item: item[1],reverse=True)[:n]

    return similaires


print(most_similar_documents(document,data,))

model.save("word2vec_model.model") ## sauvegarde du model dans un fichier
data.to_pickle("documents_embeddings.pkl")






