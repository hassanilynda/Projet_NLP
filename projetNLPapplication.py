import streamlit as st
import pandas as pd
import nltk
from gensim.models import Word2Vec
import numpy as np
from numpy import dot
from numpy.linalg import norm
import os

# Télécharger les stopwords de NLTK si nécessaire
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Charger les fichiers enregistrés
MODEL_PATH = "word2vec_model.model"
EMBEDDINGS_PATH = "documents_embeddings.pkl"
BBC_TRAIN_PATH = "BBC_News_Train.csv"

def load_model():
    if os.path.exists(MODEL_PATH):
        return Word2Vec.load(MODEL_PATH)
    return None

def load_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        return pd.read_pickle(EMBEDDINGS_PATH)
    return None

def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def most_similar_documents(vec_document, data, doc_id=None, n=5):
    distances_document = {}
    for doc in data["ArticleId"]:
        # Exclure le document comparé (doc_id) de la liste des similaires
        if doc == doc_id:
            continue
        vec_autre_doc = data[data["ArticleId"] == doc]["moy"].values[0]
        distances_document[doc] = cosine_similarity(vec_document, vec_autre_doc)
    
    # Trier les documents par similarité et garder les n meilleurs
    similaires = sorted(distances_document.items(), key=lambda item: item[1], reverse=True)[:n]
    return similaires

# Charger les données et le modèle
model = load_model()
data = load_embeddings()
bbc_train = pd.read_csv(BBC_TRAIN_PATH, sep=',', names=["ArticleId", "Text", "Category"])

# Interface utilisateur
st.title("Analyse de similarité de documents")

# Option pour choisir le mode d'entrée
option = st.radio(
    "Choisissez une option :",
    ("Écrire un texte à analyser", "Uploader un fichier texte contenant des documents", "Entrer un ID de document")
)

if option == "Écrire un texte à analyser":
    input_text = st.text_area("Entrez votre texte :")

    if st.button("Analyser le texte"):
        if input_text.strip():
            if model is None or data is None:
                st.error("Le modèle ou les embeddings ne sont pas disponibles. Veuillez les préparer.")
            else:
                # Prétraiter le texte d'entrée
                input_tokens = nltk.word_tokenize(input_text.lower())
                stopwords_list = nltk.corpus.stopwords.words('english')
                filtered_tokens = [token for token in input_tokens if token.isalpha() and token not in stopwords_list]

                if filtered_tokens:
                    # Calculer la moyenne des vecteurs pour le texte d'entrée
                    input_vectors = [model.wv[token] for token in filtered_tokens if token in model.wv]
                    
                    st.write(len(input_vectors))
                    
                    if input_vectors:
                        input_moy = np.mean(input_vectors, axis=0)
                    else:
                        input_moy = np.zeros(model.vector_size)
                    
                    # Trouver les documents similaires (en excluant l'ID fictif)
                    similaires = most_similar_documents(input_moy, data, doc_id=None, n=5)

                    st.write("### Documents similaires :")
                    for doc_id, similarity in similaires:
                        texte_original = bbc_train[bbc_train["ArticleId"] == doc_id]["Text"].values[0]
                        st.write(f"**Document ID:** {doc_id}, **Similarité:** {similarity:.4f}")
                        st.write(f"**Texte:** {texte_original}")
                else:
                    st.warning("Le texte ne contient pas de mots exploitables. Essayez un texte différent.")
        else:
            st.warning("Le texte ne peut pas être vide.")

elif option == "Uploader un fichier texte contenant des documents":
    uploaded_file = st.file_uploader("Choisissez un fichier texte contenant des documents", type="txt")

    if uploaded_file:
        # Lire le contenu du fichier texte
        file_content = uploaded_file.read().decode("utf-8")
        st.write("### Aperçu du fichier chargé :")
        st.text(file_content[:500])  # Afficher un aperçu du contenu du fichier (les 500 premiers caractères)

        # Séparer le fichier en lignes (si chaque ligne représente un document)
        documents = file_content.split("\n")
        
        # Analyse de chaque document
        if st.button("Analyser les documents du fichier"):
            if model is None or data is None:
                st.error("Le modèle ou les embeddings ne sont pas disponibles. Veuillez les préparer.")
            else:
                for doc_idx, document in enumerate(documents):
                    document = document.strip()
                    if document:
                        # Prétraiter chaque document
                        input_tokens = nltk.word_tokenize(document.lower())
                        stopwords_list = nltk.corpus.stopwords.words('english')
                        filtered_tokens = [token for token in input_tokens if token.isalpha() and token not in stopwords_list]

                        if filtered_tokens:
                            # Calculer la moyenne des vecteurs pour le document
                            input_vectors = [model.wv[token] for token in filtered_tokens if token in model.wv]
                            
                            if input_vectors:
                                input_moy = np.mean(input_vectors, axis=0)
                            else:
                                input_moy = np.zeros(model.vector_size)
                            
                            # Trouver les documents similaires
                            similaires = most_similar_documents(input_moy, data, doc_id=None, n=5)

                            st.write(f"### Résultats pour le document {doc_idx + 1}:")
                            st.write(f"**Texte du document:** {document}")
                            for doc_id, similarity in similaires:
                                texte_similaire = bbc_train[bbc_train["ArticleId"] == doc_id]["Text"].values[0]
                                st.write(f"**Document ID:** {doc_id}, **Similarité:** {similarity:.4f}")
                                st.write(f"**Texte similaire:** {texte_similaire}")
                        else:
                            st.write(f"Le document {doc_idx + 1} ne contient pas de mots exploitables.")
        else:
            st.warning("Veuillez télécharger un fichier texte contenant des documents.")
            
elif option == "Entrer un ID de document":
    st.subheader("Entrez l'ID du document que vous souhaitez analyser.")
    
    # Entrée de l'ID du document
    doc_id_input = st.text_input("ID du document", "")
    
    if st.button("Analyser l'ID"):
        if doc_id_input.isdigit():
            doc_id_input = int(doc_id_input)
            if doc_id_input in data["ArticleId"].values:
                # Récupérer l'embedding du document en fonction de l'ID
                doc_embedding = data[data["ArticleId"] == doc_id_input]["moy"].values[0]
                st.write(f"### Document ID: {doc_id_input}")
                
                # Afficher le texte du document
                texte_original = bbc_train[bbc_train["ArticleId"] == doc_id_input]["Text"].values[0]
                st.write(f"**Texte du document:** {texte_original}")
                
                # Trouver les documents similaires
                similaires = most_similar_documents(doc_embedding, data, doc_id_input, n=5)

                st.write("### Documents similaires :")
                for similar_doc_id, similarity in similaires:
                    texte_similaire = bbc_train[bbc_train["ArticleId"] == similar_doc_id]["Text"].values[0]
                    st.write(f"**Document ID:** {similar_doc_id}, **Similarité:** {similarity:.4f}")
                    st.write(f"**Texte similaire:** {texte_similaire}")
            else:
                st.warning("L'ID du document n'existe pas dans les données.")
        else:
            st.warning("Veuillez entrer un ID de document valide.")
