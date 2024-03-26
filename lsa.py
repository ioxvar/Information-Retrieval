import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd

nltk.download('stopwords')
nltk.download('punkt')

xml_file = 'cran.all.1400.xml'
tree = ET.parse(xml_file)
root = tree.getroot()

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
documents = []
for doc in root.findall('doc'):
    title = doc.find('title').text.strip().lower() if doc.find('title') is not None and doc.find('title').text is not None else ''
    author = doc.find('author').text.strip().lower() if doc.find('author') is not None and doc.find('author').text is not None else ''
    bib = doc.find('bib').text.strip().lower() if doc.find('bib') is not None and doc.find('bib').text is not None else ''
    text = doc.find('text').text.strip().lower() if doc.find('text') is not None and doc.find('text').text is not None else ''

    title_tokens = [stemmer.stem(token) for token in word_tokenize(title) if token not in stop_words]
    author_tokens = [stemmer.stem(token) for token in word_tokenize(author) if token not in stop_words]
    bib_tokens = [stemmer.stem(token) for token in word_tokenize(bib) if token not in stop_words]
    text_tokens = [stemmer.stem(token) for token in word_tokenize(text) if token not in stop_words]

    doc_text = ' '.join(title_tokens + author_tokens + bib_tokens + text_tokens)
    documents.append(doc_text)

stop_words_list = list(stop_words)

#Vectorize using TF-IDF
vectorizer = TfidfVectorizer(stop_words=stop_words_list, tokenizer=lambda doc: doc.split())
tfidf_matrix = vectorizer.fit_transform(documents)

#LSA using SVD
n_components = 150  
lsa = TruncatedSVD(n_components=n_components)
document_topics = lsa.fit_transform(tfidf_matrix)

def preprocess_query(query):
    query_tokens = word_tokenize(query.lower())
    filtered_query_tokens = [word for word in query_tokens if word not in stop_words]
    return ' '.join(stemmer.stem(word) for word in filtered_query_tokens)

def encode_query(query):
    query_tfidf = vectorizer.transform([query])
    query_topics = lsa.transform(query_tfidf)
    return query_topics

query_file = 'cran.qry.xml'
queries_df = pd.read_xml(query_file)
queries_df.index += 1  
queries = list(zip(queries_df.index, queries_df['title']))

with open("checklsa.txt", "w") as output_file:
    for query_id, query_text in queries: 
        preprocessed_query = preprocess_query(query_text)
        query_topics = encode_query(preprocessed_query)
        similarities = cosine_similarity(query_topics, document_topics).flatten()
        ranked_documents = similarities.argsort()[::-1]
        
        for rank, doc_id in enumerate(ranked_documents, start=1):
            output_file.write(f"{query_id} Q0 {doc_id} {rank} {similarities[doc_id]} lsa\n")
