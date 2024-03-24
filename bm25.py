import math
from collections import Counter
import pandas as pd
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load XML data into a DataFrame
xml_file = 'cran.all.1400.xml'
tree = ET.parse(xml_file)
root = tree.getroot()

# Preprocess documents
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
preprocessed_documents = []
all_doc_tokens = []  # Store all document tokens for IDF calculation
for doc in root.findall('doc'):
    title = doc.find('title').text.strip().lower() if doc.find('title') is not None and doc.find('title').text is not None else ''
    author = doc.find('author').text.strip().lower() if doc.find('author') is not None and doc.find('author').text is not None else ''
    bib = doc.find('bib').text.strip().lower() if doc.find('bib') is not None and doc.find('bib').text is not None else ''
    text = doc.find('text').text.strip().lower() if doc.find('text') is not None and doc.find('text').text is not None else ''

    # Tokenize, stem, and remove stop words for title, author, bib, and text
    title_tokens = [stemmer.stem(token) for token in word_tokenize(title) if token not in stop_words]
    author_tokens = [stemmer.stem(token) for token in word_tokenize(author) if token not in stop_words]
    bib_tokens = [stemmer.stem(token) for token in word_tokenize(bib) if token not in stop_words]
    text_tokens = [stemmer.stem(token) for token in word_tokenize(text) if token not in stop_words]

    # Combine tokens for title, author, bib, and text into a single list
    doc_tokens = title_tokens + author_tokens + bib_tokens + text_tokens
    preprocessed_documents.append(doc_tokens)
    all_doc_tokens.extend(set(doc_tokens))  # Use set for faster lookup and remove duplicates

# Calculate IDF for all terms
term_counts = Counter(all_doc_tokens)
document_count = len(preprocessed_documents)
idf_values = {term: math.log((document_count - term_counts[term] + 0.5) / (term_counts[term] + 0.5) + 1) for term in term_counts}

# Function to preprocess query
def preprocess_query(query):
    query_tokens = word_tokenize(query.lower())
    filtered_query_tokens = [word for word in query_tokens if word not in stop_words]
    return [stemmer.stem(word) for word in filtered_query_tokens]


def calculate_bm25_score(document, preprocessed_query, document_length, avg_document_length, k1, b):
    score = 0.0
    doc_term_freq = Counter(document)
    for term in preprocessed_query:
        if term not in document:
            continue
        term_frequency = doc_term_freq[term]
        numerator = term_frequency * (k1 + 1)
        denominator = term_frequency + k1 * (1 - b + b * (document_length / avg_document_length))
        score += idf_values[term] * (numerator / denominator)
    return score

# Read queries from cran.qry.xml
query_file = 'cran.qry.xml'
queries_df = pd.read_xml(query_file)
queries_df.index += 1  # Increment index by 1 to start from 1
queries = list(zip(queries_df.index, queries_df['title']))

# BM25 parameters
avg_document_length = sum(len(doc) for doc in preprocessed_documents) / document_count
k1 = 3
b = 0.6

# Write results to file
with open("checkbm.txt", "w") as output_file:
    for query_id, query_text in queries:
        preprocessed_query = preprocess_query(query_text)
        scores = []
        for i, document in enumerate(preprocessed_documents, start=1):
            score = calculate_bm25_score(document, preprocessed_query, len(document), avg_document_length, k1, b)
            scores.append((i, score))
        ranked_documents = sorted(scores, key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(ranked_documents, start=1):
            output_file.write(f"{query_id} 0 {doc_id} {rank} {score} bm25\n")
