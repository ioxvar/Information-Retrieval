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
idf_values = {term: math.log(document_count / term_counts[term]) for term in term_counts}

# Function to preprocess query
def preprocess_query(query):
    query_tokens = word_tokenize(query.lower())
    filtered_query_tokens = [word for word in query_tokens if word not in stop_words]
    return [stemmer.stem(word) for word in filtered_query_tokens]

# Function to calculate Query Likelihood score for a document
def calculate_ql_score(document, preprocessed_query):
    score = 0.0
    doc_term_freq = Counter(document)
    for term in preprocessed_query:
        if term not in document:
            continue
        term_frequency = doc_term_freq[term]
        score += math.log(term_frequency) if term_frequency > 0 else 0
    return score

# Function to calculate Query Likelihood score for a document with Dirichlet smoothing
def calculate_ql_score_dirichlet(document, preprocessed_query, mu=400):
    score = 0.0
    doc_term_freq = Counter(document)
    total_terms_in_doc = len(document)
    for term in preprocessed_query:
        if term not in idf_values:  # Check if the term exists in the idf_values dictionary
            continue
        term_frequency = doc_term_freq[term]
        # Apply Dirichlet smoothing
        numerator = term_frequency + mu * idf_values[term]
        denominator = total_terms_in_doc + mu
        score += math.log(numerator / denominator)
    return score

# Function to calculate Query Likelihood score for a document with JM smoothing
def calculate_ql_score_jm(document, preprocessed_query, lambda_=0.7):
    score = 0.0
    doc_term_freq = Counter(document)
    total_terms_in_doc = len(document)
    for term in preprocessed_query:
        if term not in idf_values:  # Check if the term exists in the idf_values dictionary
            continue
        term_frequency = doc_term_freq[term]
        # Apply JM smoothing
        numerator = lambda_ * term_frequency + (1 - lambda_) * idf_values[term]
        # Check if the denominator is zero to avoid division by zero
        if total_terms_in_doc == 0:
            continue
        score += math.log(numerator / total_terms_in_doc)
    return score

# Read queries from cran.qry.xml
query_file = 'cran.qry.xml'
queries_df = pd.read_xml(query_file)
queries_df.index += 1  # Increment index by 1 to start from 1
queries = list(zip(queries_df.index, queries_df['title']))

# Write results to file without smoothing
with open("checkql.txt", "w") as output_file:
    for query_id, query_text in queries:
        preprocessed_query = preprocess_query(query_text)
        scores = []
        for i, document in enumerate(preprocessed_documents, start=1):
            score = calculate_ql_score(document, preprocessed_query)
            scores.append((i, score))
        ranked_documents = sorted(scores, key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(ranked_documents, start=1):
            output_file.write(f"{query_id} Q0 {doc_id} {rank} {score} QL_NoSmoothing\n")

# Write results to file with Dirichlet smoothing
with open("checkql_dirichlet.txt", "w") as output_file:
    for query_id, query_text in queries:
        preprocessed_query = preprocess_query(query_text)
        scores = []
        for i, document in enumerate(preprocessed_documents, start=1):
            score = calculate_ql_score_dirichlet(document, preprocessed_query)
            scores.append((i, score))
        ranked_documents = sorted(scores, key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(ranked_documents, start=1):
            output_file.write(f"{query_id} Q0 {doc_id} {rank} {score} QL_Dirichlet\n")

# Write results to file with JM smoothing
with open("checkql_jm.txt", "w") as output_file:
    for query_id, query_text in queries:
        preprocessed_query = preprocess_query(query_text)
        scores = []
        for i, document in enumerate(preprocessed_documents, start=1):
            score = calculate_ql_score_jm(document, preprocessed_query)
            scores.append((i, score))
        ranked_documents = sorted(scores, key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(ranked_documents, start=1):
            output_file.write(f"{query_id} Q0 {doc_id} {rank} {score} QL_JM\n")