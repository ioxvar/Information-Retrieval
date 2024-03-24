from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# parse XML
def parse(data):
    docs = []
    tree = ET.parse(data)
    root = tree.getroot()
    for doc in root.findall('doc'):
        doc_dict = {}
        doc_dict['title'] = doc.find('title').text.strip().lower() if doc.find('title') is not None and doc.find('title').text is not None else ''
        doc_dict['author'] = doc.find('author').text.strip().lower() if doc.find('author') is not None and doc.find('author').text is not None else ''
        doc_dict['bib'] = doc.find('bib').text.strip().lower() if doc.find('bib') is not None and doc.find('bib').text is not None else ''
        doc_dict['text'] = doc.find('text').text.strip().lower() if doc.find('text') is not None and doc.find('text').text is not None else ''
        docs.append(doc_dict)
    return docs

data = 'cran.all.1400.xml'
elements = parse(data)
df = pd.DataFrame(elements)

#extracting query_list
def query_from(data):
    query_list = {}
    tree = ET.parse(data)
    root = tree.getroot()
    queries_list = root.findall('top')

    for index, query_from in enumerate(queries_list):
        query_text = query_from.find('title').text.strip()
        query_list[index + 1] = query_text

    return query_list

query_list = query_from('cran.qry.xml')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(title, author, bib, text):
    all_text = ' '.join([title, author, bib, text])
    tokens = word_tokenize(all_text.lower())
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return ' '.join(stemmed_tokens)

# Preprocess documents and query_list
df['preprocessed_text'] = df.apply(lambda row: preprocess_text(row['title'], row['author'], row['bib'], row['text']), axis=1)
preprocessed_queries = {query_id: preprocess_text(query_text, '', '', '') for query_id, query_text in query_list.items()}

# vectorizer parameters for better performance
# using trial and error, these parameters gave me the highest score 
tfidf_vectorizer = TfidfVectorizer(max_df=0.72, min_df=0.001, max_features=4422, ngram_range=(1, 2))

# Fit TF-IDF vectorizer on preprocessed documents
tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocessed_text'])

output_file = "checkvsm.txt"
with open(output_file, 'w') as f:
    for query_id, query_text in preprocessed_queries.items():
        # Transform the preprocessed query_from using the trained TF-IDF vectorizer
        query_vector = tfidf_vectorizer.transform([query_text])

        # cosine similarity between the query_from vector and document vectors
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)

        # Get the indices of documents sorted by similarity score
        sorted_indices = cosine_similarities.argsort()[0][::-1]

        #top 100 results
        for rank, idx in enumerate(sorted_indices[:100], start=1):
            docno = idx + 1  # Since docno starts from 1
            similarity = cosine_similarities[0][idx]
            f.write(f"{query_id} 0 {docno} {rank} {similarity} vsm\n")
 