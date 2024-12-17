import networkx as nx
from itertools import combinations
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()

# Define stopwords
CUSTOM_STOPWORDS = set(["what", "is", "the", "of", "and", "in", "role", "a", "to", "for", "are", "have", "has", "do", "does", "not", "didn't",])
stop_words = set(stopwords.words("english")).union(CUSTOM_STOPWORDS)

# Load Contextual Embeddings Model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower()) 
    return [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]

def generate_ngrams(words, n=3):
    ngrams = []
    for i in range(1, n + 1):
        ngrams.extend([" ".join(words[j:j + i]) for j in range(len(words) - i + 1)])
    return ngrams


def build_graph_with_embeddings(ngrams, embeddings, window_size=3):
    graph = nx.Graph()

    for ngram in ngrams:
        graph.add_node(ngram)

    for i in range(len(ngrams) - window_size + 1):
        window = ngrams[i:i + window_size]
        for ngram1, ngram2 in combinations(window, 2):
            if graph.has_edge(ngram1, ngram2):
                graph[ngram1][ngram2]['weight'] += 1
            else:
                graph.add_edge(ngram1, ngram2, weight=1)

    for ngram1, ngram2 in graph.edges:
        similarity = util.cos_sim(embeddings[ngram1], embeddings[ngram2]).item()
        graph[ngram1][ngram2]['weight'] += similarity

    return graph

def textrank_with_embeddings(graph, top_n=10):
    scores = nx.pagerank(graph, weight='weight')
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [ngram for ngram, score in sorted_scores[:top_n]]

def extract_keywords_advanced_with_relevance(query, corpus, ngram_range=2, top_n=5, relevance_threshold=0.7):
    query_words = preprocess_text(query)
    corpus_words = [preprocess_text(doc) for doc in corpus]

    query_ngrams = generate_ngrams(query_words, n=ngram_range)
    corpus_ngrams = [generate_ngrams(words, n=ngram_range) for words in corpus_words]
    all_ngrams = query_ngrams + [ngram for doc in corpus_ngrams for ngram in doc]

    unique_ngrams = list(set(all_ngrams))
    ngram_embeddings = {ngram: embedding_model.encode(ngram) for ngram in unique_ngrams}

    graph = build_graph_with_embeddings(unique_ngrams, ngram_embeddings, window_size=3)
    initial_keywords = textrank_with_embeddings(graph, top_n=top_n)

    
    query_embedding = embedding_model.encode(query)
    relevant_keywords = []
    for keyword in initial_keywords:
        similarity = util.cos_sim(query_embedding, ngram_embeddings[keyword]).item()
        if similarity >= relevance_threshold:
            relevant_keywords.append(keyword)

    relevant_keywords.extend([kw for kw in query_ngrams if kw not in relevant_keywords])
    final_keywords = [kw for kw in relevant_keywords if len(kw.split()) > 1]

    return final_keywords

query = "AI developer in noida with 3 years of experience skilled in ML, genAI, LLM, langchain."
corpus = [
    "OpenAI develops cutting-edge artificial intelligence solutions.",
    "Healthcare applications of AI include predictive modeling and diagnosis.",
    "Machine learning and data science are integral to automation.",
    "Deep learning advances have transformed the field of computer vision.",
]

# Extract Keywords with Relevance
keywords = extract_keywords_advanced_with_relevance(query, corpus, ngram_range=2, top_n=5)
print("Extracted Relevant Keywords:", keywords)

