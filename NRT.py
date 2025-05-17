# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 01:33:13 2025

@author: nitin
"""

# === Imports ===
import streamlit as st
import requests
import docx
import PyPDF2
import tempfile
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from newspaper import Article
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return tokens



# Download NLTK data (stopwords)
nltk.download('punkt')
nltk.download('stopwords')

# === API Keys ===
NEWS_API_KEY = "977ae5dcda884bcd81e33fcdaff53144"
GOOGLE_API_KEY = "AIzaSyDG-R8LJjnCtOxA0lCweZdo1Oos5u7AMb0"
GOOGLE_CSE_ID = "52b0feb1ebb9f4261"

# === Models ===
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# === Text Extraction ===
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text if text else "No extractable text found in the PDF."
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text if text else "No extractable text found in the DOCX file."
    except Exception as e:
        return f"Error extracting text from DOCX: {str(e)}"

# === Summarization ===
def summarize_text(text, ratio=0.3, max_words=200):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer('english'))
        sentences = list(parser.document.sentences)
        summary_length = max(1, int(len(sentences) * ratio))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentences_count=summary_length)
        summary_text = ' '.join(str(sentence) for sentence in summary)
        summary_words = summary_text.split()
        if len(summary_words) > max_words:
            summary_text = ' '.join(summary_words[:max_words]) + "..."
        return summary_text
    except Exception as e:
        return f"Summary could not be generated. Error: {str(e)}"

# === Google Search ===
@st.cache_data
def google_search(query, num_results=10):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}&num={num_results}&lr=lang_en"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad HTTP responses
        results = response.json().get('items', [])
        return [{"title": item['title'], "link": item['link'], "snippet": item.get('snippet', '')} for item in results]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Google Search results: {e}")
        return []

# === Fetch News (NewsAPI) ===
@st.cache_data
def fetch_news(api_key, query, page_size=20):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize={page_size}&sortBy=publishedAt&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad HTTP responses
        data = response.json()
        articles = data.get('articles', [])
        return [{"title": article['title'], "url": article['url']} for article in articles]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching NewsAPI results: {e}")
        return []

# === Extract article content ===
def extract_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return None

# === Text Preprocessing (Removing stopwords and stemming) ===
def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    # Remove stopwords and apply stemming
    tokens = [ps.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# === Relevance Scoring ===
def rank_documents(query, documents):
    # Preprocess the query and documents
    processed_query = preprocess_text(query)
    corpus = [preprocess_text(doc['title'] + ' ' + doc.get('snippet', '')) for doc in documents]

    # BM25
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(processed_query.split())

    # Cosine Similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus + [processed_query])
    cosine_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Combine
    combined_scores = 0.5 * bm25_scores + 0.5 * cosine_scores

    ranked_docs = [doc for _, doc in sorted(zip(combined_scores, documents), key=lambda pair: pair[0], reverse=True)]

    return ranked_docs

# === Display Helper ===
def display_and_summarize_items(items, key_prefix="item"):
    for idx, item in enumerate(items, 1):
        link = item['link'] if 'link' in item else item['url']

        with st.expander(f"{idx}. {item['title']}"):
            st.markdown(f"[🔗 Read Full Article]({link})", unsafe_allow_html=True)

            if st.button(f"Summarize {idx}", key=f"summarize_{key_prefix}_{idx}"):
                with st.spinner(f"Summarizing {item['title']}..."):
                    article_content = extract_article_content(link)
                    if article_content:
                        summary = summarize_text(article_content, max_words=200)
                        st.subheader("📝 Summary")
                        st.write(summary)
                    else:
                        st.error(f"Could not fetch content from the link: {link}")

# === Sidebar and Main Interface ===
st.title("📰 News Research Tool")

st.sidebar.title("Choose Action")
option = st.sidebar.selectbox("Select a function", ["Google Search", "Fetch News", "Upload Document"])

# === Google Search ===
if option == "Google Search":
    query = st.text_input("Enter your search query:")
    if st.button("Search"):
        results = google_search(query)
        if results:
            ranked_results = rank_documents(query, results)
            st.session_state['google_results'] = ranked_results

    if 'google_results' in st.session_state:
        display_and_summarize_items(st.session_state['google_results'], key_prefix="google")

# === Fetch News with Subcategories and User Query ===
elif option == "Fetch News":
    category = st.selectbox("Select News Category", ["World", "Politics", "Entertainment", "Sports", "Business", "Technology", "Health", "Science", "General"])
    user_query = st.text_input("Enter a specific topic (optional):")

    final_query = category
    if user_query:
        final_query += " " + user_query

    if st.button("Fetch News"):
        articles = fetch_news(NEWS_API_KEY, final_query)
        if articles:
            ranked_articles = rank_documents(final_query, articles)
            st.session_state['news_articles'] = ranked_articles

    if 'news_articles' in st.session_state:
        display_and_summarize_items(st.session_state['news_articles'], key_prefix="news")

# === Upload Document ===
elif option == "Upload Document":
    uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()

        if st.button("Summarize"):
            content = ""
            if file_type == "pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    content = extract_text_from_pdf(tmp.name)
            elif file_type == "docx":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                    tmp.write(uploaded_file.read())
                    content = extract_text_from_docx(tmp.name)
            elif file_type == "txt":
                content = uploaded_file.read().decode("utf-8")

            if content:
                st.subheader("📄 Extracted Text")
                st.text_area("Extracted Content", content[:5000], height=250)

                st.subheader("📝 Summary")
                summary = summarize_text(content, max_words=200)
                st.write(summary)