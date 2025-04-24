import streamlit as st
import requests
import re
from textblob import TextBlob, Word
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
from gensim import corpora, models
import pandas as pd

# Download resources
nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

@st.cache_data
def download_hamlet():
    url = "https://www.gutenberg.org/files/1524/1524-0.txt"
    response = requests.get(url)
    return response.text

def clean_text(text):
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()

def preprocess_words(text):
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    blob = TextBlob(text)
    cleaned_words = []
    for word in blob.words:
        if word not in stop_words and len(word) > 2:
            lemmatized = Word(word).lemmatize()
            stemmed = ps.stem(lemmatized)
            cleaned_words.append(stemmed)
    return cleaned_words

def generate_wordcloud_from_words(words):
    text = " ".join(words)
    return WordCloud(width=800, height=400, background_color='white').generate(text)

def extract_named_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_keywords(text, num_keywords=10):
    words = [word for word in text.lower().split() if word.isalpha()]
    dictionary = corpora.Dictionary([words])
    corpus = [dictionary.doc2bow(words)]
    tfidf_model = models.TfidfModel(corpus)
    tfidf_scores = tfidf_model[corpus[0]]
    sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    return [(dictionary[id], round(score, 3)) for id, score in sorted_scores[:num_keywords]]

def run_topic_modeling(text, num_topics=3):
    words = [word for word in text.lower().split() if word.isalpha()]
    dictionary = corpora.Dictionary([words])
    corpus = [dictionary.doc2bow(words)]
    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=5)
    return lda.print_topics(num_words=5)

# Main app
def main():
    st.set_page_config(page_title="NLP Dashboard", layout="wide")
    st.title("🧠 NLP Dashboard with TextBlob, spaCy, and Gensim")

    # Sidebar for file upload and task selection
    st.sidebar.header("📂 Upload or Use Default Text")
    uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])
    
    st.sidebar.markdown("## 🔍 Select Task")
    run_sentiment = st.sidebar.checkbox("Sentiment Analysis")
    run_noun_phrases = st.sidebar.checkbox("Noun Phrase Extraction")
    run_wordcloud = st.sidebar.checkbox("Word Cloud")
    run_ner = st.sidebar.checkbox("Named Entity Recognition")
    # run_keywords = st.sidebar.checkbox("TF-IDF Keyword Extraction")
    # run_topics = st.sidebar.checkbox("Topic Modeling (LDA)")

    # Load and clean text
    if uploaded_file is not None:
        raw_text = uploaded_file.read().decode("utf-8")
        st.success("Custom file uploaded successfully.")
    else:
        raw_text = download_hamlet()
        st.info("Using default text: *Hamlet* by Shakespeare")

    cleaned_text = clean_text(raw_text)
    blob = TextBlob(cleaned_text)

    # Run selected tasks
    if run_sentiment:
        st.header("🧠 Sentiment of Sample Sentences")
        for i, sentence in enumerate(blob.sentences[:5]):
            st.markdown(f"**Sentence {i+1}**: {sentence}")
            st.write(f"→ Polarity: {sentence.sentiment.polarity:.2f}, Subjectivity: {sentence.sentiment.subjectivity:.2f}")

    if run_noun_phrases:
        st.header("📌 Noun Phrases")
        st.write(blob.noun_phrases[:20])

    if run_wordcloud:
        st.header("☁️ Word Cloud (Preprocessed)")
        words = preprocess_words(cleaned_text)
        wordcloud = generate_wordcloud_from_words(words)
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    if run_ner:
        st.header("🏷️ Named Entities (spaCy)")
        entities = extract_named_entities(cleaned_text[:5000])
        df = pd.DataFrame(entities, columns=["Text", "Entity Type"])
        st.dataframe(df)

    # if run_keywords:
    #     st.header("🔑 Top Keywords (TF-IDF via Gensim)")
    #     keywords = extract_keywords(cleaned_text)
    #     st.table(keywords)

    # if run_topics:
    #     st.header("📖 Topics (LDA)")
    #     topics = run_topic_modeling(cleaned_text)
    #     for i, topic in topics:
    #         st.markdown(f"**Topic {i+1}:** {topic}")

if __name__ == "__main__":
    main()
