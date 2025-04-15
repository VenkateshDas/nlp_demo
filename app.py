import logging
import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import re
import plotly.express as px
import numpy as np
import time
import openai
from openai import OpenAI
import gensim
import gensim.corpora as corpora
from gensim.models import LdaMulticore
from nltk.corpus import stopwords
import os
from dotenv import load_dotenvgpt

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide", page_title="NLP Demo")
logger.info("Streamlit app started.")
load_dotenv()
logger.info("Environment variables loaded.")

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.max_length = 2000000
    logger.info("spaCy model 'en_core_web_sm' loaded.")
except OSError:
    logger.error("SpaCy model 'en_core_web_sm' not found.")
    st.error("SpaCy model 'en_core_web_sm' not found. Please run: `python -m spacy download en_core_web_sm`")
    st.stop()

stop_words = stopwords.words('english')
logger.info("NLTK stopwords loaded.")

def safe_float_convert(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

@st.cache_data
def preprocess_for_gensim(text):
    if not isinstance(text, str): text = str(text)
    result = gensim.utils.simple_preprocess(text, deacc=True)
    result = [word for word in result if word not in stop_words]
    return result

@st.cache_data
def preprocess_for_sklearn(text):
    if not isinstance(text, str): text = str(text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop and not token.is_punct and not token.is_space
    ]
    return " ".join(tokens) if tokens else ""

api_key_env = os.getenv("OPENAI_API_KEY")
if api_key_env:
    logger.info("OpenAI API key loaded from environment.")
else:
    logger.warning("OpenAI API key not found in environment.")
st.sidebar.header("🔑 OpenAI Configuration")
# Remove the text input for API key; only use environment variable
api_key_input = api_key_env

client = None
if api_key_input:
    try:
        client = OpenAI(api_key=api_key_input)
        st.sidebar.success("OpenAI client initialized.", icon="✅")
        logger.info("OpenAI client initialized.")
    except Exception as e:
        st.sidebar.error(f"Failed to initialize OpenAI client: {e}", icon="❌")
        logger.error(f"Failed to initialize OpenAI client: {e}")
        client = None
else:
    st.sidebar.warning("OpenAI API Key needed for sentiment analysis. Please set the OPENAI_API_KEY environment variable or .env file.", icon="⚠️")
    logger.warning("OpenAI API Key needed for sentiment analysis.")

@st.cache_data(show_spinner=False)
def get_sentiment_openai(_client, text):
    if not _client:
        logger.error("OpenAI client not initialized for sentiment analysis.")
        return "Error: OpenAI client not initialized.", 0.0, 0.0, "API key missing or invalid."
    if not isinstance(text, str) or len(text.strip()) < 10:
        logger.warning("Input text too short for sentiment analysis.")
        return "Neutral", 0.0, 0.5, "Input text too short."
    try:
        response = _client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[
                {"role": "system", "content": "You are an expert sentiment analysis assistant. Analyze the sentiment of the following text. Respond ONLY with the sentiment category (Positive, Negative, or Neutral) followed by a comma, then a sentiment polarity score (float between -1.0 and 1.0), then a comma, then a subjectivity score (float between 0.0 and 1.0). Example: Positive, 0.8, 0.75"},
                {"role": "user", "content": text[:1000]}
            ],
            temperature=0.2,
            max_tokens=30,
            n=1,
            stop=None,
        )
        result_text = response.choices[0].message.content.strip()
        parts = result_text.split(',')
        if len(parts) == 3:
            sentiment = parts[0].strip().capitalize()
            polarity = safe_float_convert(parts[1].strip(), 0.0)
            subjectivity = safe_float_convert(parts[2].strip(), 0.5)
            if sentiment not in ["Positive", "Negative", "Neutral"]:
                sentiment = "Neutral"
            polarity = max(-1.0, min(1.0, polarity))
            subjectivity = max(0.0, min(1.0, subjectivity))
            logger.info(f"Sentiment analysis result: {sentiment}, polarity: {polarity}, subjectivity: {subjectivity}")
            return sentiment, polarity, subjectivity, "Analysis successful."
        else:
            sentiment_simple = "Neutral"
            if "positive" in result_text.lower(): sentiment_simple = "Positive"
            elif "negative" in result_text.lower(): sentiment_simple = "Negative"
            logger.warning("Could not parse structured response from OpenAI, used simple classification.")
            return sentiment_simple, 0.0, 0.5, "Could not parse structured response, used simple classification."
    except openai.AuthenticationError:
         logger.error("OpenAI Authentication Error: Invalid API Key provided.")
         st.error("OpenAI Authentication Error: Invalid API Key provided.", icon="🚨")
         return "Error", 0.0, 0.0, "Invalid API Key."
    except openai.RateLimitError:
        logger.warning("OpenAI Rate Limit Error. Waiting before retry.")
        st.warning("OpenAI Rate Limit Error. Please wait and try again.", icon="⏳")
        time.sleep(5)
        return "Error", 0.0, 0.0, "Rate limit hit."
    except Exception as e:
        logger.error(f"OpenAI API Error: {e}")
        st.error(f"OpenAI API Error: {e}", icon="🚨")
        return "Error", 0.0, 0.0, f"API Error: {e}"

@st.cache_resource
def train_gensim_lda(processed_docs_gensim, n_topics):
    if not processed_docs_gensim or len(processed_docs_gensim) < n_topics:
        logger.warning("Not enough valid documents for LDA topic modeling.")
        return None, None, "Not enough valid documents for LDA."
    try:
        id2word = corpora.Dictionary(processed_docs_gensim)
        id2word.filter_extremes(no_below=1, no_above=1.0)
        corpus = [id2word.doc2bow(text) for text in processed_docs_gensim]
        if not corpus or all(not doc for doc in corpus):
             logger.warning("Corpus became empty after dictionary filtering for LDA.")
             return None, None, "Corpus became empty after dictionary filtering. Adjust parameters or check input text."
        lda_model = LdaMulticore(
            corpus=corpus,
            id2word=id2word,
            num_topics=n_topics,
            random_state=100,
            chunksize=100,
            passes=10,
            alpha='asymmetric',
            eta='auto',
            per_word_topics=True,
            workers=os.cpu_count()-1 if os.cpu_count() > 1 else 1
        )
        logger.info(f"LDA model trained with {n_topics} topics.")
        return lda_model, corpus, None
    except Exception as e:
        logger.error(f"Gensim LDA training failed: {e}")
        return None, None, f"Gensim LDA training failed: {e}"

st.title("🚀 Advanced NLP Demo: Sentiment, Topics & Clustering")
st.markdown("""
This demo analyzes documents using:
1.  **Sentiment Analysis:** OpenAI API
2.  **Topic Modeling:** Gensim LDA
3.  **Document Clustering:** Scikit-learn K-Means

Includes interactive visualizations using Plotly.
""")
st.markdown("---")

st.header("1. Input Documents")
st.markdown("Paste documents below, separated by `---` or blank lines.")
default_text = """
The system upgrade drastically improved processing speed and reduced latency. Highly satisfied.
---
Users are complaining about the new login process. It's confusing and often fails on mobile devices. We need a fix ASAP.
---
Our latest marketing initiative generated significant buzz online, driving record engagement numbers. Great work team!
---
Customer support wait times are unacceptable. Many negative reviews mention long holds and unhelpful agents. Urgent attention required.
---
Analysis of competitor pricing strategies suggests we may need to adjust our premium tier offering to remain competitive.
---
The company retreat fostered strong team bonding and generated innovative ideas for future projects. A worthwhile investment.
---
Supply chain disruptions are impacting delivery schedules for critical components. This poses a risk to Q4 targets.
---
Our research team's publication on sustainable AI practices has been widely cited, boosting our industry credibility.
---
The new dashboard interface is intuitive and has reduced training time for new hires.
---
Several users reported bugs in the latest software release, particularly with the export feature.
---
The marketing team exceeded their quarterly goals, increasing brand awareness across multiple channels.
---
A recent security audit identified vulnerabilities that need to be addressed immediately.
---
The product roadmap was well received by stakeholders, with particular excitement around the AI integration.
---
Customer feedback indicates a strong preference for live chat support over email.
---
The logistics team successfully navigated recent shipping delays, ensuring all orders arrived on time.
---
Employee satisfaction scores have improved following the implementation of flexible work hours.
---
The latest update to the mobile app has been downloaded over 10,000 times in the first week.
---
There is confusion among users regarding the new subscription tiers. More communication is needed.
---
The IT department resolved the network outage quickly, minimizing downtime for all staff.
---
Our new sustainability initiative has been featured in several industry publications.
---
The helpdesk has seen a 20% reduction in ticket volume since launching the self-service portal.
---
A competitor has launched a similar product at a lower price point, which may impact our sales.
---
The annual company picnic was a great success, with high participation and positive feedback.
---
The finance team identified cost-saving opportunities in our vendor contracts.
---
The onboarding process for new clients is too lengthy and needs to be streamlined.
---
Our social media campaign went viral, resulting in a significant uptick in website traffic.
---
The new CRM system has improved lead tracking and conversion rates.
---
There are ongoing issues with the payment gateway, causing frustration for customers.
---
The training program for remote employees received high marks for clarity and engagement.
---
The recent product recall was handled efficiently, with minimal negative press.
---
Our partnership with local charities has enhanced our corporate social responsibility profile.
---
The new office layout has increased collaboration among teams.
---
The customer loyalty program is driving repeat business and higher average order values.
---
The website redesign has improved navigation and reduced bounce rates.
---
The latest round of layoffs has affected morale in several departments.
---
The R&D team is making progress on the next-generation product prototype.
---
The recent webinar attracted over 500 participants from around the globe.
---
The new compliance regulations require updates to our data handling procedures.
---
The product packaging redesign has received positive feedback from customers.
---
The sales team closed several large deals this quarter, exceeding targets.
---
The customer success team has implemented a new feedback loop to improve service quality.
---
The new knowledge base articles have reduced the number of basic support queries.
"""
input_text = st.text_area("Paste text here:", default_text, height=300)

st.sidebar.header("⚙️ Analysis Parameters")
min_docs_required = 3

n_topics = st.sidebar.slider(
    "Number of Topics (Gensim LDA):", min_value=2, max_value=10, value=3, step=1,
    help=f"Requires at least {min_docs_required} valid documents after preprocessing."
)
n_clusters = st.sidebar.slider(
    "Number of Clusters (K-Means):", min_value=2, max_value=10, value=3, step=1,
    help=f"Requires at least {min_docs_required} valid documents after preprocessing."
)
min_df = st.sidebar.slider(
    "Min Doc Freq (Sklearn TF-IDF):", min_value=1, max_value=5, value=2, step=1,
    help="Min docs a word must be in for K-Means vectorization."
)
max_df = st.sidebar.slider(
    "Max Doc Freq % (Sklearn TF-IDF):", min_value=0.70, max_value=1.0, value=0.95, step=0.05,
    help="Max % of docs a word can be in for K-Means vectorization."
)

analyze_button = st.button("📊 Analyze Documents")
st.sidebar.markdown("---")
st.sidebar.info(f"Topic Modeling & Clustering require at least {min_docs_required} docs remaining after preprocessing.")

if analyze_button and input_text:
    if not api_key_input or not client:
         logger.error("OpenAI API Key is missing or invalid. Cannot perform Sentiment Analysis.")
         st.error("OpenAI API Key is missing or invalid. Cannot perform Sentiment Analysis.", icon="🚨")
         st.stop()
    st.markdown("---")
    st.header("📈 Analysis Results")
    docs_raw = re.split(r'\n---\n|\n\n+', input_text.strip())
    docs_raw = [doc.strip() for doc in docs_raw if doc.strip()]
    if not docs_raw:
        logger.warning("No documents found in input.")
        st.warning("No documents found. Please paste valid text.")
        st.stop()
    st.write(f"Detected {len(docs_raw)} documents.")
    logger.info(f"Parsed {len(docs_raw)} documents from input.")
    df = pd.DataFrame({'Original Text': docs_raw})
    results_placeholder = st.empty()
    st.subheader("Sentiment Analysis (OpenAI)")
    sentiments_data = []
    with st.spinner("Calling OpenAI API for sentiment analysis... (This may take a moment)"):
        for i, text in enumerate(df['Original Text']):
            sentiment, polarity, subjectivity, status = get_sentiment_openai(client, text)
            sentiments_data.append({
                'Sentiment': sentiment,
                'Polarity': polarity,
                'Subjectivity': subjectivity,
                'API Status': status
            })
            logger.info(f"Sentiment for doc {i+1}: {sentiment}, polarity: {polarity}, subjectivity: {subjectivity}")
            time.sleep(0.1)
    sentiment_df = pd.DataFrame(sentiments_data)
    df = pd.concat([df, sentiment_df], axis=1)
    st.dataframe(df[['Original Text', 'Sentiment', 'Polarity', 'Subjectivity', 'API Status']].style.format({
        'Polarity': "{:.3f}",
        'Subjectivity': "{:.3f}"
    }))
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig_sentiment = px.pie(sentiment_counts, names='Sentiment', values='Count',
                           title='Sentiment Distribution',
                           color='Sentiment',
                           color_discrete_map={'Positive':'green', 'Negative':'red', 'Neutral':'grey', 'Error':'orange'})
    fig_sentiment.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_sentiment, use_container_width=True)
    st.subheader("Preprocessing for Topic Modeling & Clustering")
    with st.spinner("Preprocessing texts..."):
        processed_docs_gensim = [preprocess_for_gensim(doc) for doc in df['Original Text']]
        valid_indices_gensim = [i for i, doc in enumerate(processed_docs_gensim) if doc]
        processed_docs_gensim_valid = [doc for i, doc in enumerate(processed_docs_gensim) if i in valid_indices_gensim]
        processed_docs_sklearn = [preprocess_for_sklearn(doc) for doc in df['Original Text']]
        valid_indices_sklearn = [i for i, doc in enumerate(processed_docs_sklearn) if doc]
        processed_docs_sklearn_valid = [doc for i, doc in enumerate(processed_docs_sklearn) if i in valid_indices_sklearn]
    logger.info(f"Preprocessing complete. {len(processed_docs_gensim_valid)} docs valid for Gensim, {len(processed_docs_sklearn_valid)} docs valid for Sklearn Clustering.")
    st.write(f"Preprocessing complete. {len(processed_docs_gensim_valid)} docs valid for Gensim, {len(processed_docs_sklearn_valid)} docs valid for Sklearn Clustering.")
    st.subheader(f"Topic Modeling (Gensim LDA - {n_topics} Topics)")
    if len(processed_docs_gensim_valid) < min_docs_required or n_topics > len(processed_docs_gensim_valid):
        logger.warning(f"Insufficient documents ({len(processed_docs_gensim_valid)}) for {n_topics} topics after preprocessing.")
        st.warning(f"Insufficient documents ({len(processed_docs_gensim_valid)}) for {n_topics} topics after preprocessing.")
        topic_modeling_successful = False
    else:
        with st.spinner(f"Training Gensim LDA model with {n_topics} topics..."):
             lda_model, corpus, error_msg = train_gensim_lda(processed_docs_gensim_valid, n_topics)
        if lda_model:
            st.write("**Identified Topics & Top Words:**")
            topic_data = []
            topics = lda_model.show_topics(num_topics=n_topics, num_words=10, formatted=False)
            for i, topic in topics:
                words = [word for word, prop in topic]
                topic_data.append({'Topic': f"Topic {i + 1}", 'Top Words': ", ".join(words)})
            topics_df = pd.DataFrame(topic_data)
            st.table(topics_df)
            topic_term_data = []
            for i, topic in topics:
                 for word, prob in topic:
                     topic_term_data.append({'Topic': f"Topic {i + 1}", 'Word': word, 'Probability': prob})
            topic_term_df = pd.DataFrame(topic_term_data)
            fig_topics = px.bar(topic_term_df.sort_values(['Topic', 'Probability'], ascending=[True, False]),
                                x='Probability', y='Word', color='Topic',
                                orientation='h',
                                facet_col='Topic', facet_col_wrap=min(5, n_topics),
                                title='Top Words per Topic (Gensim LDA)',
                                height=200 * ((n_topics // 5) + 1) + 100,
                                labels={'Probability': 'Word Probability within Topic'},
                                category_orders={"Topic": [f"Topic {i+1}" for i in range(n_topics)]})
            fig_topics.update_yaxes(matches=None, showticklabels=True)
            fig_topics.update_layout(yaxis_title=None)
            st.plotly_chart(fig_topics, use_container_width=True)
            logger.info(f"LDA topic modeling successful with {n_topics} topics.")
            topic_modeling_successful = True
        else:
            logger.error(f"Topic Modeling Failed: {error_msg}")
            st.error(f"Topic Modeling Failed: {error_msg}")
            topic_modeling_successful = False
    st.subheader(f"Document Clustering (K-Means - {n_clusters} Clusters)")
    if len(processed_docs_sklearn_valid) < min_docs_required or n_clusters > len(processed_docs_sklearn_valid) :
         logger.warning(f"Insufficient documents ({len(processed_docs_sklearn_valid)}) for {n_clusters} clusters after preprocessing.")
         st.warning(f"Insufficient documents ({len(processed_docs_sklearn_valid)}) for {n_clusters} clusters after preprocessing.")
    else:
         with st.spinner(f"Vectorizing and running K-Means with {n_clusters} clusters..."):
             try:
                 vectorizer_cluster = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words='english', ngram_range=(1, 2))
                 tfidf_matrix_cluster = vectorizer_cluster.fit_transform(processed_docs_sklearn_valid)
                 if tfidf_matrix_cluster.shape[0] > 0 and tfidf_matrix_cluster.shape[1] > 0:
                     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                     kmeans.fit(tfidf_matrix_cluster)
                     cluster_labels = kmeans.labels_
                     df['Cluster'] = pd.Series(dtype='string')
                     df['Cluster Label'] = pd.Series(dtype='int')
                     df.loc[valid_indices_sklearn, 'Cluster Label'] = cluster_labels
                     df['Cluster'] = df['Cluster Label'].apply(lambda x: f"Cluster {int(x + 1)}" if pd.notna(x) else "N/A")
                     st.write("**Document Cluster Assignments:**")
                     st.dataframe(df[['Original Text', 'Cluster']])
                     st.write("**Cluster Visualization (PCA Projection):**")
                     pca = PCA(n_components=2, random_state=42)
                     tfidf_matrix_dense = np.asarray(tfidf_matrix_cluster.todense())
                     n_samples = tfidf_matrix_dense.shape[0]
                     if n_samples >= 2:
                        pca_components = pca.fit_transform(tfidf_matrix_dense)
                        pca_df = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
                        pca_df['Cluster'] = df.loc[valid_indices_sklearn, 'Cluster'].values
                        pca_df['Text Snippet'] = df.loc[valid_indices_sklearn, 'Original Text'].apply(lambda x: x[:100] + '...').values
                        fig_cluster = px.scatter(pca_df, x='PCA1', y='PCA2', color='Cluster',
                                                title=f'Document Clusters ({n_clusters}) projected onto 2 PCA Components',
                                                hover_name='Cluster',
                                                hover_data={'Cluster': False, 'Text Snippet': True, 'PCA1':':.2f', 'PCA2':':.2f'},
                                                color_discrete_sequence=px.colors.qualitative.Plotly)
                        fig_cluster.update_traces(marker=dict(size=10, opacity=0.8))
                        fig_cluster.update_layout(legend_title_text='Cluster')
                        st.plotly_chart(fig_cluster, use_container_width=True)
                        logger.info(f"K-Means clustering and PCA visualization successful with {n_clusters} clusters.")
                     else:
                        logger.warning("Cannot perform PCA visualization with fewer than 2 valid documents for clustering.")
                        st.warning("Cannot perform PCA visualization with fewer than 2 valid documents for clustering.")
                     cluster_counts = df['Cluster'].value_counts().drop('N/A', errors='ignore').sort_index().reset_index()
                     cluster_counts.columns = ['Cluster', 'Count']
                     fig_dist = px.bar(cluster_counts, x='Cluster', y='Count', color='Cluster',
                                     title='Cluster Document Counts',
                                     color_discrete_sequence=px.colors.qualitative.Plotly)
                     st.plotly_chart(fig_dist, use_container_width=True)
                 else:
                     logger.warning("TF-IDF matrix for clustering is empty after vectorization. Cannot perform K-Means.")
                     st.warning("TF-IDF matrix for clustering is empty after vectorization. Cannot perform K-Means.")
             except Exception as e:
                 logger.error(f"An error occurred during Clustering: {e}")
                 st.error(f"An error occurred during Clustering: {e}")
elif analyze_button:
    logger.warning("Analyze button pressed but no input text provided.")
    st.warning("Please paste some text into the input area above.")
st.markdown("---")
st.caption(f"Advanced NLP Demo | Current Date: {pd.Timestamp('now', tz='Europe/Berlin').strftime('%Y-%m-%d %H:%M:%S %Z')}")