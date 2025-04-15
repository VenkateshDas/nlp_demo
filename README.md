# Advanced NLP Demo: Sentiment, Topics & Clustering

This Streamlit application demonstrates several Natural Language Processing (NLP) techniques applied to user-provided text documents.

## Features

1.  **Sentiment Analysis:** Utilizes the OpenAI API (GPT-3.5 Turbo) to determine the sentiment (Positive, Negative, Neutral) of each document, along with polarity and subjectivity scores.
2.  **Topic Modeling:** Employs Gensim's LDA (Latent Dirichlet Allocation) algorithm to identify underlying topics within the document set. Displays the top words for each identified topic.
3.  **Document Clustering:** Uses Scikit-learn's K-Means algorithm with TF-IDF vectorization to group similar documents into clusters.
4.  **Interactive Visualizations:** Presents results using Plotly charts, including:
    *   Sentiment distribution (pie chart)
    *   Top words per topic (bar chart)
    *   Cluster assignments (table)
    *   PCA projection of clusters (scatter plot)
    *   Cluster distribution (bar chart)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    *You'll need to create a `requirements.txt` file based on the imports in `app.py`.* A potential list is:
    ```
    streamlit
    pandas
    spacy
    scikit-learn
    plotly
    numpy
    openai
    gensim
    nltk
    python-dotenv
    ```
    Install them using:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download spaCy model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Download NLTK stopwords:**
    Run python and execute:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

6.  **Configure OpenAI API Key:**
    *   Create a file named `.env` in the project root.
    *   Add your OpenAI API key to this file:
        ```
        OPENAI_API_KEY='your_api_key_here'
        ```
    *   Alternatively, you can enter the API key directly in the Streamlit sidebar when running the app, but using `.env` is recommended.

## Usage

1.  Ensure your virtual environment is activated.
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  The application will open in your web browser.
4.  Paste your documents into the text area (separated by `---` or blank lines).
5.  Adjust the analysis parameters (Number of Topics, Number of Clusters, TF-IDF settings) in the sidebar if needed.
6.  Enter your OpenAI API key in the sidebar if you haven't set it in the `.env` file.
7.  Click the "Analyze Documents" button to view the results.

## Dependencies

*   Streamlit
*   Pandas
*   spaCy (`en_core_web_sm` model)
*   Scikit-learn
*   Plotly
*   NumPy
*   OpenAI
*   Gensim
*   NLTK (stopwords)
*   python-dotenv 