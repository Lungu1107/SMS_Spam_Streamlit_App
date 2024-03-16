import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import os

# Download necessary NLTK resources if not already installed (comment out if not needed)
nltk_data_to_download = ['averaged_perceptron_tagger', 'wordnet', 'stopwords']
for data in nltk_data_to_download:
    nltk.download(data)

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load stopwords and punctuation outside the function to optimize performance
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}
    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)


def transform_text(text, return_tokens=False):
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = []
    for token in tokens:
        if token.isalnum() and token not in stop_words and token not in punctuation:
            lemma_token = lemmatizer.lemmatize(token, get_wordnet_pos(token))
            filtered_tokens.append(lemma_token)

    if return_tokens:
        return filtered_tokens
    return " ".join(filtered_tokens)


def highlight_significant_words(tfidf_vectorizer, tokens_to_highlight):
    feature_names = tfidf_vectorizer.get_feature_names_out()
    scores = tfidf_vectorizer.transform([" ".join(tokens_to_highlight)]).toarray().flatten()
    token_scores = {token: scores[feature_names.tolist().index(token)] for token in tokens_to_highlight if
                    token in feature_names}
    significant_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)
    return significant_tokens[:5]  # Adjust the number of tokens based on your preference


def load_pickle(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    else:
        st.error(f"File {file_path} not found. Please ensure the file is correctly placed.")
        return None


# Load the Vectorizer and the model from disk
tfidf = load_pickle('vectorizer.pkl')
model = load_pickle('model.pkl')

if tfidf is None or model is None:
    st.stop()

st.title("SMS Spam Classifier")

st.markdown("""
    **Instructions:**
    - Enter the SMS message you want to classify in the text area below.
    - Press the **Predict** button to classify the message.
    - Messages will be classified as either **Spam** or **Not Spam**.
    - The words that might have influenced the classification will be highlighted below.
""")

with st.sidebar:
    st.header("Enter SMS Message")
    input_sms = st.text_area("Message", height=150, help="Type the SMS message here.")

if st.sidebar.button('Predict'):
    try:
        # Preprocess and get tokens
        processed_tokens = transform_text(input_sms, return_tokens=True)
        # Vectorize
        vector_input = tfidf.transform([" ".join(processed_tokens)]).toarray()
        # Predict
        result = model.predict(vector_input)[0]
        # Display
        if result == 1:
            st.markdown(f'<p style="color: red; font-size: 48px;">Spam</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="color: green; font-size: 48px;">Not Spam</p>', unsafe_allow_html=True)

        # Highlight significant words
        significant_words = highlight_significant_words(tfidf, processed_tokens)
        # Display significant words
        st.markdown("**Words that might have influenced this classification:**")
        st.write(", ".join([f"{word} (score: {score:.2f})" for word, score in significant_words]))
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("""
    **Example Spam SMS:**
    - Congratulations! You've won a free cruise to the Bahamas! To claim your prize now, call us at 123-456-7890.
""")
