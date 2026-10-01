import streamlit as st
import pandas as pd
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Konfigurasi Layout & Mengurangi Ruang Kosong (Padding atas)
st.set_page_config(
    page_title="Twitter Sentiment Analysis Portfolio | Elieser Pasaribu",
    page_icon="🎭💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk implementasi gradasi gambar, efek glassmorphism, dan estetika
st.markdown("""
    <style>
        /* Mengubah Latar Belakang Utama Aplikasi dengan Gradasi dan Garis Diagonal */
        .stApp {
            background: 
                repeating-linear-gradient(
                    135deg,
                    rgba(255, 255, 255, 0.05) 0px,
                    rgba(255, 255, 255, 0.05) 1px,
                    transparent 1px,
                    transparent 60px
                ),
                linear-gradient(135deg, #e90065 0%, #6f0088 45%, #18004b 100%);
            background-attachment: fixed;
            color: #ffffff !important;
        }

        /* Menyesuaikan warna teks secara umum agar terlihat jelas */
        h1, h2, h3, p, label {
            color: #ffffff !important;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        /* Info Card dengan efek Glassmorphism (Kaca) */
        .info-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #ff4081; /* Disesuaikan dengan warna tema */
            border-top: 1px solid rgba(255, 255, 255, 0.2);
            border-right: 1px solid rgba(255, 255, 255, 0.2);
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            margin-bottom: 20px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }

        /* Styling Form Input dan Text Area */
        .stTextArea textarea, .stTextInput input {
            background-color: rgba(10, 15, 30, 0.4) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 8px !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: rgba(15, 10, 45, 0.7) !important;
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Pastikan resource NLTK terunduh aman
@st.cache_resource
def download_nltk():
    resources = [
        'stopwords', 
        'wordnet', 
        'punkt', 
        'punkt_tab', 
        'averaged_perceptron_tagger', 
        'averaged_perceptron_tagger_eng'
    ]
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass

download_nltk()

# --- DEFINISI FUNGSI TEXT PREPROCESSING ---
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def text_preprocessing(text):
    text_str = str(text)
    text_tokenize = word_tokenize(text_str)
    if len(text_tokenize) == 0:
        return []
    entity = text_tokenize[0]
    text_content = text_tokenize[1:]
    text_pos = pos_tag(text_content)
    remove_words = set(list(string.punctuation) + stopwords.words('english'))
    text_remove = [(word, pos) for (word, pos) in text_pos if word not in remove_words]
    word_lem = WordNetLemmatizer()
    text_lem = []
    for word, pos in text_remove:
        wn_pos = get_wordnet_pos(pos)
        if wn_pos is not None:
            text_lem.append((word_lem.lemmatize(word, pos=wn_pos), pos))
        else:
            text_lem.append((word_lem.lemmatize(word), pos))
    text_lem.append((entity,))
    return text_lem

# Load Model & Pipeline
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load('nlp_pipeline.pkl')
    model = joblib.load('sentiment_model.pkl')
    return pipeline, model

pipeline, model = load_artifacts()

# --- SIDEBAR MENU & DEVELOPED BY ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/twitter-circled--v1.png", width=60)
    st.title("Portfolio Navigation")
    menu = st.selectbox("Select Analysis Mode", ["Single Text Prediction", "Batch File Upload (CSV)"])
    
    st.markdown("---")
    st.markdown("### Developed by")
    st.markdown("""
    <div style='font-size: 1.05rem; color: #ffffff; margin-bottom: 4px; font-weight: bold;'>Elieser Pasaribu</div>
    <div style='font-size: 0.88rem; color: #ff9ed6; line-height: 1.4;'>Data Analyst | Data Scientist | Machine Learning</div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 Architecture Note")
    st.markdown("✨ *End-to-End NLP System utilizing POS Tagging, Lemmatization, and TF-IDF Vectorization.*")

# --- MAIN CONTENT ---
st.title("🎭💬 Twitter Sentiment Analysis & Entity Intelligence")

st.markdown("""
<div class="info-card">
    <p style="margin: 0px; color: #f1f5f9; font-size: 0.98rem; line-height: 1.6;">
        🎯 <strong>Project Objective:</strong> This interactive portfolio web application demonstrates an advanced 
        Natural Language Processing (NLP) pipeline. It analyzes public tweets associated with specific entities 
        to classify public sentiments in <em>real-time</em> or via <em>batch processing</em>, delivering actionable insights from social media data.
    </p>
</div>
""", unsafe_allow_html=True)

if menu == "Single Text Prediction":
    st.subheader("🔍 Real-Time Sentiment Prediction")
    st.markdown("Enter a target entity (e.g., *Samsung*, *Apple*, *Overwatch*) and a tweet/review in English to analyze its sentiment.")
    
    entity_input = st.text_input("Entity / Topic Name", "Samsung")
    tweet_input = st.text_area("Tweet / Comment Content (English):", "Samsung phone battery life is terrible and disappointing.")
    
    if st.button("🚀 Run Sentiment Analysis", type="primary"):
        if entity_input and tweet_input:
            combined_text = entity_input + " " + tweet_input
            
            # Prediksi model
            processed_text = pipeline.transform([combined_text])
            prediction = model.predict(processed_text)[0]
            
            st.markdown("---")
            st.subheader("🧠✨ Analysis Results:")
            
            # Output dengan alert box khas Streamlit dan narasi 2-3 kalimat
            if prediction == "Positive":
                st.success(
                    f"Predicted Sentiment: **{prediction.upper()}** 🎉\n\n"
                    "The model evaluates this text as expressing satisfaction, high praise, or positive engagement toward the specified entity. "
                    "This indicates strong customer approval and favorable public perception within the analyzed context."
                )
            elif prediction == "Negative":
                st.error(
                    f"Predicted Sentiment: **{prediction.upper()}** ⚠️\n\n"
                    "The model detects critical feedback, complaints, or expressions of dissatisfaction regarding the target entity. "
                    "This highlights potential product issues, poor user experiences, or areas that require immediate operational attention."
                )
            else:
                st.warning(
                    f"Predicted Sentiment: **{prediction.upper()}** 📌\n\n"
                    "The text reflects general remarks, objective statements, or neutral commentary without strong emotional polarization. "
                    "It represents baseline public interest or informational sharing rather than direct praise or criticism."
                )

        else:
            st.warning("Please fill in both the entity and tweet content fields!")

elif menu == "Batch File Upload (CSV)":
    st.subheader("📂 Large-Scale Batch Processing")
    st.markdown("Upload a CSV file containing bulk tweet data to run automated sentiment analysis.")
    
    uploaded_file = st.file_uploader("Upload CSV file (must include 'entity' and 'tweet content' columns)", type=["csv"])
    
    if uploaded_file is not None:
        df_user = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df_user.head(3))
        
        if st.button("⚡ Execute Batch Prediction", type="primary"):
            with st.spinner("Processing all tweets through the NLP pipeline..."):
                if 'entity' in df_user.columns and 'tweet content' in df_user.columns:
                    df_user['combined'] = df_user['entity'].astype(str) + " " + df_user['tweet content'].astype(str)
                    processed_batch = pipeline.transform(df_user['combined'])
                    df_user['Predicted_Sentiment'] = model.predict(processed_batch)
                    
                    st.success("Batch Analysis Completed Successfully!")
                    st.dataframe(df_user.head(15), use_container_width=True)
                    
                    csv_download = df_user.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Result as CSV", csv_download, "sentiment_analysis_results.csv", "text/csv")
                else:
                    st.error("Your CSV file must contain 'entity' and 'tweet content' columns!")