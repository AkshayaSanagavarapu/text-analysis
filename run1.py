import streamlit as st
from gtts import gTTS
import os
import speech_recognition as sr
from pydub import AudioSegment
import io
import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
import spacy
from spacy.cli import download as spacy_download
from spacy import displacy

# --------------------- NLTK setup for TextBlob ---------------------
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# --------------------- spaCy setup ---------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --------------------- Streamlit App ---------------------
st.title("🧠 Unstructured Data Analysis")

tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "🎧 Audio Analysis", "📝 Text Analysis"])

with tab3:
    # Hardcoded sample stories
    stories = [
        "Your first story here...",
        "Your second story here...",
        "Your third story here...",
        "Your fourth story here...",
        "Your fifth story here..."
    ]

    # Initialize session_state for text area
    if "text_area" not in st.session_state:
        st.session_state.text_area = ""

    # Random story button
    if st.button("🎲 Random Story"):
        st.session_state.text_area = random.choice(stories)

    # Text area
    st.session_state.text_area = st.text_area(
        "Paste or modify your text here:",
        value=st.session_state.text_area,
        height=250
    )

    # Analyze button
    if st.button("Analyze Text 🚀"):
        text = st.session_state.text_area.strip()

        if text:
            blob = TextBlob(text)
            words_and_tags = blob.tags  # (word, POS tag)

            # POS extraction
            nouns = [word for word, tag in words_and_tags if tag.startswith('NN')]
            verbs = [word for word, tag in words_and_tags if tag.startswith('VB')]
            adjectives = [word for word, tag in words_and_tags if tag.startswith('JJ')]
            adverbs = [word for word, tag in words_and_tags if tag.startswith('RB')]

            # WordCloud generator
            def make_wordcloud(words, color):
                if not words:
                    st.warning("No words found for this category.")
                    return None
                text_for_wc = " ".join(words)
                wc = WordCloud(width=500, height=400, background_color='black', colormap=color).generate(text_for_wc)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                return fig

            # Layout 2x2
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            with col1:
                st.markdown("### 🧠 Nouns")
                fig = make_wordcloud(nouns, "plasma")
                if fig: st.pyplot(fig)
            with col2:
                st.markdown("### ⚡ Verbs")
                fig = make_wordcloud(verbs, "inferno")
                if fig: st.pyplot(fig)
            with col3:
                st.markdown("### 🎨 Adjectives")
                fig = make_wordcloud(adjectives, "cool")
                if fig: st.pyplot(fig)
            with col4:
                st.markdown("### 💨 Adverbs")
                fig = make_wordcloud(adverbs, "magma")
                if fig: st.pyplot(fig)

            # Quick stats
            st.markdown("### 📊 POS Counts")
            st.write({
                "Nouns": len(nouns),
                "Verbs": len(verbs),
                "Adjectives": len(adjectives),
                "Adverbs": len(adverbs)
            })
        else:
            st.warning("Please paste or select some text first.")

    # --------------------- Named Entity Recognition ---------------------
    text = st.session_state.get('text_area', '').strip()
    if text:
        doc = nlp(text)
        html = displacy.render(doc, style="ent", jupyter=False)
        st.write("**Detected Entities:**", unsafe_allow_html=True)
        st.markdown(html, unsafe_allow_html=True)

        entities = [(ent.text, ent.label_) for ent in doc.ents]
        if entities:
            st.markdown("**Entity Table:**")
            st.table(entities)
        else:
            st.info("No named entities found.")
    else:
        st.info("Paste or select some text to see NER results.")
