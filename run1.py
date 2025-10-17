import streamlit as st
from gtts import gTTS
import os
import speech_recognition as sr
from pydub import AudioSegment
import io

from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import random


st.title("🧠 Unstructured Data Analysis")


tab1, tab2, tab3 = st.tabs(
    ["🖼️ Image Analysis", "🎧 Audio Analysis", "📝 Text Analysis"])
with tab3:
    # Hardcoded sample stories
    stories = [
    """In a remote kingdom nestled between jagged mountains and endless forests, Princess Elara spent her days exploring the sprawling royal gardens, which stretched beyond what the eye could see. Ancient trees, taller than any castle spire, whispered secrets of centuries past. Streams glimmered like silver ribbons, and hidden nooks revealed forgotten statues of long-lost rulers. Each evening, the princess wandered alone, following the soft glow of luminescent flowers that seemed to respond to her presence. She discovered a small, crystal-clear pond where the water shimmered with reflections of constellations that weren’t visible in the night sky above. Mystical creatures, some with wings of iridescent silk and others with scales that reflected the sunlight, emerged from the undergrowth. They spoke in melodious tones, sharing wisdom about magic, courage, and the legacy of her ancestors. Elara meticulously recorded everything in her leather-bound journal, eager to preserve these secrets. Days passed, and she began experimenting with the spells and enchantments whispered by the flora and fauna, learning to manipulate elements subtly. The castle staff often wondered at her prolonged absences, yet the king, wise and patient, trusted that his daughter’s heart and mind were being shaped by forces beyond ordinary understanding. Every choice Elara made was infused with the lessons of the garden: patience, observation, and empathy. When the kingdom faced political unrest, she used her newfound knowledge to mediate conflicts, negotiating peace with neighboring realms through insight and subtle magic rather than brute force. Her reputation as a wise and capable ruler spread far and wide, and travelers from distant lands ventured to witness the kingdom where nature and magic intertwined seamlessly.""",

    """During the bustling era of the 1920s, in a city that never slept, Detective Samuel Hart navigated the labyrinthine streets of New York. The roar of jazz music spilled from dimly lit speakeasies, blending with the clatter of streetcars and the occasional wail of a distant siren. Samuel’s office, cluttered with case files and newspapers, smelled of tobacco and ink. Each case presented new challenges: jewel thefts orchestrated with surgical precision, clandestine meetings under the flickering glow of lampposts, and coded messages that tested his intellect. Night after night, he roamed the city in a long trench coat and fedora, cigarette smoke curling around his face as he pieced together clues overlooked by ordinary citizens. The criminal underworld was intricate, with alliances shifting like sand dunes in the desert, and every wrong move could prove fatal. Samuel’s reputation for solving impossible cases made him both feared and respected. Alongside the chase for criminals, he struggled with his own personal demons: memories of a lost love, the guilt of past failures, and the weight of responsibility. Yet, through the labyrinth of shadows and danger, Samuel remained vigilant. He discovered secret societies, underground gambling rings, and smuggling operations that stretched from the docks to the penthouses of the elite. By piecing together these disparate threads, he not only prevented crimes but also uncovered a web of corruption that threatened the entire city’s stability. His intuition, honed over years of observation and deduction, became his greatest weapon, and every solved case reinforced his unyielding belief in justice and perseverance.""",

    """On a distant exoplanet, where the sky shimmered in surreal hues of emerald and violet, Captain Rhea led a team of explorers through canyons carved by ancient rivers of liquid crystal. Each step on the iridescent terrain revealed new flora and fauna, alien yet strangely familiar, pulsating in rhythm with the wind. Towering spires of stone rose like jagged fingers, and the explorers’ suits glimmered with embedded sensors that captured every subtle vibration. Bioluminescent plants reacted to human presence, illuminating pathways that seemed almost deliberately designed. The team documented behaviors of translucent-winged creatures, some of which emitted harmonic tones that resonated with the planet’s magnetic field. Captain Rhea, with her scientific curiosity, took meticulous notes and samples, aware that every observation could revolutionize understanding of life and physics. Nights brought challenges: temperatures dropped sharply, and mysterious energy storms painted the sky with auroras of indescribable colors. The explorers huddled in portable shelters, analyzing data, and theorizing about ecological interdependencies that were far beyond terrestrial analogues. As they progressed, they uncovered ruins of an ancient civilization, with inscriptions that hinted at advanced knowledge of energy manipulation and interstellar communication. Each discovery raised more questions than answers, but the team pressed on, driven by curiosity and the thrill of discovery. The mission would redefine humanity’s understanding of life beyond Earth, blending awe, danger, and revelation at every turn.""",

    """In the neon-lit heart of Tokyo, young coder Akira toiled over lines of code that promised to revolutionize urban transportation. Streets pulsed with energy as neon advertisements flickered over crowded crosswalks, and the hum of trains beneath the city created a constant rhythm. Akira’s AI was designed to predict traffic congestion and prevent accidents, analyzing millions of data points in real-time. His workspace was cluttered with multiple monitors, coffee cups, and mechanical keyboards, each keystroke echoing determination and fatigue. Nights blurred into mornings as he tested algorithms, refined neural networks, and debugged edge cases. The AI began to anticipate patterns in human behavior, predicting jaywalkers, sudden lane changes, and even subtle signals of driver distraction. Akira faced ethical dilemmas: how much autonomy should the AI have, and what risks were acceptable in the pursuit of safety? Despite the challenges, he persisted, collaborating with urban planners, engineers, and ethicists to ensure the system balanced efficiency with human life. When the AI successfully prevented its first potential accident, Akira felt a profound sense of accomplishment, knowing his work could save countless lives. Yet, the city continued to evolve, presenting new challenges daily, and he remained vigilant, refining the system and pushing the boundaries of artificial intelligence, ethics, and human-machine collaboration.""",

    """Deep in the Amazon rainforest, a team of scientists embarked on an unprecedented expedition to discover rare medicinal plants. Guided by indigenous elders with knowledge passed down through generations, they navigated treacherous rivers teeming with exotic wildlife, thick canopies that blocked sunlight, and terrain that shifted unpredictably with every step. Nights were spent around flickering campfires, where the symphony of nocturnal creatures filled the air with haunting melodies. Each day brought new discoveries: plants with leaves that changed color in response to touch, fungi that glowed faintly in the dark, and insects exhibiting intricate social behaviors previously undocumented. Researchers meticulously cataloged every observation, photographing, sampling, and recording the interactions between species. The expedition wasn’t without danger; venomous snakes, sudden storms, and disorienting labyrinths of vines tested their resilience. Yet, their perseverance paid off as they uncovered a plant with compounds potentially capable of treating rare diseases, promising breakthroughs in medicine. Through collaboration, observation, and respect for the delicate ecosystem, the team combined ancient indigenous wisdom with modern scientific techniques, leaving a legacy of knowledge that would inform future research and conservation efforts."""
    ]

    # Initialize session_state for text area
if "text_area" not in st.session_state:
        st.session_state.text_area = ""

    # Random story button
if st.button("🎲 Random Story"):
        st.session_state.text_area = random.choice(stories)

    # Text area (shows random story if chosen, else empty / user input)
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
                if not words or len(words) == 0:
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







import streamlit as st
import spacy
from spacy import displacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Use the current text from your text area
text = st.session_state.get('text_area', '')

if text.strip():
    doc = nlp(text)

    # Display entities with colors using displaCy HTML
    html = displacy.render(doc, style="ent", jupyter=False)
    st.write("**Detected Entities:**", unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)

    # Optional: show entity table
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    if entities:
        st.markdown("**Entity Table:**")
        st.table(entities)
    else:
        st.info("No named entities found.")

else:
    st.info("Paste or select some text to see NER results.")








