import streamlit as st
import os
import tempfile
import re
import random
from collections import Counter
import google.generativeai as genai
from audio_processor import AudioProcessor
import spacy
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("spaCy model not found. Please install en_core_web_sm.")
    st.stop()

# Helper Functions
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_summary(notes, top_n=3):
    if not notes:
        return "Summary could not be generated due to insufficient conceptual content."

    # Strip the bullet points and whitespace
    sentences = [n.replace("• ", "").strip() for n in notes]

    # If the notes are already short enough, just join them into a paragraph
    if len(sentences) <= top_n:
        return " ".join(sentences)

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(sentences)
        similarity = (X * X.T).toarray()

        graph = nx.from_numpy_array(similarity)
        scores = nx.pagerank(graph)

        ranked = sorted(
            ((scores[i], s) for i, s in enumerate(sentences)),
            reverse=True
        )

        return " ".join([s for _, s in ranked[:top_n]])
    except ValueError:
        # Fallback if TF-IDF fails (e.g., all stop words)
        return " ".join(sentences[:top_n])

def is_discourse_sentence(s):
    discourse_phrases = [
        "of course",
        "and that's",
        "and then",
        "you can",
        "we have to",
        "that's how",
        "i think",
        "you need",
        "we need to write",
        "we have to write",
        "simple notepad"
    ]

    s = s.lower()
    return any(p in s for p in discourse_phrases)

def has_concept_density(sentence, nlp):
    doc = nlp(sentence)

    noun_count = sum(1 for t in doc if t.pos_ in ["NOUN", "PROPN"])
    verb_count = sum(1 for t in doc if t.pos_ == "VERB")

    return noun_count >= 2 and verb_count >= 1

def normalize_concept_sentence(sentence):
    s = sentence.lower()

    mappings = {
        "write that code, your machine need to understand it":
            "Computer systems require code to be converted into machine-understandable instructions.",

        "work with that data, we need software":
            "Software is required to store, retrieve, and process data.",

        "we have to write some code":
            "Programming is used to instruct computers to perform tasks."
    }

    for k, v in mappings.items():
        if k in s:
            return v

    return sentence.strip().capitalize()

def is_question(sentence):
    return sentence.strip().endswith("?")

def starts_like_instruction(sentence):
    bad_starts = [
        "so let's",
        "let's",
        "but before",
        "and if you",
        "so what happens",
    ]

    s = sentence.lower().strip()
    return any(s.startswith(b) for b in bad_starts)

def is_filler_sentence(sentence):
    fillers = ["so ", "and ", "but ", "so, ", "and, ", "but, ", "that's it", "thats it"]
    s = sentence.lower().strip()
    return any(s.startswith(f) for f in fillers) or s in ["that's it.", "thats it."]

def meets_minimum_length(sentence, min_words=7):
    return len(sentence.split()) >= min_words



def generate_structured_notes(text, nlp, max_points=6):
    doc = nlp(text)
    notes = []

    for sent in doc.sents:
        s = sent.text.strip()

        if not meets_minimum_length(s):
            continue

        if is_question(s):
            continue

        if is_discourse_sentence(s):
            continue

        if starts_like_instruction(s):
            continue

        if is_filler_sentence(s):
            continue

        if not has_concept_density(s, nlp):
            continue

        clean = normalize_concept_sentence(s)
        notes.append("• " + clean)

        if len(notes) >= max_points:
            break

    return notes

def generate_quiz(text, num_questions=3):
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""
    You are an expert educator. Extract the {num_questions} most critical concepts from the following lecture transcript and create a multiple-choice quiz testing understanding of those specific concepts.
    
    Instructions:
    1. Identify the most important topics discussed in the audio transcript natively.
    2. Formulate exactly {num_questions} clear, unambiguous questions targeting these topics.
    3. Provide exactly 4 options per question where only 1 is correct.
    4. Ensure the 3 incorrect options (distractors) are highly plausible and strictly related to the subject matter, not easily guessable.
    
    Format the output EXACTLY as a valid JSON array of objects. Do not include any markdown formatting like ```json or ```.
    Each object must have the following structure:
    {{
        "question": "The question text testing the core concept",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "answer": "The EXACT string from the options list that is the correct answer"
    }}
    
    Transcript:
    {text}
    """
    
    import time
    import re
    import json
    
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            json_str = response.text.strip()
            
            # Use Regex to explicitly rip out the JSON array if Gemini adds conversational text
            match = re.search(r'\[.*\]', json_str, re.DOTALL)
            if match:
                json_str = match.group(0)
                
            quiz = json.loads(json_str)
            return quiz
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Quota exceeded" in error_str:
                return [{"error": "Rate limit exceeded. Please wait a minute before trying again."}]
            print(f"Error generating quiz (Attempt {attempt+1}): {e}")
            time.sleep(5)
            
    return []

def generate_flashcards(text, max_cards=6):
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""
    You are an expert educational AI. Analyze the following lecture transcript thoroughly.
    Your task is to create exactly {max_cards} highly effective study flashcards derived STRICTLY and UNIQUELY from the provided text.
    
    Instructions:
    1. Focus on the most critical concepts, definitions, and unique insights actually discussed in the audio transcript.
    2. Do NOT use general external knowledge; the flashcards must represent the specific content of this lecture.
    3. Ensure every answer is perfectly accurate, concise, and directly answers the question.
    
    Format the output EXACTLY as a valid JSON array of objects. Do not include any markdown formatting like ```json or ```.
    Each object must have the following structure:
    {{
        "question": "The specific question, concept, or term being asked about",
        "answer": "The perfectly accurate, concise definition or explanation derived from the text"
    }}
    
    Transcript:
    {text}
    """
    
    import time
    import re
    import json
    
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            json_str = response.text.strip()
            
            # Use Regex to explicitly rip out the JSON array
            match = re.search(r'\[.*\]', json_str, re.DOTALL)
            if match:
                json_str = match.group(0)
                
            flashcard_data = json.loads(json_str)
            return [(c["question"], c["answer"]) for c in flashcard_data]
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Quota exceeded" in error_str:
                return [("Rate Limit Exceeded", "You have exceeded your Gemini API free tier quota. Please wait a minute before generating more content.")]
            print(f"Error generating flashcards (Attempt {attempt+1}): {e}")
            time.sleep(5)
            
    return []

# Page configuration
st.set_page_config(
    page_title="LectureToStudyBuddy",
    page_icon="🎓",
    layout="wide"
)

# Premium CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Hide Sidebar Completely */
    [data-testid="stSidebar"] {display: none !important;}
    [data-testid="stSidebarNav"] {display: none !important;}
    [data-testid="collapsedControl"] {display: none !important;}

    /* Premium Header / Navigator */
    .nav-header {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(15px);
        padding: 15px 40px;
        border-radius: 0 0 20px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-bottom: 30px;
        position: sticky;
        top: 0;
        z-index: 999;
    }
    
    .nav-logo {
        font-size: 24px;
        font-weight: 700;
        color: #1e3a8a !important;
    }

    /* Glassmorphism Card Effect */
    .premium-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        margin-bottom: 25px;
        transition: transform 0.3s ease;
    }
    
    .premium-card:hover {
        transform: translateY(-5px);
    }

    /* Hero Section Override */
    .hero-text {
        color: #1e3a8a !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Feature Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background-color: #2563eb !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
    }
    

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.5);
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        color: #1e3a8a !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }

    /* --- HEADER CHATBOT STYLING --- */
    /* Popover Action Button */
    div[data-testid="stPopover"] button {
        background-color: #2563eb !important;
        border-radius: 12px !important;
        height: 3em !important;
        width: 100% !important;
        padding: 0 20px !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stPopover"] button:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
    }
    
    div[data-testid="stPopover"] button p,
    div[data-testid="stPopover"] button span {
        color: white !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stPopover"] button svg {
        display: none !important;
    }
    
    /* Internal Popover Panel Styling */
    div[data-testid="stPopoverBody"] {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        box-shadow: 0 15px 40px rgba(0,0,0,0.15) !important;
        padding: 25px !important;
        width: 450px !important; 
        height: 600px !important; 
        max-height: 80vh !important;
        overflow-y: auto !important;
    }
    
    /* Custom Chat Message Bubbles inside popover */
    div[data-testid="stPopoverBody"] .stChatMessage {
        background-color: transparent !important;
        padding: 10px 0 !important;
    }
    div[data-testid="stPopoverBody"] .stChatMessage.user {
        background-color: rgba(37, 99, 235, 0.1) !important;
        border-radius: 15px 15px 0 15px;
        padding: 15px !important;
        margin-bottom: 10px !important;
    }
    div[data-testid="stPopoverBody"] .stChatMessage.assistant {
        background-color: #f1f5f9 !important;
        border-radius: 15px 15px 15px 0;
        padding: 15px !important;
        margin-bottom: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "users" not in st.session_state:
    st.session_state.users = {}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 🔐 AUTHENTICATION GATE ---
if not st.session_state.logged_in:
    st.markdown("""
        <style>
            .stTabs [data-baseweb="tab-list"] { display: none; }
            .stTabs [data-baseweb="tab-panel"] { padding-top: 0px; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align:center; padding-top: 80px; margin-bottom: 20px;">
            <h1 class="hero-text" style="font-size: 3rem;">🎓 LectureToStudyBuddy</h1>
            <p style="font-size:20px; color: #4b5563;">AI-Powered Lecture Transcription & Study Assistant</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        mode = st.radio("Choose Action", ["Login", "Sign Up"], horizontal=True, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)

        if mode == "Login":
            username = st.text_input("Username", key="login_user", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 Sign In", use_container_width=True):
                if username in st.session_state.users and st.session_state.users[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.current_user = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")
                    
        else:
            new_user = st.text_input("New Username", key="signup_user", placeholder="Create a username")
            new_pass = st.text_input("New Password", type="password", key="signup_pass", placeholder="Create a password")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("✨ Create Account", use_container_width=True):
                if new_user in st.session_state.users:
                    st.warning("Username taken")
                elif new_user.strip() == "" or new_pass.strip() == "":
                    st.warning("Fields cannot be empty")
                else:
                    st.session_state.users[new_user] = new_pass
                    st.success("Account created! Select Login to continue.")
                    
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# --- 🧭 PREMIUM NAVIGATOR ---
# Restoring the premium header container look
st.markdown('<div class="nav-header">', unsafe_allow_html=True)
header_col1, header_col2, header_col3 = st.columns([5.5, 2, 1.5], gap="small")

with header_col1:
    st.markdown(f"""
    <div style="display: flex; align-items: center; height: 100%;">
        <div class="nav-logo" style="margin-right: 25px;">🎓 LectureToStudyBuddy</div>
        <span style="color: #4b5563; font-weight: 600; background: rgba(37, 99, 235, 0.1); padding: 8px 20px; border-radius: 25px; border: 1px solid rgba(255,255,255,0.5);">🧑‍🎓 {st.session_state.current_user}</span>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    if st.session_state.logged_in:
        with st.popover("🤖 AI Study Buddy", use_container_width=True):
            st.markdown("<h3 style='text-align: center; color: #1e3a8a; margin-bottom: 0;'>AI Study Buddy</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-size: 14px; color: #64748b;'>Ask me anything about the lecture!</p>", unsafe_allow_html=True)
            st.markdown("<hr style='margin-top: 10px; margin-bottom: 15px;'>", unsafe_allow_html=True)
            
            # Create a scrollable container for chat messages
            chat_container = st.container(height=450, border=False)
            
            # Display chat messages in the scrollable container
            with chat_container:
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                    
            # Chat Input at the bottom
            if prompt := st.chat_input("Ask a question about the lecture..."):
                # Add user message to state and display immediately in the container
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Process with Gemini
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                model = genai.GenerativeModel('gemini-2.5-flash')
                                
                                context = st.session_state.get('transcript', 'No lecture uploaded yet.')
                                full_prompt = f"""
                                You are a helpful study assistant. Use the following lecture transcript as context to answer the user's question.
                                If the answer is not in the context, use your general knowledge but mention that it wasn't strictly covered in the lecture.
                                
                                Lecture Transcript context:
                                {context}
                                
                                User Question: {prompt}
                                """
                                
                                response = model.generate_content(full_prompt)
                                st.markdown(response.text)
                                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                            except Exception as e:
                                error_str = str(e)
                                if "429" in error_str or "Quota exceeded" in error_str:
                                    error_msg = "⚠️ Rate limit exceeded. I am currently out of API quota. Please wait a minute before asking another question."
                                else:
                                    error_msg = f"Sorry, I encountered an error: {error_str}"
                                st.error(error_msg)
                                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

with header_col3:
    if st.button("🔓 Logout", key="header_logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 🏠 LECTURE CONVERTER MAIN INTERFACE ---
st.markdown(
    f"""
    <div style="text-align:center; padding-bottom: 20px; padding-top: 10px;">
        <p style="font-size:32px; color: #1e3a8a; font-weight: 700; margin-bottom: 5px;">Welcome back, {st.session_state.current_user}! 👋</p>
        <p style="font-size: 18px; color: #64748b;">Ready to transform your audio into knowledge?</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="premium-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload audio recording (MP3, WAV, M4A)",
    type=["mp3", "wav", "m4a"]
)

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("🚀 Start Transcription"):
        with st.spinner("🔍 AI is processing your lecture recording..."):
            try:
                processor = AudioProcessor()

                # ✅ PASS FILE OBJECT (NOT PATH)
                transcript = processor.process_audio(uploaded_file)

                # NLP processing
                transcript = clean_text(transcript)
                st.session_state["transcript"] = transcript
                st.session_state["notes"] = generate_structured_notes(transcript, nlp)
                st.session_state["summary"] = generate_summary(st.session_state["notes"])
                st.session_state["quiz"] = generate_quiz(transcript)
                st.session_state["flashcards"] = generate_flashcards(transcript)

                st.success("✅ Success! Your lecture has been converted.")

            except Exception as e:
                st.error(f"Error during transcription: {e}")
st.markdown('</div>', unsafe_allow_html=True)

if 'transcript' in st.session_state:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 Notes", "📌 Summary", "❓ Quiz", "🧠 Flashcards"]
    )

    with tab1:
        st.subheader("Structured Notes")
        for n in st.session_state.get('notes', []):
            st.write(n)

    with tab2:
        st.subheader("Summary")
        st.write(st.session_state.get('summary', ''))

    with tab3:
        st.subheader("Interactive Quiz")

        if "score" not in st.session_state:
            st.session_state.score = 0

        quiz_data = st.session_state.get('quiz', [])
        if quiz_data:
            for i, q in enumerate(quiz_data):
                st.markdown(f"**Q{i+1}. {q['question']}**")

                user_answer = st.radio(
                    f"Choose an answer for Q{i+1}",
                    q["options"],
                    key=f"quiz_{i}"
                )

                if st.button(f"Submit Q{i+1}", key=f"submit_{i}"):
                    if user_answer == q["answer"]:
                        st.success("✅ Correct!")
                        st.session_state.score += 1
                    else:
                        st.error(f"❌ Wrong! Correct answer: {q['answer']}")

                st.markdown("---")

            st.info(f"🎯 Your Score: {st.session_state.score} / {len(quiz_data)}")
        else:
            st.info("No quiz questions available yet. Please transcribe a lecture.")

    with tab4:
        st.subheader("Interactive Flashcards")
        st.markdown('<p style="color: #64748b; font-size: 14px; text-align: center;">Click and hold (or tap) a card to see the answer!</p>', unsafe_allow_html=True)
        # Use mock flashcards if none exist yet for demonstration purposes
        flashcards = st.session_state.get('flashcards', [
            ("What is the main advantage of the new interactive flashcard UI?", "It uses an isolated iframe component to bypass Streamlit's Markdown sanitizer, ensuring smooth 3D CSS animations and JavaScript interactions work reliably."),
            ("How does Streamlit typically handle HTML `onclick` attributes?", "For security reasons, Streamlit strips inline JavaScript like `onclick` when rendering HTML via `st.markdown(..., unsafe_allow_html=True)`."),
            ("What does the `transform-style: preserve-3d;` CSS property do?", "It indicates that the element's children should be positioned in 3D space, which is essential for creating realistic flip animations for the front and back of the cards.")
        ])
        
        if flashcards:
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap');
                body {
                    font-family: 'Outfit', sans-serif;
                    margin: 0;
                    padding: 10px;
                }
                .flashcards-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                    gap: 20px;
                    padding: 10px;
                }
                .flashcard {
                    background-color: transparent;
                    perspective: 1000px;
                    height: 250px;
                    cursor: pointer;
                }
                .flashcard-inner {
                    position: relative;
                    width: 100%;
                    height: 100%;
                    text-align: center;
                    transition: transform 0.6s cubic-bezier(0.4, 0.2, 0.2, 1);
                    transform-style: preserve-3d;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    border-radius: 20px;
                }
                .flashcard.flipped .flashcard-inner {
                    transform: rotateY(180deg);
                }
                .flashcard-front, .flashcard-back {
                    position: absolute;
                    width: 100%;
                    height: 100%;
                    -webkit-backface-visibility: hidden;
                    backface-visibility: hidden;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 24px;
                    border-radius: 20px;
                    box-sizing: border-box;
                }
                .flashcard-front {
                    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
                    color: #1e3a8a;
                    border: 1px solid #e2e8f0;
                }
                .flashcard-back {
                    background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
                    color: white;
                    transform: rotateY(180deg);
                }
                .flashcard-label {
                    font-size: 13px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    margin-bottom: 12px;
                    opacity: 0.8;
                }
                .flashcard-content {
                    font-size: 18px;
                    font-weight: 600;
                    line-height: 1.5;
                }
            </style>
            </head>
            <body>
                <div class="flashcards-grid">
            """
            
            for i, (q, a) in enumerate(flashcards):
                # Escape potential quotes in text
                safe_q = q.replace("'", "&#39;").replace('"', '&quot;')
                safe_a = a.replace("'", "&#39;").replace('"', '&quot;')
                
                html_content += f"""
                    <div class="flashcard" onclick="this.classList.toggle('flipped')">
                        <div class="flashcard-inner">
                            <div class="flashcard-front">
                                <div class="flashcard-label">Question</div>
                                <div class="flashcard-content">{safe_q}</div>
                            </div>
                            <div class="flashcard-back">
                                <div class="flashcard-label">Answer</div>
                                <div class="flashcard-content">{safe_a}</div>
                            </div>
                        </div>
                    </div>
                """
                
            html_content += """
                </div>
            </body>
            </html>
            """
            
            import streamlit.components.v1 as components
            # We calculate height based on number of cards. 
            # 1 column mode: ~270px per row. Let's assume 2 columns on average desktop.
            rows = max(1, (len(flashcards) + 1) // 2)
            components.html(html_content, height=rows * 300)
        else:
            st.info("No flashcards could be generated. Try uploading a longer lecture recording.")
    
    st.markdown("---")
    st.markdown("#### 📄 Full Transcript")
    with st.expander("View Full Transcript"):
        st.write(st.session_state['transcript'])
    
    st.markdown("#### 📂 Export Options")
    st.download_button(
        label="📥 Download as TXT",
        data=st.session_state['transcript'],
        file_name="lecture_transcript.txt",
        mime="text/plain"
    )
    st.markdown('</div>', unsafe_allow_html=True)




