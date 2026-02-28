# 📘 Lecture to Study Buddy

An AI-powered web application that converts lecture audio recordings into **structured notes, summaries, quizzes, and flashcards**, helping students learn more effectively through NLP and Generative AI.

---

## 🚀 Project Overview

Students often struggle to revise long lecture recordings. This project solves that problem by automatically converting lecture audio into concise and interactive study material using **Speech-to-Text**, **Natural Language Processing (NLP)**, and **Generative AI**.

The system allows users to:
- Upload lecture audio
- Generate structured notes
- View concise summaries
- Practice with quizzes
- Revise using flashcards
- Interact with an AI Study Buddy chatbot

---

## 🎯 Objectives

- Convert lecture audio into text using Speech Recognition
- Extract key concepts using NLP techniques
- Generate summaries and structured notes
- Create interactive quizzes and flashcards
- Provide an intuitive and user-friendly web interface
- Reduce manual effort in note-taking

---

## 🧠 Technologies Used

### 🔹 Frontend & UI
- **Streamlit** – Interactive web application framework

### 🔹 Speech Processing
- **Whisper** – Speech-to-text transcription

### 🔹 NLP (Machine Learning)
- **NLTK** – Text preprocessing and tokenization  
- **spaCy** – Linguistic analysis and keyword extraction  
- **Scikit-learn** – TF-IDF based text summarization  

### 🔹 Generative AI
- **Google Gemini API** – Quiz, flashcard, and chatbot generation

---

## 🏗️ System Architecture

1. User uploads lecture audio (MP3/WAV/M4A)
2. Audio is transcribed using Whisper
3. Transcription is processed using NLP techniques
4. Structured notes and summaries are generated
5. Gemini AI generates quizzes and flashcards
6. Results are displayed through Streamlit UI

---

## ⚙️ Algorithm (High-Level)

1. Accept lecture audio input
2. Perform speech-to-text conversion
3. Clean and preprocess text
4. Apply NLP techniques:
   - Sentence segmentation
   - Keyword extraction
   - TF-IDF scoring
5. Generate structured notes and summary
6. Generate quiz and flashcards using GenAI
7. Display outputs to the user

---

## 🖥️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/batch---7-AICTE.git
cd batch---7-AICTE
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download NLP Models
```bash
python -m nltk.downloader punkt stopwords
python -m spacy download en_core_web_sm
```

### 4️⃣ Add Environment Variables
Create a `.env` file:
```env
GEMINI_API_KEY=your_api_key_here
```

### 5️⃣ Run the Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
lecture_to_study_buddy/
│
├── app.py
├── audio_processor.py
├── requirements.txt
├── README.md
├── .env
├── pages/
└── .streamlit/
```

---

## 📊 Results

- Accurate transcription of lecture audio
- Meaningful structured notes generated using NLP
- Concise summaries for quick revision
- Interactive quizzes and flashcards
- Improved learning efficiency and engagement

---

## 🧪 Limitations

- Free-tier API rate limits
- Accuracy depends on audio quality
- Very long lectures may take more processing time

---

## 🔮 Future Scope

- Multilingual lecture support
- PDF & PPT upload support
- Student performance analytics
- Personalized learning recommendations
- Offline processing mode
- Mobile application version

---

## 📚 References

- Whisper: https://github.com/openai/whisper  
- Streamlit: https://streamlit.io  
- NLTK: https://www.nltk.org  
- spaCy: https://spacy.io  
- Google Gemini API: https://ai.google.dev  

---

## 👨‍🎓 Internship Details

- **Internship:** AICTE – IBM SkillsBuild / Edunet Foundation  
- **Project Type:** AI + NLP + ML  
- **Level:** Academic / Internship Project  

---

### ✅ Final Note
This project demonstrates the **practical application of NLP and AI** in education and fulfills internship evaluation requirements successfully.
