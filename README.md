# RAG-Troubleshooting
An end-to-end AI-powered voice troubleshooting assistant built using FastAPI, Google Vertex AI (Gemini), and Google Cloud services.

This project enables users to upload audio queries and receive intelligent troubleshooting solutions in audio format, powered by Retrieval-Augmented Generation (RAG).

🚀 Features
🔊 Voice-Based Interaction
Upload audio queries (MP3/WAV/FLAC)
Automatic speech-to-text conversion using Google Speech-to-Text

📚 RAG (Retrieval-Augmented Generation)
Loads troubleshooting documents (PDF/Text) from:
Google Cloud Storage ☁️
Local files 📂
Splits into semantic chunks
Uses TF-IDF + Cosine Similarity for retrieval

🤖 AI-Powered Solution Generation
Uses Gemini (Vertex AI) for:
Context-aware troubleshooting responses
Combining user query + document knowledge

🔉 Audio Response Generation
Converts generated solutions into speech using:
Google Text-to-Speech
Returns final answer as downloadable MP3

🎧 Advanced Audio Pipeline
Audio format normalization (via ffmpeg)
Noise-resilient transcription flow

🧠 Fully Voice-Driven Flow
User Audio → Transcription → Retrieval → LLM → Clean Text → TTS → Audio Response

🧩 Project Structure
├── app.py                  # Gemini Audio-to-Audio API
├── main.py                 # Full RAG Voice Troubleshooting Pipeline
├── network_troubleshooting.pdf  # Example knowledge base
├── .env                    # Environment variables

⚙️ Tech Stack
Component	Technology
Backend	FastAPI
LLM	Gemini (Vertex AI)
Speech-to-Text	Google Cloud Speech API
Text-to-Speech	Google Cloud TTS
Storage	Google Cloud Storage
Retrieval	TF-IDF + Cosine Similarity
Audio Processing	FFmpeg
PDF Parsing	PyPDF2

🔑 Setup Instructions
1️⃣ Clone Repository
git clone https://github.com/your-username/voice-rag-troubleshooting.git
cd voice-rag-troubleshooting

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Install System Dependencies
sudo apt update
sudo apt install ffmpeg

4️⃣ Configure Environment Variables
Create a .env file:
PROJECT_ID=your-gcp-project-id
REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# Optional (for GCS document loading)
BUCKET_NAME=your-bucket-name
FILE_NAME=your-document.pdf

▶️ Running the Application
Start the server:
python main.py
or
uvicorn main:app --reload

📡 API Endpoints
🎤 Voice Troubleshooting Endpoint
POST /voice_troubleshoot
Request:
audio_file: Upload audio file
Response:
Returns an MP3 audio file with the solution

🔊 Gemini Audio-to-Audio Endpoint
POST /voice_generate
Description:
Directly sends audio to Gemini
Returns AI-generated audio response

🔄 Workflow
Upload audio query
Convert audio → WAV (16kHz mono)
Transcribe speech → text
Retrieve relevant document chunk
Generate solution using Gemini
Clean response text
Convert text → speech (MP3)
Return audio response
