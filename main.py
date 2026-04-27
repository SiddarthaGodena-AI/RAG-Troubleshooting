import os
import re
import subprocess
import tempfile
import time
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Google Cloud clients for Storage, Speech, and TTS
from google.cloud import storage, speech_v1p1beta1 as speech, texttospeech
# For Vertex AI pre-trained Gemini model
from vertexai.generative_models import GenerativeModel, SafetySetting
# For loading credentials
from google.oauth2 import service_account
# For PDF extraction
from PyPDF2 import PdfReader
from io import BytesIO

# Optionally load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize credentials and Vertex AI
key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", r"ABCD123#")
credentials = service_account.Credentials.from_service_account_file(
    key_path, scopes=['https://www.googleapis.com/auth/cloud-platform']
)
PROJECT_ID = os.getenv("PROJECT_ID", "inoday-retail")
REGION = os.getenv("REGION", "us-central1")
import vertexai
vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)

app = FastAPI(title="Voice-Driven Troubleshooting RAG App")

# Global variable to store document segments loaded from GCS or a local PDF/text file.
DOCUMENT_SEGMENTS: List[str] = []


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes using PyPDF2.
    """
    pdf_stream = BytesIO(pdf_bytes)
    reader = PdfReader(pdf_stream)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def load_document_from_gcs(bucket_name: str, file_name: str) -> List[str]:
    """
    Load a document from Google Cloud Storage.
    If the file is a PDF, extract text using PyPDF2; otherwise, assume it's a text file.
    Split the resulting text into segments based on double newlines.
    """
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    if file_name.lower().endswith('.pdf'):
        pdf_bytes = blob.download_as_bytes()
        content = extract_text_from_pdf(pdf_bytes)
    else:
        content = blob.download_as_text(encoding="utf-8")
    
    segments = [seg.strip() for seg in content.split("\n\n") if seg.strip()]
    return segments


def load_document_from_local(file_path: str) -> List[str]:
    """
    Load a local document.
    If it's a PDF, extract text using PyPDF2; otherwise, load as UTF-8 text.
    """
    if file_path.lower().endswith('.pdf'):
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        content = extract_text_from_pdf(pdf_bytes)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    segments = [seg.strip() for seg in content.split("\n\n") if seg.strip()]
    return segments


def build_vector_index(docs: List[str]):
    """
    Build a TF-IDF index for a list of document segments.
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(docs)
    return vectorizer, vectors


def retrieve_relevant(docs: List[str], query: str) -> str:
    """
    Retrieve the document segment most relevant to the query.
    """
    vectorizer, vectors = build_vector_index(docs)
    query_vec = vectorizer.transform([query])
    cos_sim = cosine_similarity(query_vec, vectors).flatten()
    best_idx = cos_sim.argmax()
    return docs[best_idx]


def convert_audio_to_wav(input_path: str) -> str:
    """
    Converts any input audio file (mp3, flac, etc.) to WAV (LINEAR16)
    using ffmpeg. Returns the path to the converted WAV file.
    """
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    command = [
        "ffmpeg",
        "-y",  # Overwrite if exists
        "-i", input_path,
        "-ac", "1",  # Force mono channel
        "-ar", "16000",  # 16kHz sampling rate
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Audio conversion failed: {e.stderr.decode()}")
    return output_path


def transcribe_audio(audio_file_path: str) -> str:
    """
    Uses Google Cloud Speech-to-Text to transcribe an audio file.
    Assumes audio is in LINEAR16 encoding (WAV).
    """
    client = speech.SpeechClient(credentials=credentials)
    with open(audio_file_path, "rb") as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US"
    )
    response = client.recognize(config=config, audio=audio)
    transcript = " ".join(result.alternatives[0].transcript for result in response.results)
    return transcript


def converse_with_gemini(user_message: str, base_solution: str) -> str:
    """
    Uses the pre-trained Gemini model (gemini-1.5-pro-001) to generate a detailed troubleshooting solution.
    Combines the user transcript and the retrieved base solution into a prompt.
    """
    prompt = f"""
    Analyze the following conversation transcript and provide a detailed troubleshooting solution either consideer data from the document or from using gemini capabilities but do not say anything about where the data was provided from.
    
    Transcript:
    {user_message}
    
    Base Information:
    {base_solution}
    
    Provide a clear, detailed solution:
    """
    model = GenerativeModel("gemini-1.5-pro-001")
    generation_config = {
        "max_output_tokens": 1500,
        "temperature": 0.2,
        "top_p": 0.95,
    }
    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
    ]
    responses = model.generate_content(
        [prompt],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )
    generated_response = ""
    for response in responses:
        generated_response += response.text
    return generated_response


def text_to_speech(text: str) -> str:
    """
    Uses Google Cloud Text-to-Speech to convert text into an MP3 audio file.
    Returns the local path to the generated MP3 file.
    """
    client = texttospeech.TextToSpeechClient(credentials=credentials)
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Wavenet-F",  # Female Wavenet voice
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_audio.write(response.audio_content)
    temp_audio.close()
    return temp_audio.name


def clean_text(text: str) -> str:
    """
    Clean unwanted characters such as '*', '#', '/', '\' from the text.
    Also remove specific prompt labels so the output audio only provides the solution.
    """
    # Remove special characters: *, #, /, \
    cleaned = re.sub(r"[\*\#\/\\]", "", text)
    # Remove prompt labels if present
    for keyword in ["Transcript:", "Base Information:", "Provide a clear, detailed solution:"]:
        cleaned = cleaned.replace(keyword, "")
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@app.on_event("startup")
async def startup_event():
    """
    On startup, load the troubleshooting document.
    If environment variables for GCS are set, load from GCS;
    otherwise, load from a local file.
    """
    global DOCUMENT_SEGMENTS
    bucket_name = os.getenv("BUCKET_NAME")  # e.g., "troubleshootingdata"
    file_name = os.getenv("FILE_NAME")      # e.g., "Troubleshooting_WLAN_Issues.pdf"
    if bucket_name and file_name:
        DOCUMENT_SEGMENTS = load_document_from_gcs(bucket_name, file_name)
    else:
        local_file = "network_troubleshooting.pdf"
        DOCUMENT_SEGMENTS = load_document_from_local(local_file)
    if not DOCUMENT_SEGMENTS:
        raise Exception("No document segments were loaded.")


@app.post("/voice_troubleshoot", response_class=FileResponse)
async def voice_troubleshoot(audio_file: UploadFile = File(...)):
    """
    Endpoint for a completely voice-driven troubleshooting interaction.
    Accepts any common audio file (WAV, MP3, FLAC), converts it to a WAV
    (mono, 16kHz), transcribes the audio, retrieves a relevant troubleshooting tip,
    generates a solution using Gemini, cleans the output, converts the solution to speech (MP3),
    and returns the audio file.
    """
    # Save the uploaded audio file to a temporary file.
    temp_input = tempfile.NamedTemporaryFile(delete=False)
    temp_input.write(await audio_file.read())
    temp_input.close()

    # ALWAYS convert the input audio to WAV (mono, 16kHz)
    temp_input_path = convert_audio_to_wav(temp_input.name)
    os.remove(temp_input.name)

    # 1. Transcribe the audio to text.
    transcript = transcribe_audio(temp_input_path)
    if not transcript:
        raise HTTPException(status_code=500, detail="Failed to transcribe audio.")

    # 2. Retrieve the most relevant base solution from the loaded document.
    base_solution = retrieve_relevant(DOCUMENT_SEGMENTS, transcript)

    # 3. Generate a detailed troubleshooting solution via Gemini.
    detailed_solution = converse_with_gemini(transcript, base_solution)

    # 4. Clean the generated solution so it doesn't include unwanted characters or prompt references.
    cleaned_solution = clean_text(detailed_solution)

    # 5. Convert the cleaned solution to speech (MP3).
    output_audio_path = text_to_speech(cleaned_solution)

    # Clean up the temporary converted file.
    os.remove(temp_input_path)

    # Return the MP3 file as the response.
    return FileResponse(output_audio_path, media_type="audio/mpeg", filename="solution.mp3")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
