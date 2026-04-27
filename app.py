import base64
import os
import tempfile
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

# Import Vertex AI generative models
import vertexai
from vertexai.generative_models import GenerativeModel,SafetySetting,GenerateContentConfig,SpeechConfig,VoiceConfig,PrebuiltVoiceConfig,Content,Part
# Optionally load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize Vertex AI
PROJECT_ID = os.getenv("PROJECT_ID", "inoday-retail")
REGION = os.getenv("REGION", "us-central1")
vertexai.init(project=PROJECT_ID, location=REGION)

app = FastAPI(title="Voice-Driven Gemini Audio Bot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_audio_response(input_audio_b64: str) -> bytes:
    """
    Uses the Gemini model (gemini-2.5-pro-exp-03-25) to generate an audio response
    from an audio input (provided as a base64-encoded string). It sends the audio input
    along with a text instruction and requests an output in AUDIO modality.
    Returns the generated audio bytes.
    """
    # Create an audio part from the input audio bytes
    audio_part = Part.from_bytes(
        data=base64.b64decode(input_audio_b64),
        mime_type="audio/mpeg",
    )
    
    # Create a text instruction part
    instruction_part = Part.from_text(
        text="Provide a concise troubleshooting solution in audio format for the question asked in the audio."
    )
    
    contents = [
        Content(
            role="user",
            parts=[audio_part, instruction_part]
        )
    ]
    
    # Configure the generation to output AUDIO
    config = GenerateContentConfig(
        temperature=1,
        top_p=1,
        seed=0,
        max_output_tokens=65535,
        response_modalities=["AUDIO"],
        speech_config=SpeechConfig(
            voice_config=VoiceConfig(
                prebuilt_voice_config=PrebuiltVoiceConfig(
                    voice_name="zephyr"  # Adjust the voice as needed
                )
            )
        ),
        safety_settings=[
            SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]
    )
    
    generated_audio = b""
    model = GenerativeModel("gemini-2.5-pro-exp-03-25")
    for chunk in model.generate_content_stream(
        contents=contents,
        config=config,
        stream=True,
    ):
        if hasattr(chunk, "audio"):
            generated_audio += chunk.audio
        else:
            raise HTTPException(status_code=500, detail="Model did not return audio output.")
    
    return generated_audio

@app.post("/voice_generate")
async def voice_generate(audio_file: UploadFile = File(...)):
    """
    Endpoint that accepts an audio file (e.g. captured via microphone or uploaded by the client),
    sends it to the Gemini model (as a base64-encoded blob) for processing, and returns an MP3 audio file
    with the generated solution.
    """
    try:
        audio_bytes = await audio_file.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio file: {e}")
    
    generated_audio_bytes = generate_audio_response(audio_b64)
    
    # Write the generated audio bytes to a temporary file
    temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    with open(temp_audio_path, "wb") as f:
        f.write(generated_audio_bytes)
    
    # Return the MP3 file as a FileResponse (for local testing)
    return FileResponse(temp_audio_path, media_type="audio/mpeg", filename="generated_solution.mp3")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
