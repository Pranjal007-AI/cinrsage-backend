from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import httpx
import io
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

model = ChatMistralAI(
    model="mistral-large-latest",
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
)

Prompts = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert AI assistant specialized in extracting key information and summarizing content clearly.
Your responses must be accurate, structured, and easy to read.
Do not hallucinate. If information is missing, simply skip it.
Think internally but do not show reasoning.
CRITICAL: Do NOT use markdown formatting. No asterisks (*), no bold (**text**), no # headers. Plain text only.
"""),
    ("human", """
Analyze the text below and produce a well-structured output.
STRICT RULES:
- No markdown, no asterisks (*), no bold, no # headers. Plain text only.
- Use EXACTLY the labels below, nothing else.
- Each list item must start with a dash (-)
========================
INPUT:
{input_text}
========================
Title:
Category:
Key Entities:
-
Key Points:
-
Important Data:
-
Timeline (if any):
-
Insights:
-
Sentiment:
Summary:
""")
])


# ── Request Models ────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    input_text: str

class VoiceRequest(BaseModel):
    text: str

class ImageRequest(BaseModel):
    prompt: str


# ── Root ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "CinrSage Backend is running!"}


# ── Text Analysis ─────────────────────────────────────────────────
@app.options("/analyze")
async def options_analyze():
    return JSONResponse(content={"message": "OK"})

@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    final_prompt = Prompts.invoke({"input_text": request.input_text})
    result = model.invoke(final_prompt)
    return {"result": result.content}


# ── Voice Generation (ElevenLabs) ─────────────────────────────────
@app.options("/voice")
async def options_voice():
    return JSONResponse(content={"message": "OK"})

@app.post("/voice")
async def generate_voice(request: VoiceRequest):
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # Default: George

    # Trim text to 500 chars for free tier
    text = request.text[:500]

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            return JSONResponse(
                status_code=response.status_code,
                content={"error": "ElevenLabs API error", "detail": response.text}
            )
        audio_bytes = response.content

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=summary.mp3"}
    )


# ── Image Generation (HuggingFace FLUX.1-schnell) ─────────────────
@app.options("/generate-image")
async def options_image():
    return JSONResponse(content={"message": "OK"})

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

   url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
   payload = {
    "inputs": request.prompt
}
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            return JSONResponse(
                status_code=response.status_code,
                content={"error": "HuggingFace API error", "detail": response.text}
            )
        image_bytes = response.content

    return StreamingResponse(
        io.BytesIO(image_bytes),
        media_type="image/jpeg",
        headers={"Content-Disposition": "inline; filename=generated.jpg"}
    )
