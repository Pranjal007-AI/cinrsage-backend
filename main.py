from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

load_dotenv()

app = FastAPI()

# Allow requests from your Netlify frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your Netlify URL after deploy
    allow_methods=["POST"],
    allow_headers=["*"],
)

model = ChatMistralAI(
    model="mistral-large-latest",
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
)

Prompts = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert AI assistant specialized in:
- extracting key information
- summarizing content clearly

Your responses must be accurate, structured, and easy to read.

Do not hallucinate.
If information is missing, simply skip it.

Think internally but do not show reasoning.
"""),
    ("human", """
Analyze the text below and produce a well-structured output.

========================
INPUT:
{input_text}
========================

### Your Output Must Follow This Format:

Title:
<main topic or heading>

Category:
<type like Movie, Article, News, Product, etc.>

Key Entities:
- ...
- ...
- ...

Key Points:
- ...
- ...
- ...

Important Data:
- ...
- ...

Timeline (if any):
- ...
- ...

Insights:
- ...
- ...

Sentiment:
<Positive / Neutral / Negative>

Summary:
<clear, concise paragraph under 120 words>

---

### Rules:
- Keep it clean and readable
- Do not add extra explanations
- Do not repeat information
- Be precise and factual
""")
])


class AnalyzeRequest(BaseModel):
    input_text: str


@app.get("/")
def root():
    return {"status": "CinrSage Backend is running!"}


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    final_prompt = Prompts.invoke({"input_text": request.input_text})
    result = model.invoke(final_prompt)
    return {"result": result.content}
