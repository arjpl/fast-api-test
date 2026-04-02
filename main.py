
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import spacy

# load spacy model
nlp = spacy.load("en_core_web_sm")

app = FastAPI(
    title="Text Analysis API",
    description="Send text and get back NLP insights such as sentiment, entities, keywords, etc",
    version = "1.0.0",
)

# input model for the API requests
class TextAnalysisInput(BaseModel):
    text: str
    include_entities: Optional[bool] = True
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Apple, Anthropic, etc are leading the tech world.",
                    "include_entities": True,
                }
            ]
        }
    }
    
# output model / response model
class TextAnalysisOutput(BaseModel):
    word_count: int
    sentence_count: int
    tokens: list[str]
    entities: Optional[list[dict]] = None
    noun_chunks: list[str]
    top_keywords: list[str]
    
