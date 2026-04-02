
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
    
def extract_keywords(doc) -> list[str]:
    keywords = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and token.pos_ in ("NOUN", "PROPN", "VERB", "ADJ")
    ]
    
    freq = {}
    for word in keywords:
        freq[word] = freq.get(word,0) + 1
    sorted_keywords = sorted(freq, key=freq.get, reverse=True)
    return sorted_keywords[:10]

@app.post("/analyze", response_model = TextAnalysisOutput, tags = ["NLP"])
def analyze_text(input: TextAnalysisInput):
    """
    Analyse text and return word count, sentences, tokens,
    named entities, noun chunks, and top keywords.
    """
    if not input.text.strip():
        raise HTTPException(status_code = 400,
                            detail = "Text cannot be empty")
        
    doc = nlp(input.text)
    
    entities = None
    if input.include_entities:
        entities = [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
        ]
    
    return TextAnalysisOutput(
        word_count=len([t for t in doc if not t.is_space]),
        sentence_count=len(list(doc.sents)),
        tokens=[token.text for token in doc if not token.is_punct and not token.is_space],
        entities=entities,
        noun_chunks=[chunk.text for chunk in doc.noun_chunks],
        top_keywords=extract_keywords(doc),
    )
    
    
