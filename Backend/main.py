from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

import spacy
import joblib

from transformers import MarianMTModel, MarianTokenizer


# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI(title="NutriCheck API")


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

NER_MODEL_PATH = PROJECT_DIR / "modeloNER_6_entidades"

ML_MODEL_PATH = PROJECT_DIR / "medication_allergy_model.pkl"

TFIDF_PATH = PROJECT_DIR / "medication_allergy_tfidf.pkl"

FRONTEND_DIR = PROJECT_DIR / "Frontend"


# ============================================================
# LOAD NER MODEL
# ============================================================

nlp = spacy.load(NER_MODEL_PATH)

print("NER model loaded successfully")
print("NER labels:", nlp.get_pipe("ner").labels)


# ============================================================
# LOAD MEDICATION ALLERGY MODEL
# ============================================================

allergy_model = joblib.load(ML_MODEL_PATH)

print("Medication allergy model loaded successfully")


# ============================================================
# LOAD TF-IDF
# ============================================================

tfidf_vectorizer = joblib.load(TFIDF_PATH)

print("TF-IDF vectorizer loaded successfully")


# ============================================================
# LOAD PORTUGUESE -> ENGLISH TRANSLATION MODEL
# ============================================================

TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-ROMANCE-en"

translator_tokenizer = MarianTokenizer.from_pretrained(
    TRANSLATION_MODEL_NAME
)

translator_model = MarianMTModel.from_pretrained(
    TRANSLATION_MODEL_NAME
)

print("Portuguese to English translation model loaded successfully")


# ============================================================
# TRANSLATION FUNCTION
# ============================================================

def translate_to_english(text):

    if not text:
        return ""

    inputs = translator_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    translated = translator_model.generate(
        **inputs
    )

    result = translator_tokenizer.batch_decode(
        translated,
        skip_special_tokens=True
    )

    return result[0]


# ============================================================
# REQUEST MODEL
# ============================================================

class MedicalText(BaseModel):
    text: str


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def home():

    return FileResponse(
        FRONTEND_DIR / "index.html"
    )


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(data: MedicalText):

    text = data.text

    # ========================================================
    # 1. NER PREDICTION
    # ========================================================

    doc = nlp(text)

    entities = {
        "symptoms": [],
        "medications": [],
        "treatments": [],
        "conditions": [],
        "diagnosis": [],
        "exams": []
    }


    # ========================================================
    # 2. EXTRACT SIX ENTITIES
    # ========================================================

    for ent in doc.ents:

        label = ent.label_.lower()

        # Translate Portuguese entity to English
        value = translate_to_english(ent.text)

        if "symptom" in label:

            entities["symptoms"].append(value)

        elif "medication" in label:

            entities["medications"].append(value)

        elif "treatment" in label:

            entities["treatments"].append(value)

        elif "condition" in label:

            entities["conditions"].append(value)

        elif "diagnos" in label:

            entities["diagnosis"].append(value)

        elif "exam" in label:

            entities["exams"].append(value)


    # ========================================================
    # 3. CREATE ML INPUT
    # ========================================================

    combined_text = (
        "symptom: " + " ".join(entities["symptoms"]) +
        " medication: " + " ".join(entities["medications"]) +
        " treatment: " + " ".join(entities["treatments"]) +
        " diagnosis: " + " ".join(entities["diagnosis"]) +
        " condition: " + " ".join(entities["conditions"]) +
        " exam: " + " ".join(entities["exams"])
    )


    # ========================================================
    # 4. TF-IDF
    # ========================================================

    ml_input = tfidf_vectorizer.transform(
        [combined_text]
    )


    # ========================================================
    # 5. MEDICATION ALLERGY PREDICTION
    # ========================================================

    medication_allergy = int(
        allergy_model.predict(ml_input)[0]
    )


    # ========================================================
    # 6. FINAL RESPONSE
    # ========================================================

    return {
        "entities": entities,
        "medication_allergy": medication_allergy
    }


# ============================================================
# SERVE FRONTEND
# ============================================================

app.mount(
    "/",
    StaticFiles(
        directory=FRONTEND_DIR,
        html=True
    ),
    name="frontend"
)