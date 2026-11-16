# NutriCheck Frontend

A bluish-black medical NLP dashboard for the NutriCheck project.

## Files

- index.html — main interface
- style.css — complete visual styling
- script.js — API connection and result handling

## Run

1. Start your FastAPI backend.
2. Make sure the prediction endpoint is:
   http://127.0.0.1:8000/predict
3. Open `index.html` in a browser.

The frontend currently sends:

POST /predict

JSON:
{
  "text": "medical record text"
}

If your FastAPI request field or endpoint is different, edit `API_URL` and the request body in `script.js`.
