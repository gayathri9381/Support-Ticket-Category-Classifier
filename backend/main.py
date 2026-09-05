import os
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from backend.utils import clean_text

app = FastAPI(
    title="Customer Support Ticket Triage API",
    description="Inference API to categorize support tickets into departments.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "ticket_classifier_pipeline.pkl"
)
model_pipeline = None


@app.on_event("startup")
def load_model():
  global model_pipeline
  if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model file not found at {MODEL_PATH}")
    return
  try:
    with open(MODEL_PATH, "rb") as f:
      model_pipeline = pickle.load(f)
    print("Model pipeline loaded successfully.")
  except Exception as e:
    print(f"\n[ERROR LOADING MODEL]: {e}\n")
    import traceback

    traceback.print_exc()


class TicketRequest(BaseModel):
  customer_name: str
  customer_email: EmailStr
  customer_age: int
  customer_gender: str
  product_purchased: str
  ticket_subject: str
  ticket_description: str
  ticket_priority: str
  ticket_channel: str


class TicketResponse(BaseModel):
  ticket_type: str
  confidence_score: float | None
  assigned_queue: str


@app.get("/health")
def health_check():
  return {"status": "ok", "model_loaded": model_pipeline is not None}


@app.post("/predict", response_model=TicketResponse)
def predict_ticket(ticket: TicketRequest):
  if model_pipeline is None:
    raise HTTPException(
        status_code=503,
        detail=(
            "Model failed to load. Check backend console for unpickling error."
        ),
    )

  combined_text = f"{ticket.ticket_subject} {ticket.ticket_description}"
  processed_text = clean_text(combined_text)

  prediction = model_pipeline.predict([processed_text])[0]

  confidence = None
  if hasattr(model_pipeline, "predict_proba"):
    probabilities = model_pipeline.predict_proba([processed_text])[0]
    confidence = float(max(probabilities))

  return TicketResponse(
      ticket_type=str(prediction),
      confidence_score=confidence,
      assigned_queue=f"{prediction} Operations Queue",
  )