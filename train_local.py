import os
import pickle
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


def clean_text(text: str) -> str:
  text = str(text).lower()
  text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
  text = re.sub(r"\<.*?\>", "", text)
  text = text.translate(str.maketrans("", "", string.punctuation))
  text = re.sub(r"\s+", " ", text).strip()
  return text


# Check for dataset or create a quick benchmark dataset
DATA_PATH = "customer_support_tickets.csv"

if os.path.exists(DATA_PATH):
  print(f"Loading local dataset from {DATA_PATH}...")
  df = pd.read_csv(DATA_PATH)
  subject_col = "Ticket Subject" if "Ticket Subject" in df.columns else None
  desc_col = (
      "Ticket Description"
      if "Ticket Description" in df.columns
      else "Description"
  )
  target_col = "Ticket Type" if "Ticket Type" in df.columns else "Type"

  df = df.dropna(subset=[desc_col, target_col]).copy()
  if subject_col:
    df["full_text"] = df[subject_col].fillna("") + " " + df[desc_col].fillna("")
  else:
    df["full_text"] = df[desc_col].fillna("")
else:
  print(
      "No CSV found in root directory. Creating a baseline support ticket"
      " training dataset..."
  )
  sample_data = {
      "full_text": [
          (
              "Product setup issues I cannot connect the device to wifi or"
              " finish installation"
          ),
          (
              "Hardware malfunction screen flickers and power button is"
              " unresponsive"
          ),
          (
              "Billing question charge was applied twice on my credit card"
              " account"
          ),
          (
              "Refund request I want to cancel my subscription and get a full"
              " refund"
          ),
          (
              "Account locked forgot password and reset link is not arriving to"
              " email"
          ),
          (
              "Software bug app crashes immediately upon opening the dashboard"
              " view"
          ),
          (
              "Shipping status where is my package order tracking number has no"
              " updates"
          ),
          "Broken item received package was damaged during delivery transit",
      ]
      * 15,
      "Ticket Type": [
          "Technical Issue",
          "Hardware Issue",
          "Billing Inquiry",
          "Refund Request",
          "Account Access",
          "Bug Report",
          "Shipping & Delivery",
          "Damaged Product",
      ]
      * 15,
  }
  df = pd.DataFrame(sample_data)
  target_col = "Ticket Type"

print("Cleaning text data...")
df["clean_text"] = df["full_text"].apply(clean_text)

X = df["clean_text"]
y = df[target_col]

print(f"Training pipeline on {len(df)} samples...")
pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english", max_features=10000, ngram_range=(1, 2)
        ),
    ),
    (
        "classifier",
        SGDClassifier(
            loss="log_loss", penalty="l2", alpha=1e-4, random_state=42
        ),
    ),
])

pipeline.fit(X, y)

output_dir = os.path.join("backend", "models")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "ticket_classifier_pipeline.pkl")

with open(output_path, "wb") as f:
  pickle.dump(pipeline, f)

print(f"\nSuccess! Compatible model generated at: {output_path}")