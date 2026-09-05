import re
import string


def clean_text(text: str) -> str:
  text = str(text).lower()
  text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
  text = re.sub(r"\<.*?\>", "", text)
  text = text.translate(str.maketrans("", "", string.punctuation))
  text = re.sub(r"\s+", " ", text).strip()
  return text