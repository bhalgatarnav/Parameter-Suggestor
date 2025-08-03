import openai
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import time

# --- SETUP ---
openai.api_key = "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"
DB_URI = "mysql+mysqlconnector://username:password@localhost/materials_db"

# --- CONNECT TO DB ---
engine = create_engine(DB_URI)
df = pd.read_sql("SELECT * FROM materials", engine)

# --- COMBINE TEXT FIELDS FOR EMBEDDING ---
def build_prompt(row):
    fields = [
        row["name"],
        row.get("description", ""),
        ", ".join(eval(row.get("appearance_tags", "[]"))),
        ", ".join(eval(row.get("intent_tags", "[]"))),
        row.get("visual_behavior", ""),
        row.get("real_world_reference", "")
    ]
    return " | ".join(str(f) for f in fields if f)

df["embedding_text"] = df.apply(build_prompt, axis=1)

# --- EMBEDDING GENERATION ---
def get_embedding(text, model="text-embedding-ada-002", retries=3):
    for _ in range(retries):
        try:
            res = openai.embeddings.create(input=[text], model=model)
            return res.data[0].embedding
        except Exception as e:
            print(f"Retrying due to error: {e}")
            time.sleep(1)
    return None

print("Generating embeddings...")
df["embedding"] = df["embedding_text"].apply(get_embedding)

# --- SAVE BACK TO SQL ---
# You can store as a separate table, or update your main table.
embeddings_df = df[["name", "embedding"]].copy()
embeddings_df.to_sql("material_embeddings", engine, if_exists="replace", index=False)

print(" Embeddings saved to SQL table 'material_embeddings'.")
