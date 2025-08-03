import pandas as pd
import torch
import open_clip
import numpy as np
import openai
import ast

# CONFIG
openai.api_key = "your-gpt-api-key-here"  # ⚠️ Secure in production

df = pd.read_csv("data/text_prompts.csv")
text_features = np.load("data/text_embeddings.npy")
text_features = torch.tensor(text_features, dtype=torch.float32)

# Load model
model_name = "ViT-B-32"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
tokenizer = open_clip.get_tokenizer(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

def refine_query_with_gpt(user_input):
    prompt = f"""
You're assisting a designer selecting a 3D rendering material. The user said: "{user_input}"

1. What specific use case or product is it for?
2. What performance or visual traits matter (e.g., matte, durable)?
3. Rewrite this as a complete and clear design intent.

Only return the final clarified prompt.
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("GPT clarification failed:", e)
        return user_input

def compute_boost(row, query, user_filter):
    boost = 0.0
    intent_match = ['high temp', 'heat', 'recycled', 'sustainable']
    if any(term in query.lower() for term in intent_match):
        if 'High Temp' in str(row.get('environment_suitability', '')):
            boost += 0.08
        if 'Sustainable' in str(row.get('intent_tags', '')):
            boost += 0.05
    if user_filter and user_filter.lower() in str(row.get('material_class', '')).lower():
        boost += 0.03
    return boost

def safe_eval_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, str):
        if ";" in value:
            return [v.strip() for v in value.split(";") if v.strip()]
        try:
            return ast.literal_eval(value)
        except Exception:
            return [v.strip() for v in value.strip("[]").replace("'", "").split(",") if v.strip()]
    return value

# --- SEARCH LOOP ---
print("\n Advisor ready! Type 'exit' to quit.\n")

while True:
    user_query = input(" Describe your material intent: ")
    if user_query.lower() == "exit":
        break

    # refined_query = refine_query_with_gpt(user_query)
    refined_query = user_query  # <- Replace this line to activate GPT enrichment
    user_filter = input("Filter by material class (e.g., Wood, Metal, Plastic) or press Enter to skip: ").strip()

    with torch.no_grad():
        tokenized = tokenizer([refined_query]).to(device)
        query_embedding = model.encode_text(tokenized)
        query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

    sims = (text_features @ query_embedding.T).squeeze()

    if user_filter:
        mask = df['material_class'].str.lower() == user_filter.lower()
        filtered_df = df[mask].copy()
        filtered_sims = sims[mask.to_numpy()]
    else:
        filtered_df = df.copy()
        filtered_sims = sims

    if len(filtered_df) == 0:
        print(" No materials found in the selected class.")
        continue

    filtered_df["score"] = [
        filtered_sims[i].item() + compute_boost(row, refined_query, user_filter)
        for i, (_, row) in enumerate(filtered_df.iterrows())
    ]

    top_matches = filtered_df.sort_values("score", ascending=False).head(5)

    print("\n Top Matches:\n")
    for i, (_, row) in enumerate(top_matches.iterrows(), 1):
        score = row["score"]
        print(f"{i}. {row['name']} (Match Confidence: {score:.3f})")
        print(f"   🔸 Description: {row['description']}")
        print(f"   🔸 Appearance: {safe_eval_list(row['appearance_tags'])}")
        print(f"   🔸 Intent Tags: {safe_eval_list(row['intent_tags'])}")
        print(f"   🔸 Use Cases: {safe_eval_list(row['suggested_use_cases'])}")
        print(f"   🔸 Class: {row['material_class']}")
        print("—" * 40)
