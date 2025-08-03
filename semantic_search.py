import pandas as pd
import numpy as np
import torch
import open_clip
import faiss

# Load text prompts and embeddings
df = pd.read_csv("text_prompts.csv")
embeddings = np.load("text_embeddings.npy").astype('float32')

# Set up FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

def get_query_embedding(text_query):
    tokenized = tokenizer([text_query]).to(device)
    with torch.no_grad():
        query_embedding = model.encode_text(tokenized)
    return query_embedding.cpu().numpy().astype('float32')

# Ask user for input
user_query = input("Enter a material description (e.g., 'eco-friendly matte plastic'): ")
query_vec = get_query_embedding(user_query)

# Perform semantic search
k = 5
D, I = index.search(query_vec, k)

# Show top matches
print("\nTop material suggestions for:", user_query)
for rank, idx in enumerate(I[0]):
    print(f"{rank + 1}. {df.iloc[idx]['text_prompt']}  (Score: {D[0][rank]:.4f})")
