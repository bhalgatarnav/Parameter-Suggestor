from database import SessionLocal
from models import Material, AppearanceTag, IntentTag, SuggestedUseCase, EnvironmentSuitability, UserKeyword
from sentence_transformers import SentenceTransformer
import json

# Load embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_material_tags(material_id, db):
    """Combine all tags for a material into one string."""
    tags = []
    
    # Appearance tags
    appearance = db.query(AppearanceTag).filter_by(material_id=material_id).all()
    tags.extend([a.tag for a in appearance])
    
    # Intent tags
    intent = db.query(IntentTag).filter_by(material_id=material_id).all()
    tags.extend([i.tag for i in intent])
    
    # Use cases
    use_cases = db.query(SuggestedUseCase).filter_by(material_id=material_id).all()
    tags.extend([u.use_case for u in use_cases])
    
    # Environment suitability
    environments = db.query(EnvironmentSuitability).filter_by(material_id=material_id).all()
    tags.extend([e.env_condition for e in environments])
    
    # User keywords
    keywords = db.query(UserKeyword).filter_by(material_id=material_id).all()
    tags.extend([k.keyword for k in keywords])
    
    return " ".join(tags)

def generate_all_embeddings():
    db = SessionLocal()
    
    try:
        materials = db.query(Material).all()
        print(f"Found {len(materials)} materials in DB.")

        for material in materials:
            combined_text = f"{material.name or ''} {material.description or ''} {get_material_tags(material.id, db)}"
            combined_text = combined_text.strip()

            if not combined_text:
                print(f"Skipping empty material: {material.id}")
                continue

            embedding = model.encode(combined_text)
            material.embedding = json.dumps(embedding.tolist())

        db.commit()
        print("Embeddings generated and stored successfully.")
    
    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
    
    finally:
        db.close()

if __name__ == "__main__":
    generate_all_embeddings()
