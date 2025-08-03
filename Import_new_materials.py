import pandas as pd
import mysql.connector
import re
import json
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables with explicit path
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')

# Try multiple locations for .env file
possible_env_paths = [
    env_path,  # Same directory as script
    os.path.join(os.path.dirname(script_dir), '.env'),  # Parent directory
    os.path.join(os.getcwd(), '.env'),  # Current working directory
]

env_loaded = False
for env_file in possible_env_paths:
    if os.path.exists(env_file):
        print(f"Found .env file at: {env_file}")
        load_dotenv(env_file)
        env_loaded = True
        break

if not env_loaded:
    print("No .env file found in any of these locations:")
    for path in possible_env_paths:
        print(f"  - {path}")

# -------------- CONFIG --------------

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Material1100',
    'database': 'material_library'
}

CSV_FILE = 'C:\\Users\\anush\\OneDrive\\Desktop\\material-advisor\\data\\New_materials.csv'

# Initialize OpenAI client once at module level
def get_openai_client():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        # Try alternative environment variable names
        api_key = os.getenv('OPENAI_API_KEY_LOCAL') or os.getenv('OPENAI_KEY')
    
    if not api_key:
        print("Available environment variables:")
        for key in sorted(os.environ.keys()):
            if 'OPENAI' in key.upper() or 'API' in key.upper():
                print(f"  {key}: {os.environ[key][:10]}..." if os.environ[key] else f"  {key}: (empty)")
        raise ValueError("OPENAI_API_KEY environment variable is not set or empty")
    
    return OpenAI(api_key=api_key)

# ------------------------------------

# -------------- UTILS --------------
def parse_boolean(value):
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return int(value)
    value_str = str(value).strip().lower()
    return 1 if value_str in ['true', '1', 'yes'] else 0

def clean_multi_value_field(value):
    if pd.isna(value):
        return []
    value = str(value)
    value = re.sub(r"^\[|\]$", "", value)
    value = value.replace(";", ",")
    items = [v.strip(" '\"\n") for v in value.split(",") if v.strip()]
    return items
# ------------------------------------

# -------------- DB SETUP --------------
def add_columns_if_not_exist(cursor):
    alter_queries = [
        "ALTER TABLE materials ADD COLUMN keyshot_category TEXT",
        "ALTER TABLE materials ADD COLUMN sustainability_level ENUM('low', 'medium', 'high')",
        "ALTER TABLE materials ADD COLUMN manufacturability_notes TEXT"
    ]
    for q in alter_queries:
        try:
            cursor.execute(q)
        except mysql.connector.errors.ProgrammingError:
            pass  # Already exists
# ------------------------------------

# -------------- INSERT MATERIAL + TAGS --------------
def insert_material_and_tags(cursor, row):
    insert_material = """
    INSERT INTO materials (
        material_id, name, metalness, roughness, color_r, color_g, color_b,
        specular_r, specular_g, specular_b, material_class, description,
        visual_behavior, color_family, has_physical_equivalent,
        real_world_reference, preview_image_path, final_prompt
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (
        row['material_id'], row['name'], row['metalness'], row['roughness'],
        row['color_r'], row['color_g'], row['color_b'],
        row['specular_r'], row['specular_g'], row['specular_b'],
        row['material_class'], row['description'],
        row['visual_behavior'], row['color_family'],
        parse_boolean(row['has_physical_equivalent']),
        row['real_world_reference'], row['preview_image_path'], row['final_prompt']
    )
    cursor.execute(insert_material, values)
    material_id = cursor.lastrowid

    # Insert tags
    tag_tables = {
        'intent_tags': clean_multi_value_field(row.get('intent_tags', '')),
        'appearance_tags': clean_multi_value_field(row.get('appearance_tags', '')),
        'texture_map_types': clean_multi_value_field(row.get('texture_map_types', '')),
        'suggested_use_cases': clean_multi_value_field(row.get('suggested_use_cases', '')),
        'environment_suitability': clean_multi_value_field(row.get('environment_suitability', '')),
        'user_keywords': clean_multi_value_field(row.get('user_keywords', ''))
    }

    for table, tags in tag_tables.items():
        if not tags:
            continue
        field = {
            'intent_tags': 'tag',
            'appearance_tags': 'tag',
            'texture_map_types': 'map_type',
            'suggested_use_cases': 'use_case',
            'environment_suitability': 'env_condition',
            'user_keywords': 'keyword'
        }[table]

        for tag in tags:
            cursor.execute(f"INSERT INTO {table} (material_id, {field}) VALUES (%s, %s)", (material_id, tag))

    return material_id
# ------------------------------------

# -------------- GPT ENRICHMENT --------------
def gpt_enrich_material(material, client):
    prompt = f"""
You are an expert in 3D materials and rendering. Given:

Name: {material['name']}
Class: {material['material_class']}
Metalness: {material['metalness']}
Roughness: {material['roughness']}
Description: {material['description']}

Suggest:
- KeyShot category path
- Sustainability level (low / medium / high)
- Manufacturability notes

Respond in JSON:
{{
  "keyshot_category": "...",
  "sustainability_level": "...",
  "manufacturability_notes": "..."
}}
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
    except json.JSONDecodeError:
        print(f"GPT JSON parsing failed for: {material['name']}")
        return {
            "keyshot_category": "Unknown",
            "sustainability_level": "medium",
            "manufacturability_notes": "No notes available"
        }
    except Exception as e:
        print(f"GPT API error for {material['name']}: {str(e)}")
        return {
            "keyshot_category": "Unknown",
            "sustainability_level": "medium",
            "manufacturability_notes": "No notes available"
        }
# ------------------------------------

# -------------- UPDATE NEW FIELDS --------------
def update_material_with_gpt(cursor, material_id, gpt_data):
    update = """
    UPDATE materials SET
        keyshot_category = %s,
        sustainability_level = %s,
        manufacturability_notes = %s
    WHERE id = %s
    """
    cursor.execute(update, (
        gpt_data['keyshot_category'],
        gpt_data['sustainability_level'],
        gpt_data['manufacturability_notes'],
        material_id
    ))
# ------------------------------------

# -------------- MAIN --------------
def main():
    # Debug: Check if environment variable is loaded
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"API Key loaded: {'Yes' if api_key else 'No'}")
    if api_key:
        print(f"API Key starts with: {api_key[:10]}...")
    
    # Initialize OpenAI client with error handling
    try:
        client = get_openai_client()
        print("OpenAI client initialized successfully")
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease:")
        print("1. Create a .env file in the same directory as this script")
        print("2. Add this line to the .env file:")
        print("   OPENAI_API_KEY=sk-your-actual-api-key-here")
        print("3. Replace 'sk-your-actual-api-key-here' with your real OpenAI API key")
        return
    
    df = pd.read_csv(CSV_FILE)
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print("Adding new columns if needed...")
    add_columns_if_not_exist(cursor)

    print(f"Inserting {len(df)} materials with GPT enrichment...")
    for i, row in df.iterrows():
        material_data = row.to_dict()
        material_id = insert_material_and_tags(cursor, row)
        gpt_data = gpt_enrich_material(material_data, client)
        update_material_with_gpt(cursor, material_id, gpt_data)

        if (i + 1) % 10 == 0:
            conn.commit()
            print(f"Inserted {i + 1} of {len(df)}...")

    conn.commit()
    cursor.close()
    conn.close()
    print("All materials added and enriched!")

# ------------------------------------

if __name__ == '__main__':
    main()
