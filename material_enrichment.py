import mysql.connector
import json
from openai import OpenAI, AuthenticationError
import os
from dotenv import load_dotenv
import time
import re
import sys

# Load environment variables from specific path
env_path = r"C:\Users\anush\OneDrive\Desktop\material-advisor\scripts\.env"
load_dotenv(env_path)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Material1100',
    'database': 'material_library'
}

# Initialize OpenAI client with validation
def get_openai_client():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env file")
        print(f"Checked path: {env_path}")
        print("Please verify the .env file contains: OPENAI_API_KEY=your_key_here")
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    # Test API connection
    try:
        client.models.list(timeout=5)
        print("OpenAI API connection successful")
        return client
    except AuthenticationError as e:
        print(f"FATAL: OpenAI authentication failed: {e}")
        print("Please verify your API key at: https://platform.openai.com/account/api-keys")
        raise SystemExit("Authentication failed")
    except Exception as e:
        print(f"WARNING: OpenAI connection test failed: {e}")
        return client

# Enhanced GPT enrichment function with strict quality control
def gpt_enrich_material(material, client, retries=2):
    # Handle list-type description
    if isinstance(material['description'], list):
        description = ", ".join(material['description'])
    else:
        description = str(material['description'])
    
    prompt = f"""
You are a senior materials scientist with 20+ years of experience. Analyze this material for 3D rendering and manufacturing:

**MATERIAL ANALYSIS REPORT REQUEST**
Material Name: {material['name']}
Material Class: {material['material_class']}
Metalness: {material['metalness']}
Roughness: {material['roughness']}
Description: {description}

**REQUIRED OUTPUT (STRICT JSON FORMAT ONLY):**
{{
  "keyshot_category": "[Base]-[Color/Appearance]-[Specific]-[Finish]",
  "sustainability_level": "low/medium/high",
  "manufacturability_notes": "Concise technical bullet points"
}}

**INSTRUCTIONS:**
1. KEYSOT CATEGORY:
   - Format: [Base Material] - [Color/Appearance] - [Specific Name] - [Texture/Finish]
   - Base types: Metal, Plastic, Glass, Fabric, Wood, Stone, Leather, Paint, Liquid, Misc, Emissive
   - MUST include at least 3 components
   - Example: "Metal - Gold - 24k Gold - Polished" for gold material

2. SUSTAINABILITY LEVEL:
   - MUST be based on actual material properties
   - Use industry standards: 
        'low' = Non-recyclable, high energy footprint
        'medium' = Partially recyclable
        'high' = Easily recyclable, low environmental impact

3. MANUFACTURABILITY NOTES:
   - MUST include SPECIFIC technical data:
        • Processing temp range: [X-Y]°C (actual values)
        • Corrosion resistance: [Poor/Fair/Good/Excellent] against [specific agents]
        • Tensile strength: [X] MPa (actual value or range)
        • Reflectivity: [description] with percentage if possible
        • Special handling: [specific requirements]
   - NO generic phrases like "consult datasheet" or "depends on formulation"
   - Use bullet point format

**RESEARCH REQUIREMENTS:**
- Consult material databases: MatWeb, CES EduPack, Granta Design
- Reference real-world manufacturer specifications
- For fictional materials, use closest real-world analog
- ABSOLUTELY NO GENERIC RESPONSES ALLOWED

**EXAMPLE OF ACCEPTABLE RESPONSE FOR "24k Gold":**
{{
  "keyshot_category": "Metal - Gold - 24k Gold - Polished",
  "sustainability_level": "low",
  "manufacturability_notes": "• Processing temperature: 1064°C (melting point)\\n• Corrosion resistance: Excellent against all common agents\\n• Tensile strength: 120 MPa\\n• Reflectivity: 95%+ across visible spectrum\\n• Special handling: Requires electroplating for thin coatings"
}}
"""

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a materials engineering expert. Provide SPECIFIC, NON-GENERIC technical data based on real material science principles."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Strict validation against generic responses
            if is_generic_response(data):
                raise ValueError("Generic response detected - retrying")
                
            return data
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Attempt {attempt+1} failed for {material['name']}: {str(e)}")
            if attempt < retries:
                time.sleep(3)
                continue
            else:
                print(f"Using enhanced fallback for {material['name']}")
                return generate_technical_fallback(material)
        
        except Exception as e:
            print(f"GPT error for {material['name']}: {str(e)}")
            if "401" in str(e):
                raise SystemExit("Fatal API key error - stopping script")
                
            if attempt < retries:
                time.sleep(3)
                continue
            else:
                return generate_technical_fallback(material)

# Strict generic response detection
def is_generic_response(data):
    generic_keywords = [
        "unknown", "various", "general", "standard", "depends", "consult", 
        "varies", "manufacturer", "datasheet", "verify", "depends", "typical",
        "depends on", "material-dependent", "formulation-dependent"
    ]
    
    # Check category
    category = data.get("keyshot_category", "").lower()
    if any(kw in category for kw in generic_keywords):
        return True
        
    # Check sustainability - medium is often generic
    if data.get("sustainability_level", "") == "medium":
        return True
        
    # Check notes for generic phrases
    notes = data.get("manufacturability_notes", "").lower()
    if any(kw in notes for kw in generic_keywords):
        return True
        
    # Check for missing technical specifics
    technical_terms = ["°c", "mpa", "%", "excellent", "good", "fair", "poor"]
    if not any(term in notes for term in technical_terms):
        return True
        
    return False

# Technical fallback with material-specific data
def generate_technical_fallback(material):
    name = material['name'].lower()
    
    # Material-specific technical fallbacks
    if "blood" in name:
        return {
            "keyshot_category": "Liquid - Red - Blood - Viscous",
            "sustainability_level": "high",
            "manufacturability_notes": (
                "• Processing temperature: 36-38°C (biological range)\n"
                "• Corrosion resistance: Poor (oxidizes rapidly)\n"
                "• Viscosity: 3-4 cP at 37°C\n"
                "• Reflectivity: 2-3% (opaque fluid)\n"
                "• Special handling: Sterile environment required"
            )
        }
    elif "brass" in name:
        return {
            "keyshot_category": "Metal - Gold - Brass - Polished",
            "sustainability_level": "medium",
            "manufacturability_notes": (
                "• Processing temperature: 900-940°C (casting)\n"
                "• Corrosion resistance: Good (tarnishes slowly)\n"
                "• Tensile strength: 340-500 MPa\n"
                "• Reflectivity: 65-70% (polished surface)\n"
                "• Special handling: Avoid ammonia exposure"
            )
        }
    elif "gold" in name or "24k" in name:
        return {
            "keyshot_category": "Metal - Gold - 24k Gold - Polished",
            "sustainability_level": "low",
            "manufacturability_notes": (
                "• Processing temperature: 1064°C (melting point)\n"
                "• Corrosion resistance: Excellent (noble metal)\n"
                "• Tensile strength: 120 MPa\n"
                "• Reflectivity: 95%+ (visible light)\n"
                "• Special handling: Electroplating for thin coatings"
            )
        }
    elif "steel" in name:
        return {
            "keyshot_category": "Metal - Silver - Steel - Brushed",
            "sustainability_level": "medium",
            "manufacturability_notes": (
                "• Processing temperature: 1370-1530°C (casting)\n"
                "• Corrosion resistance: Fair (requires coating)\n"
                "• Tensile strength: 400-550 MPa\n"
                "• Reflectivity: 50-60% (polished)\n"
                "• Special handling: Annealing recommended"
            )
        }
    elif "plastic" in name:
        return {
            "keyshot_category": "Plastic - Various - Engineering Polymer - Smooth",
            "sustainability_level": "low",
            "manufacturability_notes": (
                "• Processing temperature: 200-300°C\n"
                "• Corrosion resistance: Excellent (chemical resistant)\n"
                "• Tensile strength: 40-80 MPa\n"
                "• Reflectivity: 5-15% (matte surface)\n"
                "• Special handling: Pre-drying required"
            )
        }
    elif "wood" in name:
        return {
            "keyshot_category": "Wood - Brown - Hardwood - Grained",
            "sustainability_level": "high",
            "manufacturability_notes": (
                "• Processing temperature: Ambient (mechanical processing)\n"
                "• Corrosion resistance: Poor (susceptible to moisture)\n"
                "• Tensile strength: 70-140 MPa\n"
                "• Reflectivity: Matte finish (diffuse reflection)\n"
                "• Special handling: Seal with varnish"
            )
        }
    else:  # Default technical fallback
        return {
            "keyshot_category": f"{material['material_class']} - Engineered - {material['name']} - Technical",
            "sustainability_level": "medium",
            "manufacturability_notes": (
                "• Processing temperature: Refer to technical specifications\n"
                "• Corrosion resistance: Material-specific performance\n"
                "• Tensile strength: Consult material datasheet\n"
                "• Reflectivity: Depends on surface treatment\n"
                "• Special handling: Follow industry protocols"
            )
        }

# Add KeyShot path components to tags
def add_keyshot_tags(cursor, material_id, keyshot_category):
    path_parts = [p.strip() for p in keyshot_category.split("-") if p.strip()]
    for part in path_parts:
        cursor.execute(
            "INSERT IGNORE INTO user_keywords (material_id, keyword) VALUES (%s, %s)",
            (material_id, part.lower())
        )

# Check if column exists in table
def column_exists(cursor, table_name, column_name):
    try:
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = DATABASE()
            AND table_name = '{table_name}'
            AND column_name = '{column_name}'
        """)
        return cursor.fetchone()[0] > 0
    except:
        return False

# Add column if it doesn't exist
def safe_add_column(cursor, table_name, column_name, column_type):
    if not column_exists(cursor, table_name, column_name):
        try:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            print(f"Added column: {table_name}.{column_name}")
            return True
        except Exception as e:
            print(f"Error adding column: {e}")
            return False
    return False

# Main enrichment process
def enrich_materials():
    # Initialize clients
    try:
        client = get_openai_client()
        print("OpenAI client initialized successfully")
    except (ValueError, SystemExit) as e:
        print(f"Fatal Error: {e}")
        return
    
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Ensure columns exist with safe method
    safe_add_column(cursor, "materials", "keyshot_category", "TEXT")
    safe_add_column(cursor, "materials", "sustainability_level", "ENUM('low', 'medium', 'high')")
    safe_add_column(cursor, "materials", "manufacturability_notes", "TEXT")
    conn.commit()
    
    # Get ALL materials for reprocessing
    cursor.execute("""
        SELECT id, name, material_class, metalness, roughness, description
        FROM materials
    """)
    materials = cursor.fetchall()
    
    print(f"Processing {len(materials)} materials for enrichment")
    
    # Process each material
    for count, (mat_id, name, mat_class, metalness, roughness, description) in enumerate(materials, 1):
        print(f"\nProcessing material {count}/{len(materials)}: {name}")
        
        material_data = {
            'name': name,
            'material_class': mat_class,
            'metalness': metalness,
            'roughness': roughness,
            'description': description
        }
        
        try:
            # Get GPT enrichment with retries
            gpt_data = gpt_enrich_material(material_data, client)
            
            # Print results for verification
            print(f"Keyshot Category: {gpt_data['keyshot_category']}")
            print(f"Sustainability: {gpt_data['sustainability_level']}")
            print(f"Manufacturability Notes: {gpt_data['manufacturability_notes'][:100]}...")
            
            # Update database
            cursor.execute("""
                UPDATE materials SET
                    keyshot_category = %s,
                    sustainability_level = %s,
                    manufacturability_notes = %s
                WHERE id = %s
            """, (
                gpt_data['keyshot_category'],
                gpt_data['sustainability_level'],
                gpt_data['manufacturability_notes'],
                mat_id
            ))
            
            # Add KeyShot tags
            add_keyshot_tags(cursor, mat_id, gpt_data['keyshot_category'])
            
            # Commit every 2 materials
            if count % 2 == 0:
                conn.commit()
                print(f"Committed {count} updates")
                
            # Avoid rate limiting
            time.sleep(2.5)
            
        except SystemExit as e:
            print(f"Critical error: {e}")
            conn.close()
            sys.exit(1)
        except Exception as e:
            print(f"Error processing {name}: {str(e)}")
    
    # Final commit
    conn.commit()
    print("\nAll materials enriched successfully")
    
    cursor.close()
    conn.close()

if __name__ == '__main__':
    enrich_materials()