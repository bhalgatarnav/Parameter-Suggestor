import pandas as pd
from sqlalchemy import create_engine

# ✅ EDIT THESE DETAILS:
user = "root"
password = "Material1100"
host = "localhost"
port = 3306
database = "material_advisor_db"  # <-- Replace with your actual database name

# ✅ Load your CSV
df = pd.read_csv("data/text_prompts.csv")

# Optional: clean column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# ✅ Create connection engine
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")

# ✅ Write to MySQL
df.to_sql("materials", con=engine, if_exists="replace", index=False)

print(" Data loaded into MySQL successfully.")
