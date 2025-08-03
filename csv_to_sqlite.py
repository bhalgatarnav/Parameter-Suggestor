import pandas as pd
import sqlite3

# Load the enriched CSV
df = pd.read_csv("data/text_prompts.csv", encoding="ISO-8859-1")


# Connect to SQLite (creates file if doesn't exist)
conn = sqlite3.connect("data/materials.db")

# Write the DataFrame to a table named 'materials'
df.to_sql("materials", conn, if_exists="replace", index=False)

# Close the connection
conn.close()

print(" CSV successfully converted to SQLite (materials.db)")
