import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file
api_key = os.getenv('OPENAI_API_KEY')

if api_key:
    print(f"API key found: {api_key[:10]}...{api_key[-3:]}")
    print(f"Key length: {len(api_key)} characters")
else:
    print("API key NOT found in .env file")