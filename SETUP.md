# Parameter-Suggestor Setup Guide

## Security Notice
This application requires an OpenAI API key. **Never commit API keys to Git repositories!**

## Setup Instructions

### 1. Get an OpenAI API Key
1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key (starts with `sk-proj-...`)

### 2. Configure Environment Variables

**Option A: Using .env file (Recommended)**
```bash
# Copy the example file
cp env.example .env

# Edit .env and add your API key
OPENAI_API_KEY=sk-proj-your-actual-api-key-here
FLASK_ENV=development
FLASK_DEBUG=True
```

**Option B: Using shell environment**
```bash
# Set environment variable in your shell
export OPENAI_API_KEY="sk-proj-your-actual-api-key-here"
export FLASK_ENV="development"
export FLASK_DEBUG="True"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

## Security Best Practices
- ✅ Keep `.env` in `.gitignore` (already configured)
- ✅ Use environment variables for all secrets
- ✅ Never hardcode API keys in source code
- ✅ Rotate API keys regularly
- ❌ Never commit `.env` files to Git 