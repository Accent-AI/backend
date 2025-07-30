import os
import requests
from dotenv import load_dotenv
load_dotenv()


def call_groq_api(prompt: str) -> str:
    """Call Groq API for text generation"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("⚠️ GROQ_API_KEY not found in environment variables")
            return None
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150,
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            message = result["choices"][0]["message"]["content"].strip()
            print("✅ Generated message using Groq API")
            return message
        else:
            print(f"❌ Groq API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Groq API call failed: {e}")
        return None 