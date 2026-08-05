from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Available models:")

try:
    for m in client.models.list():
        print(m.name)
except Exception as e:
    print(e)