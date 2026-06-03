from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("GOOGLE_API_KEY")
print(f"API key loaded: {key[:8]}..." if key else "ERROR: API key not found!")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
response = llm.invoke("Say exactly the words: SemaGC is ready")
print(response.content)