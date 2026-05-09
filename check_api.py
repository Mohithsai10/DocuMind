import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    print("❌ No API key found in .env")
    exit(1)

print(f"✅ Key found: {api_key[:7]}...{api_key[-4:]}")

from openai import OpenAI
client = OpenAI(api_key=api_key)

resp = client.embeddings.create(model="text-embedding-ada-002", input="DocuMind test")
print(f"✅ Embeddings API working — vector size: {len(resp.data[0].embedding)}")

chat = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Say 'DocuMind ready' only."}],
    max_tokens=10
)
print(f"✅ Chat API: {chat.choices[0].message.content}")
print("\n🚀 All systems go. Ready to build.")
