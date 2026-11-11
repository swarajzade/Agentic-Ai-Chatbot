
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

# Initialize conversation memory
MAX_MESSAGES = 50
memory = []

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# System instructions
system_message = {
    "role": "system",
    "content": (
        "You are a helpful AI chatbot named Chammak-Challo. "
        "Remember the information shared by the user during this conversation. "
        "Answer questions using the previous conversation context."
    )
}


def get_resp(query):

    # Add user query to memory
    memory.append({
        "role": "user",
        "content": query
    })

    # Keep only the latest 50 messages
    conversation = [system_message] + memory[-MAX_MESSAGES:]

    response = client.chat.completions.create(
        model="nvidia/nemotron-3.5-super-vl:free",
        messages=conversation
    )

    answer = response.choices[0].message.content

    # Store chatbot response in memory
    memory.append({
        "role": "assistant",
        "content": answer
    })

    return answer


while True:
    query = input("\nEnter your query: ")

    if query.lower().strip() == "q":
        print("Goodbye!")
        break

    try:
        response = get_resp(query)
        print("Chammak-Challo:", response)

    except Exception as e:
        print("Error:", e)
