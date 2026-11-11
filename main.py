
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

while True:
    query = input("\nEnter your query: ")

    if query.lower().strip() == "q":
        print("Goodbye!")
        break

    response = client.chat.completions.create(
        model="nvidia/nemotron-3.5-super-vl:free",
        messages=[
            {
                "role": "user",
                "content": query
            }
        ]
    )

    print("AI response:", response.choices[0].message.content)
