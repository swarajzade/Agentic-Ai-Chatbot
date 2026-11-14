
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
import os
import json

# Load environment variables
load_dotenv()


# -----------------------------------
# 1. Initialize OpenRouter
# -----------------------------------

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key
)


# -----------------------------------
# 2. Structured Output
# -----------------------------------

class SupportResponse(BaseModel):

    stage: str = Field(
        description="Stage like greeting, answered, refunded or rejected"
    )

    successful: bool

    message: str


# -----------------------------------
# 3. Conversation Memory
# -----------------------------------

MAX_MESSAGES = 50

memory = []

session_id = "customer_session_001"


# -----------------------------------
# 4. Customer Support Agent
# -----------------------------------

SYSTEM_MESSAGE = {
    "role": "system",
    "content": """
    You are Chammak-Challo, a professional customer support agent.

    Your main task is to provide customer support.

    Instructions:

    1. Greet customers politely.
    2. Understand their problems carefully.
    3. Help with order status, refunds, returns and complaints.
    4. Ask for order ID whenever necessary.
    5. Be friendly, polite and professional.
    6. Remember previous conversation messages.
    7. Never invent order details.
    8. Never claim that a refund was processed without
       confirmation from a real payment system.
    9. Provide clear and helpful answers.

    Stage guidelines:

    - greeting: When the customer greets you.
    - answered: When answering general questions.
    - refunded: Only when an actual refund is verified.
    - rejected: When a request cannot be fulfilled.

    IMPORTANT:
    Always return your response in valid JSON format.

    Use exactly these three fields:

    {
        "stage": "greeting",
        "successful": true,
        "message": "Your response here"
    }

    Do not include Markdown or code fences.
    """
}


# -----------------------------------
# 5. Customer Support Function
# -----------------------------------

def get_support_response(query):

    # Add user query to memory
    memory.append({
        "role": "user",
        "content": query
    })

    # Prepare conversation
    conversation = [
        SYSTEM_MESSAGE
    ] + memory[-MAX_MESSAGES:]

    # Call OpenRouter model
    response = client.chat.completions.create(

        model="nvidia/nemotron-3.5-super-vl:free",

        messages=conversation,

        temperature=0.3
    )

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError("Model returned an empty response.")

    # Convert JSON response into Pydantic model
    try:

        response_data = json.loads(answer)

        support_response = SupportResponse.model_validate(
            response_data
        )

    except (json.JSONDecodeError, ValidationError) as e:

        raise ValueError(
            f"Invalid structured response from model: {e}"
        )

    # Store assistant response in memory
    memory.append({
        "role": "assistant",
        "content": support_response.model_dump_json()
    })

    return support_response


# -----------------------------------
# 6. Main Function
# -----------------------------------

def main():

    print("\nCustomer Support Agent Started!")
    print("Type 'q' to exit.\n")

    while True:

        query = input("Customer: ").strip()

        if query.lower() == "q":
            print("Goodbye!")
            break

        if not query:
            continue

        try:

            result = get_support_response(query)

            # Print dictionary output
            print(result.model_dump())

        except Exception as e:

            print("Error:", str(e))


# -----------------------------------
# 7. Entry Point
# -----------------------------------

if __name__ == "__main__":
    main()
