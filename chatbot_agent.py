
from dotenv import load_dotenv
from openai import OpenAI
from tavily_agent_toolkit import search_and_format
import os
import json
import asyncio

# Load environment variables
load_dotenv()

# Initialize conversation memory
MAX_MESSAGES = 50
memory = []

# API keys
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")

if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY is missing from .env")

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key
)


# System instructions
system_message = {
    "role": "system",
    "content": (
        "You are a helpful AI chatbot named Chammak-Challo. "
        "Remember information shared by the user during this conversation. "
        "Use previous conversation context when answering questions. "
        "Use say_hello before answering any message. "
        "Use internet_search_tool whenever the user asks for "
        "latest information, current events, or information requiring "
        "an internet search. Answer using the search results."
    )
}


# Hello tool
def say_hello():
    """Before answering any message, use this tool."""
    print("HI,12345")
    return "HI,12345"


# Internet search tool
def internet_search_tool(query: str):

    print("\nSearching the internet...")

    try:
        result = asyncio.run(
            search_and_format(
                queries=[query],
                api_key=tavily_api_key,
                search_depth="basic",
                max_results=3
            )
        )

        return str(result)

    except Exception as e:
        return f"Internet search failed: {str(e)}"


# Tool definitions for OpenRouter
tools = [
    {
        "type": "function",
        "function": {
            "name": "say_hello",
            "description": "Print HI,12345 before answering any message.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "internet_search_tool",
            "description": (
                "Search the internet for latest information, "
                "current events, news, and factual information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# Execute tool calls
def execute_tool(tool_call):

    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    if function_name == "say_hello":
        return say_hello()

    elif function_name == "internet_search_tool":
        query = arguments.get("query", "")
        return internet_search_tool(query)

    return "Unknown tool requested."


# Get chatbot response
def get_resp(query):

    # Execute say_hello before processing every query
    say_hello()

    memory.append({
        "role": "user",
        "content": query
    })

    conversation = [system_message] + memory[-MAX_MESSAGES:]

    # First API call
    response = client.chat.completions.create(
        model="nvidia/nemotron-3.5-super-vl:free",
        messages=conversation,
        tools=tools,
        tool_choice="auto"
    )

    assistant_message = response.choices[0].message

    # Check whether the model requested a tool
    if assistant_message.tool_calls:

        memory.append(
            assistant_message.model_dump(exclude_none=True)
        )

        # Execute requested tools
        for tool_call in assistant_message.tool_calls:

            tool_result = execute_tool(tool_call)

            memory.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result)
            })

        # Send tool results back to the model
        conversation = [system_message] + memory[-MAX_MESSAGES:]

        response = client.chat.completions.create(
            model="nvidia/nemotron-3.5-super-vl:free",
            messages=conversation,
            tools=tools,
            tool_choice="auto"
        )

        answer = response.choices[0].message.content

    else:
        answer = assistant_message.content

    if not answer:
        answer = "I couldn't generate a response."

    # Store assistant response
    memory.append({
        "role": "assistant",
        "content": answer
    })

    return answer


# Main chatbot loop
while True:

    query = input("\nEnter your query: ")

    if query.lower().strip() == "q":
        print("Goodbye!")
        break

    if not query.strip():
        continue

    try:
        response = get_resp(query)
        print("\nChammak-Challo:", response)

    except Exception as e:
        print("\nError:", e)
