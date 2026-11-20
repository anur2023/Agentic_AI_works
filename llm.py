
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Create one reusable client
_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Store usage of the latest API call
LAST_USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0
}


def ask(
    prompt: str,
    system: str = "You are a concise technical mentor.",
    max_tokens: int = 300
) -> str:

    """Send a prompt to the model and return the reply."""

    response = _client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )

    # Save token usage
    LAST_USAGE["input_tokens"] = response.usage.prompt_tokens
    LAST_USAGE["output_tokens"] = response.usage.completion_tokens
    LAST_USAGE["total_tokens"] = response.usage.total_tokens

    # Return only generated text
    return response.choices[0].message.content
