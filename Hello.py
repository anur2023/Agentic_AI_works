
import os
from dotenv import load_dotenv
from openai import OpenAI, APIError

# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Check required configuration
if not API_KEY:
    raise ValueError("OPENAI_API_KEY is missing in your .env file")

if not MODEL_NAME:
    raise ValueError("MODEL_NAME is missing in your .env file")


# ==========================================
# 2. INITIALIZE GEMINI CLIENT
# ==========================================

client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


# ==========================================
# 3. MAKE API REQUEST
# ==========================================

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a concise technical mentor."
            },
            {
                "role": "user",
                "content": "Explain what an API is in two sentences."
            }
        ],
        max_tokens=150
    )

    # ==========================================
    # 4. EXTRACT AI RESPONSE
    # ==========================================

    ai_response = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason

    print("\n========== AI RESPONSE ==========")
    print(ai_response)

    # ==========================================
    # 5. DISPLAY FINISH REASON
    # ==========================================

    print("\n========== FINISH REASON ==========")
    print(f"Finish reason: {finish_reason}")

    # ==========================================
    # 6. TRACK TOKEN USAGE
    # ==========================================

    usage = response.usage

    print("\n========== TOKEN USAGE ==========")

    if usage:
        print(f"Input tokens:  {usage.prompt_tokens}")
        print(f"Output tokens: {usage.completion_tokens}")
        print(f"Total tokens:  {usage.total_tokens}")
    else:
        print("Token usage information is unavailable.")

    print("\n========== EXECUTION COMPLETED ==========")

except APIError as e:
    print("\nAPI Error occurred:")
    print(e)

except Exception as e:
    print("\nUnexpected error occurred:")
    print(e)
