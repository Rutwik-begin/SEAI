import os
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def generate_disease_info(prediction_label):

    prompt = f"""
    You are an agricultural plant disease expert.

    Explain the plant disease:
    "{prediction_label}"

    Provide:

    1. Crop Name
    2. Disease Name
    3. Overview
    4. Causes
    5. Symptoms
    6. Treatment
    7. Prevention

    Keep the response concise, practical,
    and farmer-friendly.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text