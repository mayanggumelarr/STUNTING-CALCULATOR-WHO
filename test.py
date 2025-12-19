from http.client import responses

from google import genai

api ="AIzaSyBAI9gt3v9-39vpedmEnCaeRJyZ72DNhMU"

client = genai.Client(api_key=api)
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Why sky is blue?"
)

print(response.text)