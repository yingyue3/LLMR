from google import genai

client = genai.Client()

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents="Hi, how are you?"
    )
    print(response.text)
finally:
    client.close()
