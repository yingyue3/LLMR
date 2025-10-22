from google import genai

client = genai.Client(api_key="AIzaSyCzBxBMaqLWIRLDk8yBbtTHddaseh-3e68")

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents="Hi, how are you?"
    )
    print(response.text)
finally:
    client.close()
