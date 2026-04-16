from openai import OpenAI

def chat(model, prompt):
    client = OpenAI(
        api_key="",
        base_url=""
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
