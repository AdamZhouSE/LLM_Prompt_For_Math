from openai import OpenAI

client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key="819d9e06-8139-4855-b953-6ff7d7786e94")


def get_full_response(prompt):
    completion = client.chat.completions.create(
        model="Meta-Llama-3.1-8B-Instruct",
        messages=prompt,
        stream=True
    )

    full_response = ""

    for chunk in completion:
        delta = chunk.choices[0].delta
        if hasattr(delta, 'content'):
            full_response += delta.content

    return full_response


