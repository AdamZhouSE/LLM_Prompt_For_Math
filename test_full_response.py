from openai import OpenAI
from baseline import zero_shot_prompt, few_shot_prompt

client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key="819d9e06-8139-4855-b953-6ff7d7786e94")


completion = client.chat.completions.create(
    model="Meta-Llama-3.1-8B-Instruct",
    messages=zero_shot_prompt,
    stream=True
)

full_response = ""
print(completion)
for chunk in completion:
    delta = chunk.choices[0].delta
    if hasattr(delta, 'content'):
        full_response += delta.content

print(full_response)
print(zero_shot_prompt)

# If you are using the original script in the .pdf, you can set streaming to false, then print(completion.choices[0].message.content)