import time

from openai import OpenAI
MODEL_NAME = "Meta-Llama-3.1-8B-Instruct"
MODEL_BASE_URL = "https://api.sambanova.ai/v1"
MODEL_API_KEY = "819d9e06-8139-4855-b953-6ff7d7786e94"
# MODEL_BASE_URL = "http://localhost:11434/v1/"
# MODEL_API_KEY = "ollama"
# MODEL_NAME = "llama3.1:8b-instruct-q6_K"


class LLM:
    def __init__(self, temperature=0.0, top_p=1):
        self.client = OpenAI(base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY)
        self.temperature = temperature
        self.top_p = top_p

    def get_full_response(self, prompt):
        """
        Get full response from the model, including answer and cost of the query
        """
        completion = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=prompt,
            stream=True,
            stream_options={"include_usage": True},
            temperature=self.temperature,
            top_p=self.top_p,
        )

        time.sleep(0.5)

        answer = ""
        response = {}
        for chunk in completion:
            if len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content'):
                    answer += delta.content
            else:
                # end of stream, return information about the query
                response = {'completion_tokens': chunk.usage.completion_tokens, 'time': chunk.usage.total_latency}
        response['answer'] = answer
        return response


if __name__ == '__main__':
    llm = LLM()
    chats = [{"role": "system", "content": "Hello, I am LLM. I am here to help you with your math problems."},
             {"role": "user", "content": "What is 2+2?"}]
    print(llm.get_full_response(chats))
