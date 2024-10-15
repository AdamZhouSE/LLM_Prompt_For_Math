import time

from openai import OpenAI
import requests
from config import MODEL_NAME, MODEL_BASE_URL, MODEL_API_KEY, LOCAL_MODEL_NAME, LOCAL_MODEL_BASE_URL
from pot_prompt import ProgramOfThoughts


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
                response = {'completion_tokens': chunk.usage.completion_tokens,
                            'prompt_tokens': chunk.usage.prompt_tokens,
                            'total_tokens': chunk.usage.total_tokens, 'time': chunk.usage.total_latency}
        response['answer'] = answer
        return response

    def get_response_from_local(self, prompt):
        data = {
            'model': LOCAL_MODEL_NAME,
            'messages': prompt,
            'stream': False,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        response = requests.post(LOCAL_MODEL_BASE_URL, json=data)
        if response.status_code == 200:
            return response.json()['message']['content']
        else:
            print(response.status_code)


if __name__ == '__main__':
    llm = LLM(0.4, 1)
    pot_prompt = ProgramOfThoughts()
    prompt = pot_prompt.n_shot_chats(8, "Olivia uploaded 72 pictures to Facebook.  She put the same number of the pics into 8 albums.  3 of the albums were selfies only and 2 of the albums were portraits.  How many portraits and selfies did she have?")
    llm.get_full_response(prompt)
    llm.get_response_from_local(prompt)
