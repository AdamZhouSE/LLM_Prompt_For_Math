import time

from evaluation import Evaluation

system_prompt = """
Your task is to solve a series of math word problems by providing the final answer. 
Use the format #### [value] to highlight your answer. 
For example, if the answer is 560, you should write #### 560."""


class ProgressiveHint(Evaluation):
    def __init__(self, llm, prompt_method, record_path):
        super().__init__(llm, prompt_method, record_path)
        self.hint = None
        self.max_hint = 5

    def question_prompt_with_hint(self, question):
        prompt = f'Question: {question}'
        if self.hint:
            prompt = f'Question: {question}(Hint: The answer is near to {self.hint})'
        return prompt

    def answer_prompt(self, answer):
        return f"Answer:\nLet's think step by step.\n{answer}"

    def n_shot_chats(self, n: int, question: str):
        chats = [{"role": "system", "content": system_prompt}]

        # for q, a in self.n_shot_list[:n]:
        #     chats.append(
        #         {"role": "user", "content": self.question_prompt(q)})
        #     chats.append(
        #         {"role": "assistant", "content": self.answer_prompt(a)})

        chats.append({"role": "user", "content": self.question_prompt_with_hint(question)})
        return chats

    def progressive_hint(self, data):
        # generate prompt
        prompt = self.generate_prompt(data['question'])
        total_completion_tokens = 0
        total_time = 0.0
        # chat with llm multiple times until the answer is the same
        last_llm_answer = None
        for i in range(self.max_hint):
            full_response = self.llm.get_full_response(prompt)
            total_completion_tokens += full_response['completion_tokens']
            total_time += full_response['time']
            llm_answer = self.convert_answer(full_response['answer'])
            print(llm_answer)
            if last_llm_answer == llm_answer:
                break
            last_llm_answer = llm_answer
        time.sleep(3)
        return last_llm_answer, total_completion_tokens, total_time

    def evaluation(self, data):
        """
        compare the result from llm with the correct answer
        record the evaluation result in a jsonl file
        """
        # get llm result
        llm_answer, total_completion_tokens, total_time = self.progressive_hint(data)
        # convert the answer into numerical form
        answer = self.convert_answer(data['answer'])
        print('question', data['question'])
        print('answer vs llm_answer', answer, llm_answer)
        self.record_evaluation(data['question'], answer, llm_answer, [data['answer']], total_completion_tokens,
                               total_time)
        return llm_answer == answer
