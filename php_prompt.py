from evaluation import Evaluation

system_prompt = '''
Your task is to solve a series of math word problems by providing the final answer. 
Use the format #### [value] to highlight your answer. 
For example, if the answer is 560, you should write #### 560.
'''


def question_prompt_with_hint(question, hint):
    prompt = f'Question: {question}'
    if len(hint) > 0:
        prompt = f'Question: {question}(Hint: The answer is near to {",".join(hint)})'
    return prompt


class ProgressiveHint(Evaluation):
    """
    PHP: Progressive Hint Prompt (https://github.com/chuanyang-Zheng/Progressive-Hint)
    Idea: Engage in multiple rounds of interaction with the model,
            using the previous round's answer as a prompt for the model,
            and end the process when the model returns the same result twice.
    """
    def __init__(self, llm, record_path, n_shots_flag=False):
        super().__init__(llm, record_path)
        self.max_hint = 10
        self.n_shots_prompt = ''
        # decide whether to use zero-shot or few-shot, default zero-shot
        if n_shots_flag:
            with open('prompt/complex_php_gsm8k.txt', 'r') as f:
                self.n_shots_prompt = f.read()

    def n_shot_chats(self, question: str, hint: list):
        chats = [{"role": "system", "content": system_prompt + self.n_shots_prompt},
                 {"role": "user", "content": question_prompt_with_hint(question, hint)}]

        return chats

    def progressive_hint(self, data):
        total_completion_tokens = 0
        total_time = 0.0
        generated = []
        # chat with llm multiple times until the answer is the same
        last_llm_answer = None
        hint = []
        for i in range(self.max_hint):
            # generate prompt
            prompt = self.generate_prompt_with_hint(data['question'], hint)
            full_response = self.llm.get_full_response(prompt)
            total_completion_tokens += full_response['completion_tokens']
            total_time += full_response['time']
            generated.append(full_response['answer'])
            llm_answer = self.convert_answer(full_response['answer'])
            print(llm_answer)
            # convert answer into numerical form successfully
            if llm_answer:
                if last_llm_answer == llm_answer:
                    break
                last_llm_answer = llm_answer
                # add new hint to question
                hint.append(last_llm_answer)
            else:
                # llm may return "I can't answer that question"
                # when the difference between the hints is significant, e.g. Q9-[15, 315, 135, 495]
                # as temperature is set to 0.0, the model will generate the same sequence of answers again
                # we can break here
                break
        return last_llm_answer, total_completion_tokens, total_time, generated

    def generate_prompt_with_hint(self, question, hint):
        return self.n_shot_chats(question, hint)

    def evaluation(self, data):
        """
        compare the result from llm with the correct answer
        record the evaluation result in a jsonl file
        """
        # get llm result
        llm_answer, total_completion_tokens, total_time, generated = self.progressive_hint(data)
        # convert the answer into numerical form
        answer = self.convert_answer(data['answer'])
        print('question', data['question'])
        print('answer vs llm_answer', answer, llm_answer)
        self.record_evaluation(data['question'], answer, llm_answer, generated, total_completion_tokens, total_time)
        return llm_answer == answer
