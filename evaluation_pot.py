import time

from evaluation import Evaluation
from constants import POT_PROMPT
import func_timeout
from pot_prompt import ProgramOfThoughts
from collections import Counter
from call_llm import LLM


class EvaluationPot(Evaluation):
    def __init__(self, llm, record_path, num_of_trials=1):
        super().__init__(llm, POT_PROMPT, record_path)
        self.num_of_trials = num_of_trials

    def evaluation(self, data):
        """
        compare the result from llm with the correct answer
        record the evaluation result in a jsonl file
        """
        # generate prompt
        prompt = self.generate_prompt(data['question'])
        total_completion_tokens = 0
        total_time = 0.0
        # call llm
        result_counter = Counter()
        # number of trials default 1 -> greedy
        # multiple trials -> self-consistency
        for i in range(self.num_of_trials):
            full_response = self.llm.get_full_response(prompt)
            total_completion_tokens += full_response['completion_tokens']
            total_time += full_response['time']
            llm_answer = self.convert_pot_answer(full_response['answer'])
            print(llm_answer)
            if llm_answer is not None:
                result_counter.update([llm_answer])
            if self.num_of_trials > 1:
                time.sleep(1.5)
        if self.num_of_trials > 1:
            time.sleep(5)
        if len(result_counter) > 0:
            llm_answer = result_counter.most_common(1)[0][0]
        else:
            llm_answer = None

        # convert the answer into numerical form
        answer = self.convert_answer(data['answer'])
        print('question', data['question'])
        print('answer vs llm_answer', answer, llm_answer)
        self.record_evaluation(data['question'], data['answer'], llm_answer, total_completion_tokens, total_time)
        return llm_answer == answer

    def generate_prompt(self, question):
        prompt_pot = ProgramOfThoughts()
        return prompt_pot.n_shot_chats(8, question)

    def convert_pot_answer(self, answer):
        exec_result = self.safe_execute(answer)
        float_answer = self.floatify_ans(exec_result)
        return self.delete_extra_zero(float_answer)

    def safe_execute(self, code_string: str, keys=None):
        def execute(x):
            try:
                exec(x)
                locals_ = locals()
                if keys is None:
                    return locals_.get('ans', None)
                else:
                    return [locals_.get(k, None) for k in keys]
            except Exception:
                return None

        try:
            ans = func_timeout.func_timeout(5, execute, args=(code_string,))
        except func_timeout.FunctionTimedOut:
            ans = None

        return ans

    def floatify_ans(self, ans):
        if ans is None:
            return None
        elif type(ans) == dict:
            ans = list(ans.values())[0]
        elif type(ans) == bool:
            ans = ans
        elif type(ans) in [list, tuple]:
            if not ans:
                return None
            else:
                try:
                    ans = float(ans[0])
                except Exception:
                    ans = str(ans[0])
        else:
            try:
                ans = float(ans)
            except Exception:
                ans = str(ans)
        return ans


if __name__ == '__main__':
    llm = LLM(0.4, 1)
    evaluation_pot = EvaluationPot(llm, 'pot_prompt.jsonl', 10)
    data = {
        "question": "Olivia uploaded 72 pictures to Facebook.  She put the same number of the pics into 8 albums.  3 of the albums were selfies only and 2 of the albums were portraits.  How many portraits and selfies did she have?",
        "answer": "Olivia had 72 pictures / 8 albums = <<72/8=9>>9 picture per album.\nOlivia had 3 selfie albums * 9 pictures = <<3*9=27>>27 pictures.\nOlivia had 2 portrait albums * 9 pictures = <<2*9=18>>18 pictures.\nThe total of portraits and selfies for Olivia is 27 + 18 = <<27+18=45>>45 pictures.\n#### 45"}
    evaluation_pot.evaluation(data)
