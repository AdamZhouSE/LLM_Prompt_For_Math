import re
import time

from baseline import nshot_chats
import json
from constants import ZERO_SHOT, FEW_SHOT


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def delete_extra_zero(n):
    """Delete the extra 0 after the decimal point"""
    try:
        n = float(n)
    except:
        # print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')  # 删除小数点后多余的0
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)  # 只剩小数点直接转int，否则转回float
        n = str(n)
        return n


def extract_ans_from_response(answer: str, eos=None):
    """
    :param answer: model-predicted solution or golden answer string
    :param eos: stop token
    :return:
    """
    if eos:
        answer = answer.split(eos)[0].strip()

    answer = answer.split('####')[-1].strip()

    for remove_char in [',', '$', '%', 'g']:
        answer = answer.replace(remove_char, '')

    try:
        return int(answer)
    except ValueError:
        return answer


def read_test_data():
    data_list = []
    with open('data/test.jsonl', 'r') as f:
        data = f.readlines()
        for line in data:
            line = json.loads(line)
            data_list.append(line)
    return data_list


def convert_answer(answer):
    answer = extract_ans_from_response(answer)
    if not isinstance(answer, int):
        answer = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', answer)[0]
        answer = delete_extra_zero(answer)
    return answer


def generate_prompt(prompt_method, question):
    if prompt_method == ZERO_SHOT:
        return nshot_chats(0, question)
    elif prompt_method == FEW_SHOT:
        return nshot_chats(8, question)
    else:
        # zero-shot by default
        return nshot_chats(0, question)


class Evaluation:
    def __init__(self, llm, prompt_method, record_path, local_model=False):
        self.data_list = read_test_data()
        self.llm = llm
        self.prompt_method = prompt_method
        self.record_path = record_path
        # whether to use local model (in case api rate limit exceed)
        self.local_model = local_model

    def evaluation(self, data):
        # generate prompt
        prompt = generate_prompt(self.prompt_method, data['question'])
        # call llm
        full_response = self.llm.get_full_response(prompt)
        # convert the answer into numerical form
        answer = convert_answer(data['answer'])
        llm_answer = convert_answer(full_response['answer'])
        print('question', data['question'])
        print('answer', answer)
        print('llm_answer', llm_answer)
        self.record_evaluation(data['question'], data['answer'], llm_answer == answer, full_response)
        return llm_answer == answer

    def record_evaluation(self, question, answer, result, response):
        """
        record the evaluation result in a jsonl file

        Args:
            question: question to be asked
            answer: complete correct answer
            result: whether llm_answer is correct
            response: response from llm
        """
        with open(self.record_path, 'a') as f:
            record = {
                'question': question,
                'answer': answer,
                'llm_answer': response['answer'],
                'result': result,
                'completion_tokens': response['completion_tokens'],
                'prompt_tokens': response['prompt_tokens'],
                'total_tokens': response['total_tokens'],
                'time': response['time']
            }
            f.write(json.dumps(record) + '\n')

    def run_evaluation_local(self):
        total_cnt = len(self.data_list)
        correct_cnt = 0
        start_index = 0
        for data in self.data_list:
            result = self.evaluation(data)
            if result:
                print('correct')
                correct_cnt += 1
            start_index += 1
            print('correct_rate', correct_cnt / start_index)
        print('total correct_rate', correct_cnt / total_cnt)

    def run_evaluation(self):
        total_cnt = len(self.data_list)
        correct_cnt = 0
        start_index = 0
        while True:
            try:
                for i in range(start_index, total_cnt):
                    result = self.evaluation(self.data_list[i])
                    start_index += 1
                    print('index', start_index)
                    if result:
                        correct_cnt += 1
                    print('correct_rate', correct_cnt / start_index)
                    time.sleep(1)
                    if start_index % 10 == 0:
                        time.sleep(10)
                break
            except Exception as e:
                print('abort', e)
                # rate_limit_exceed, sleep for 60s
                time.sleep(60)
        print('total correct_rate', correct_cnt / total_cnt)

# if __name__ == '__main__':
#     # test_solution = "Anna has 2 more apples than Elsa. So Anna has 2 + 5 = 7 apples. Elsa and Anna have 5 + 7 = 12 apples together. #### 12 apples"
#     # answer = extract_ans_from_response(test_solution)
#     # answer = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', answer)[0]
#     # answer = delete_extra_zero(answer)
#     # print(answer)
