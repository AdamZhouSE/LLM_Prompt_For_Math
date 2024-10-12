import re
import time

from baseline import nshot_chats
import json
from call_llm import get_full_response


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
    '''Delete the extra 0 after the decimal point'''
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
    '''
    :param answer: model-predicted solution or golden answer string
    :param eos: stop token
    :return:
    '''
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


def evaluation_baseline(data):
    question = data['question']
    answer = data['answer']
    answer = handle_answer(answer)
    # zero-shot prompt
    prompt = nshot_chats(0, question)
    # call llm
    llm_answer = get_full_response(prompt)
    llm_answer = handle_answer(llm_answer)
    print('question', question)
    print('answer', answer)
    print('llm_answer', llm_answer)
    return llm_answer == answer


def handle_answer(answer):
    answer = extract_ans_from_response(answer)
    if not isinstance(answer, int):
        answer = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', answer)[0]
        answer = delete_extra_zero(answer)
    return answer


def run_evaluation():
    data_list = read_test_data()
    total_cnt = len(data_list)
    correct_cnt = 0
    start_index = 0
    while True:
        try:
            for i in range(start_index, total_cnt):
                result = evaluation_baseline(data_list[i])
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


if __name__ == '__main__':
    # test_solution = "Anna has 2 more apples than Elsa. So Anna has 2 + 5 = 7 apples. Elsa and Anna have 5 + 7 = 12 apples together. #### 12 apples"
    # answer = extract_ans_from_response(test_solution)
    # answer = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', answer)[0]
    # answer = delete_extra_zero(answer)
    # print(answer)
    run_evaluation()
