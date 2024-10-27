import json


def analyze_result_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        total_count = len(lines)
        correct_count = 0
        none_count = 0
        idx = 0
        correct_idx_list = []
        for line in lines:
            idx += 1
            line = json.loads(line)
            if line['answer'] == line['llm_answer']:
                correct_count += 1
                correct_idx_list.append(idx)
            if line['llm_answer'] is None:
                none_count += 1
            # if idx == 200:
            #     break
        print('correct rate:', correct_count / total_count)
        # print(none_count)
        # print(correct_count / idx)
        return correct_idx_list


if __name__ == '__main__':
    php = analyze_result_file('result/progressive_hint/php_sc10.jsonl')
    pot = analyze_result_file('result/pro_plus/pot_new_prompt.jsonl')
    ppp = analyze_result_file('result/pro_plus/method_combine.jsonl')
    list1 = list(set(php) - set(pot))
    list2 = list(set(pot) - set(php))
    print(len(list1))
    print(len(list2))
    intersection_set = list(set(list2) & set(ppp))
    print(len(intersection_set))
