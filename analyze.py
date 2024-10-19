import json

if __name__ == '__main__':
    with open('result/progressive_hint/php_fs.jsonl', 'r') as f:
        lines = f.readlines()
        total_count = len(lines)
        correct_count = 0
        none_count = 0
        for line in lines:
            line = json.loads(line)
            if line['answer'] == line['llm_answer']:
                correct_count += 1
            if line['llm_answer'] is None:
                none_count += 1
                print(line['answer'], line['llm_answer'], line['generated'])
        print('correct rate:', correct_count / total_count)
        print(none_count)
