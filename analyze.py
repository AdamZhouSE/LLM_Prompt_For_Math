from main import POT_RESULTS
import json

if __name__ == '__main__':
    with open(POT_RESULTS, 'r') as f:
        lines = f.readlines()
        total_count = len(lines)
        correct_count = 0
        for line in lines:
            line = json.loads(line)
            if line['answer'] == line['llm_answer']:
                correct_count += 1
        print('correct rate:', correct_count / total_count)