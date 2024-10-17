from main import ZERO_SHOT_RESULTS, FEW_SHOT_RESULTS, POT_RESULTS_GREEDY, POT_RESULTS_SC10, PHP_RESULTS_ZERO_SHOT, \
    PHP_RESULTS_FEW_SHOT
import json

if __name__ == '__main__':
    with open(PHP_RESULTS_FEW_SHOT, 'r') as f:
        lines = f.readlines()
        total_count = len(lines)
        correct_count = 0
        for line in lines:
            line = json.loads(line)
            if line['answer'] == line['llm_answer']:
                correct_count += 1
        print('correct rate:', correct_count / total_count)
