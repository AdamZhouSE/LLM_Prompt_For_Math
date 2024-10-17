from call_llm import LLM
from baseline import Baseline
from pot_prompt import ProgramOfThoughts
from php_prompt import ProgressiveHint


ZERO_SHOT_RESULTS = 'result/zeroshot.baseline.jsonl'
FEW_SHOT_RESULTS = 'result/fewshot.baseline.jsonl'
POT_RESULTS_GREEDY = 'result/pot_prompt_greedy.jsonl'
POT_RESULTS_SC10 = 'result/pot_prompt_sc10.jsonl'
PHP_RESULTS_ZERO_SHOT = 'result/php_prompt_zs.jsonl'
PHP_RESULTS_FEW_SHOT = 'result/php_prompt_fs.jsonl'

if __name__ == '__main__':
    # zero-shot baseline
    # llm = LLM()
    # baseline = Baseline(llm, ZERO_SHOT_RESULTS, 0)
    # baseline.run_evaluation()

    # few-shot baseline
    # llm = LLM()
    # baseline = Baseline(llm, FEW_SHOT_RESULTS, 8)
    # baseline.run_evaluation()

    # pot prompt greedy
    # llm = LLM()
    # pot = ProgramOfThoughts(llm, POT_RESULTS_GREEDY, 8)
    # pot.run_evaluation()

    # pot prompt self-consistency k = 10
    # llm = LLM(0.4)
    # pot = ProgramOfThoughts(llm, POT_RESULTS_SC10, 8, 10)
    # pot.run_evaluation()

    # php prompt greedy zero-shot
    # llm = LLM()
    # php = ProgressiveHint(llm, PHP_RESULTS_ZERO_SHOT)
    # php.run_evaluation()

    # php prompt greedy few-shot
    llm = LLM()
    php = ProgressiveHint(llm, PHP_RESULTS_FEW_SHOT, True)
    php.run_evaluation()
