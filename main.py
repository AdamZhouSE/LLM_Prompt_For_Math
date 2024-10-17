from call_llm import LLM
from baseline import Baseline
from pot_prompt import ProgramOfThoughts
from php_prompt import ProgressiveHint


ZERO_SHOT_RESULTS = 'result/zeroshot.baseline.jsonl'
FEW_SHOT_RESULTS = 'result/fewshot.baseline.jsonl'
POT_RESULTS = 'result/pot_prompt_sc.jsonl'
PHP_RESULTS = 'result/php_prompt_greedy.jsonl'

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
    # pot = ProgramOfThoughts(llm, POT_RESULTS, 8)
    # pot.run_evaluation()

    # pot prompt self-consistency
    # llm = LLM(0.4)
    # pot = ProgramOfThoughts(llm, POT_RESULTS, 8, 10)
    # pot.run_evaluation()

    # php prompt greedy
    llm = LLM()
    php = ProgressiveHint(llm, PHP_RESULTS)
    php.run_evaluation()
