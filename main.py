from baseline import Baseline
from pot_prompt import ProgramOfThoughts
from php_prompt import ProgressiveHint
from pro_plus_prompt import ProPlusPrompt
from call_llm import LLM


def run_baseline(filepath, num_of_shots=0):
    llm = LLM()
    baseline = Baseline(llm, filepath, num_of_shots)
    baseline.run_evaluation()


def run_pot(filepath, num_of_shots=0, num_of_trials=1, temperature=0.0, top_p=1.0):
    llm = LLM(temperature, top_p)
    pot = ProgramOfThoughts(llm, filepath, num_of_shots, num_of_trials)
    pot.run_evaluation()


def run_php(filepath, num_of_shots=0, num_of_trials=1, temperature=0.0, top_p=1.0):
    llm = LLM(temperature, top_p)
    php = ProgressiveHint(llm, filepath, num_of_shots, num_of_trials)
    php.run_evaluation()


def run_ppp(filepath, num_of_shots=0, max_hint=1, num_of_trials=1, temperature=0.0, top_p=1.0):
    llm = LLM(temperature, top_p)
    ppp = ProPlusPrompt(llm, filepath, num_of_shots, max_hint, num_of_trials)
    ppp.run_evaluation()


if __name__ == '__main__':
    # baseline zero-shot
    # run_baseline('result/baseline/zeroshot.baseline.jsonl')
    # baseline few-shot
    # run_baseline('result/baseline/fewshot.baseline.jsonl', 8)
    # TODO pot zero-shot
    # run_pot('pot_zs.jsonl')
    # pot few-shot
    # run_pot('pot_fs.jsonl', 8)
    # pot sc10
    # run_pot('pot_sc10.jsonl', 8, 10, 0.7, 0.95)
    # php zero-shot
    # run_php('php_zs.jsonl')
    # php few-shot
    # run_php('php_fs.jsonl', 8)
    # php sc10
    # run_php('php_sc10.jsonl', 8, 10, 0.4, 0.95)
    # ppp new-prompt
    # run_ppp('ppp_new_prompt.jsonl', 8, 1, 1)
    # ppp with-hint
    # run_ppp('ppp_with_hint.jsonl', 8, 5, 1)
    # ppp with-hint sc10
    run_ppp('ppp_with_hint_sc10.jsonl', 8, 5, 10, 0.7, 0.95)
