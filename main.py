from call_llm import LLM
from evaluation import Evaluation
from evaluation_pot import EvaluationPot
from constants import ZERO_SHOT, FEW_SHOT


ZERO_SHOT_RESULTS = 'zeroshot.baseline.jsonl'
FEW_SHOT_RESULTS = 'fewshot.baseline.jsonl'
POT_RESULTS = 'pot_prompt_greedy.jsonl'

if __name__ == '__main__':
    llm = LLM(0.0, 1)
    # evaluation = Evaluation(llm, ZERO_SHOT, ZERO_SHOT_RESULTS)
    # evaluation.run_evaluation()
    evaluation_pot = EvaluationPot(llm, POT_RESULTS, 1)
    evaluation_pot.run_evaluation()
