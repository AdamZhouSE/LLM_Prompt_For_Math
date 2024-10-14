from call_llm import LLM
from evaluation import Evaluation
from constants import ZERO_SHOT, FEW_SHOT


ZERO_SHOT_RESULTS = 'zeroshot.baseline.jsonl'
FEW_SHOT_RESULTS = 'fewshot.baseline.jsonl'

if __name__ == '__main__':
    llm = LLM()
    evaluation = Evaluation(llm, FEW_SHOT, FEW_SHOT_RESULTS)
    evaluation.run_evaluation()
