from call_llm import LLM
from evaluation import Evaluation
from constants import ZERO_SHOT, FEW_SHOT


if __name__ == '__main__':
    llm = LLM()
    evaluation = Evaluation(llm, FEW_SHOT)
    evaluation.run_evaluation()
