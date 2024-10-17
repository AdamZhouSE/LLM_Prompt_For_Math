from pot_prompt import ProgramOfThoughts


class ProPlusPrompt(ProgramOfThoughts):
    """
    Based on pot and php prompt, we design a new prompt called PPP (ProPlusPrompt)
    Idea: In pot, we get python code from llm and run it locally to get the answer.
        The code may be wrong in some cases, including value grounding error and logic generation error.
        So we use the execution result as a new prompt to ask llm to re-generate the code.
        And if code is runnable and the answer is same as the previous one, we stop.
    """
    def __init__(self, llm, record_path):
        super().__init__(llm, record_path)
