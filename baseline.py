from evaluation import Evaluation

gsm8k_n_shots = [
    (
        'There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?',
        'There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there must have been 21 - 15 = <<21-15=6>>6 trees that were planted.\n#### 6'
    ),
    (
        'If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?',
        'There are originally 3 cars. Then 2 more cars arrive. Now 3 + 2 = <<3+2=5>>5 cars are in the parking lot.\n#### 5'
    ),
    (
        'Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?',
        'Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = <<32+42=74>>74. After eating 35, they had 74 - 35 = <<74-35=39>>39 pieces left in total.\n#### 39'
    ),
    (
        'Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?',
        'Jason had 20 lollipops originally. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = <<20-12=8>>8 lollipops.\n#### 8'
    ),
    (
        'Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?',
        'Shawn started with 5 toys. He then got 2 toys each from his mom and dad. So he got 2 * 2 = <<2*2=4>>4 more toys. Now he has 5 + 4 = <<5+4=9>>9 toys.\n#### 9'
    ),
    (
        'There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?',
        'There were originally 9 computers. For each day from monday to thursday, 5 more computers were installed. So 4 * 5 = <<4*5=20>>20 computers were added. Now 9 + 20 = <<9+20=29>>29 computers are now in the server room.\n#### 29'
    ),
    (
        'Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?',
        'Michael started with 58 golf balls. He lost 23 on Tuesday, and lost 2 more on wednesday. So he had 58 - 23 = <<58-23=35>>35 at the end of Tuesday, and 35 - 2 = <<35-2=33>>33 at the end of wednesday.\n#### 33'
    ),
    (
        'Olivia has $23. She bought five bagels for $3 each. How much money does she have left?',
        'Olivia had 23 dollars. She bought 5 bagels for 3 dollars each. So she spent 5 * 3 = <<5*3=15>>15 dollars. Now she has 23 - 15 = <<23-15=8>>8 dollars left.\n#### 8'
    )
]


class Baseline(Evaluation):
    def __init__(self, llm, prompt_method, record_path):
        super().__init__(llm, prompt_method, record_path)
        self.n_shot_list = gsm8k_n_shots

    def question_prompt(self, question):
        return f'Question: {question}'

    def answer_prompt(self, answer):
        return f"Answer:\nLet's think step by step.\n{answer}"

    def n_shot_chats(self, n: int, question: str):
        chats = [
            {"role": "system",
             "content": "Your task is to solve a series of math word problems by providing the final answer. "
                        "Use the format #### [value] to highlight your answer. "
                        "For example, if the answer is 560, you should write #### 560."}
        ]

        for q, a in self.n_shot_list[:n]:
            chats.append(
                {"role": "user", "content": self.question_prompt(q)})
            chats.append(
                {"role": "assistant", "content": self.answer_prompt(a)})

        chats.append({"role": "user", "content": self.question_prompt(question)})
        return chats

    def generate_prompt(self, question):
        return self.n_shot_chats(self.num_of_shots, question)
