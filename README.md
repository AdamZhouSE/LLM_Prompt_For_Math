# Introduction

In this research, we are going to explore the capabilities of large language modules(LLMs) for mathematical reasoning.
We will use the grade school math dataset [**GSM8K**](https://github.com/openai/grade-school-math) and the powerful
open-source model **Llama-3.1-8B-Instruct** to experiment.

By using different prompt technologies, we can instruct LLMs for desired outcomes without updating model weights. First
of all, we implement the **baseline** evaluation for the model, which tests zero-shot and few-shot prompts on the
dataset. Secondly, we implement two methods, [**Program of Thoughts**](https://arxiv.org/pdf/2211.12588)(PoT) and [*
*Progressive Hint Prompt**](https://arxiv.org/pdf/2304.09797)(PHP), from two papers and evaluate their performance.
Finally, we try to combine both methods and design a new prompt method called **Pro Plus Prompt**(PPP) to improve the
accuracy of the model's math problem-solving and achieve a high accuracy of **92.27%**.

## Code Structure

The project has been uploaded to [**Github**](https://github.com/AdamZhouSE/LLM_Prompt_For_Math). Feel free to check the
code and results. Here is a brief introduction to the project structure and functions of files.

```bat
├── baseline.py
├── php_prompt.py
├── pot_prompt.py
├── pro_plus_prompt.py
├── evaluation.py
├── call_llm.py
├── main.py
├── analyze.py
├── data
├── prompt
│   ├── php_cot_prompt.txt
│   ├── pot_prompt_improve.txt
│   └── pot_prompt_original.txt
└── result
```

We have four prompt methods, including baseline, PoT, PHP, and PPP. You can find implementations of them in
corresponding Python files, which are the first four files displayed above. We also have an `evaluation.py`, which is
inherited by each of the four methods and overwritten to satisfy the unique assessment requirements of each method,
including different prompts, self-consistency, result record, and etc. The few-shot prompts for different methods are
saved in the directory `prompt`. The result files are saved in the directory `result`.

In `main.py`, we write four methods to run evaluations for different prompts by passing different parameters. As you can
see, it is very easy to decide whether to use zero-shot or few-shot, greedy or self-consistency and adjust
hyperparameters due to our well-organized object-oriented programming design. Below is the description of each
parameter.

* `filepath`: the path to save the result `.jsonl` file
* ` num_of_shots`: decide whether to use zero-shot or few-shot and how many shots to use
* `num_of_trials`: decide how many interaction times with the model in self-consistency
* `temperature | top_p`: hyperparameters used in self-consistency

```python
def run_baseline(filepath, num_of_shots=0)


    def run_pot(filepath, num_of_shots=0, num_of_trials=1, temperature=0.0, top_p=1.0)


    def run_php(filepath, num_of_shots=0, num_of_trials=1, temperature=0.0, top_p=1.0)


    def run_ppp(filepath, temperature=0.0, top_p=1.0)
```

## Format of Result Files

Here is the format of each line in result files `*.jsonl`. Despite three mandatory fields, `input`, `prompt`,
and `output`, we also put extracted `answer` and `llm_answer` in it for comparison and recorded the cost of tokens and
time for each question. When interacting with the model in multiple rounds, the output will be in a list format to
record all the responses from the model.

```json
{
  "input": "math question",
  "prompt": [
    {
      "role": "system",
      "content": "system prompt"
    },
    {
      "role": "user",
      "content": "math question"
    }
  ],
  "output": "llm full response",
  "answer": "18",
  "llm_answer": "18",
  "completion_tokens": 103,
  "time": 0.15088224411010742
}
```

# Baseline

For the baseline zero-shot approach, we simply inform the model of its task and request results without using any
advanced prompting techniques. In the few-shot approach, we provide several question-and-answer examples to guide the
model on how to perform calculations and the expected format for its answers. As shown in the table, even the zero-shot
accuracy exceeds 80%, and the few-shot accuracy reaches 84.23%, indicating that the model has developed relatively
strong mathematical reasoning skills.

> System Prompt: Your task is to solve a series of math word problems by providing the final answer. Use the
> format #### [value] to highlight your answer. For example, if the answer is 560, you should write #### 560.

> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done,
> there will be 21 trees. How many trees did the grove workers plant today?
>
> Answer: There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there
> must have been 21 - 15 = <<21-15=6>>6 trees that were planted.\n#### 6

| Baseline      | Accuracy       |
|---------------|----------------|
| zero-shot     | 82.64%         |
| few-shot(n=8) | 84.23%(+1.60%) |

# Program of Thoughts Prompt

The idea of PoT is to separate computation from reasoning to improve the math problem-solving ability of LLMs. It uses
LLMs to transform the math problem into Python code, which represents the calculation process. Then, run the code
locally to get the result. In other words, PoT is the intermediate process of solving the problem. It assumes that LLMs
may be weak in complex calculations, so using code can ensure accurate and reliable computation.

This time, we ask the model to generate Python code and assign the result to a variable `ans` so that we can extract it
after executing the code. In a few shots, we provide a few examples for the model to know what the input and output
should look like. The system prompt and an example of a few-shot prompt are shown below.

> Your task is to solve math word problems using Python code. Assign the final result to a variable called `ans`.
> Provide only runnable Python code.

```python
# Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
# Python code, return ans
total_eggs = 16
eaten_eggs = 3
baked_eggs = 4
sold_eggs = total_eggs - eaten_eggs - baked_eggs
dollars_per_egg = 2
ans = sold_eggs * dollars_per_egg
```

After getting the response from the model, we use the method `safe_execute` to run the code in the local environment and
get the result from the variable `ans`.

```python
def safe_execute(self, code_string: str):
    def execute(x):
        try:
            exec(x)
            locals_ = locals()
            return locals_.get('ans', None)
        except Exception:
            return None

    try:
        ans = func_timeout.func_timeout(5, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans = None
    return ans
```

Before conducting the tests, some errors in the few-shot prompts provided in the article need to be corrected. The
second line of the code part in the picture below has an undefined variable `num_of_blue_fiber`. It should be the
variable defined in the first line. So we correct it into a new version.

```python
bolts_of_blue_fiber = 2
num_of_white_fiber = num_of_blue_fiber / 2
ans = num_of_blue_fiber + num_of_white_fiber
# correct version
num_of_blue_fiber = 2
num_of_white_fiber = num_of_blue_fiber / 2
ans = num_of_blue_fiber + num_of_white_fiber
```

Another mistake is here. There is an extra `b` after `num` in the variable `numb_of_chickens`.

```python
numb_of_chickens = 20
cups_for_each_chicken = 3
cups_for_all_chicken = num_of_chickens * cups_for_each_chicken
cups_in_the_morning = 15
cups_in_the_afternoon = 25
ans = cups_for_all_chicken - cups_in_the_morning - cups_in_the_afternoon
```

In PoT and other methods below, we employ an optimization method called **self-consistency**(sc) or the majority vote.
Its core idea is to interact with the model in multiple rounds under conditions of randomness(set temperature> 0) to
obtain the most frequently occurring result. This method can enhance the performance of the model but at the cost of
consuming more tokens and time. However, unlike OpenAI's API, the llama API sponsored by SambaNova System does not
support a native function to return results `k` times. So, we use a for loop to call the API `k` times to simulate this
functionality.

| Baseline      | Accuracy | PoT           | Accuracy(vs. zero-shot) |
|---------------|----------|---------------|-------------------------|
| zero-shot     | 82.63%   | few-shot(n=8) | 77.79%(-4.84%)          |
| few-shot(n=8) | 84.23%   | sc(k=10)      | 84.31%(+1.68%)          |

In PoT, we set up two tests, few-shot and self-consistency(**k=10, temperature=0.7, top-p=0.95**). We don't test k=40
like the paper does as it will cost too much time. Although PoT sounds persuasive, the result shows that the performance
of this method is even worse than the zero-shot of the baseline. The few-shot of PoT is 77.79%, which is 6.44% lower
than the few-shot of baseline. After using self-consistency, the accuracy only improves slightly, but the token
consumption and time increase several times over. Several reasons may account for the decrease. Below are examples and
analyses where the model answered correctly in the baseline but answered incorrectly in PoT.

1. The model may generate the wrong code, which can not be executed. For example, in question 34, the model incorrectly
   generated an undefined variable `x`. It can be seen that the model does not convert the math problem into correct
   Python code but instead uses an equation-like approach. `x` appears to be an unknown variable. This will cause an
   error in code execution, and `llm_answer` in the result file is `null`. There are a total of 31 problems in few-shot
   PoT.

   > Gretchen has 110 coins. There are 30 more gold coins than silver coins. How many gold coins does Gretchen have?

   ```python
   total_coins = 110
   let_say_silver_coins = x
   gold_coins = x + 30
   ans = x + 30
   x = total_coins / 2
   ans = x + 30
   ```

2. There are floating-point precision limitations in Python. For example, in question 31, the execution result of the
   code generated by the model is `109.00000000000001`. If we transform the math problem into code, there will be new
   problems in calculation due to the characteristics of Python.

   ```python
   ratio_of_darrell_age = 7
   ratio_of_allen_age = 11
   total_ratio = ratio_of_darrell_age + ratio_of_allen_age
   total_age_now = 162
   darrell_age_now = (ratio_of_darrell_age / total_ratio) * total_age_now
   allen_age_now = (ratio_of_allen_age / total_ratio) * total_age_now
   ans = allen_age_now + 10
   ```

It is not fair to claim that PoT's performance is poor, especially given that it achieves a score of 77.79% in a
few-shot context. We shouldn't be overly critical of a student who scored nearly 80 out of 100 on an exam. As mentioned
in the baseline section, the model has already shown a strong capability in mathematical reasoning. Therefore, it's
possible to experience negative optimization results after changing the method. In the following sections of this
report, we will explore ways to optimize this method.

# Progressive Hint Prompt

The idea behind PHP is to use the model's previously generated results as hints to guide it. In multiple rounds of
interaction with the model, the process **ends if the same result is obtained in two rounds**.

In this method, we use the same system prompt as the baseline and a complex chain of thought(CoT) prompt, which can be
seen in the file `php_cot_prompt.txt`. Here is an example. First of all, we append the model's answers from previous
interactions as hints after the question. Secondly, we inform the model to answer the questions based on the hints in
question at the beginning of the answer. Finally, PHP is orthogonal to CoT, so we can apply a complex version to help
the model perform better.



> Question: Susy goes to a large school with 800 students, while Sarah goes to a smaller school with only 300 students.
> At the start of the school year, Susy had 100 social media followers. She gained 40 new followers in the first week of
> the school year, half that in the second week, and half of that in the third week. Sarah only had 50 social media
> followers at the start of the year, but she gained 90 new followers the first week, a third of that in the second week,
> and a third of that in the third week. After three weeks, how many social media followers did the girl with the most
> total followers have? **(Hint: The answer is near to 180, 160)**.

> Answer: **We know the Answer Hints: 180, 160. With the Answer Hints: 180, 160, we will answer the question.**
> Let's think step by step.
> After one week, Susy has 100+40 = 140 followers.
> In the second week, Susy gains 40/2 = 20 new followers.
> In the third week, Susy gains 20/2 = 10 new followers.
> In total, Susy finishes the three weeks with 140+20+10 = 170 total followers.
> After one week, Sarah has 50+90 = 140 followers.
> After the second week, Sarah gains 90/3 = 30 followers.
> After the third week, Sarah gains 30/3 = 10 followers.
> So, Sarah finishes the three weeks with 140+30+10 = 180 total followers.
> Thus, Sarah is the girl with the most total followers with a total of 180.
> The answer is 180
>
> *#### 180*

| Baseline      | Accuracy | PHP           | Accuracy(vs. zero-shot) |
|---------------|----------|---------------|-------------------------|
| zero-shot     | 82.64%   | zero-shot     | 80.06%(-2.58%)          |
| few-shot(n=8) | 84.23%   | few-shot(n=8) | 85.52%(+2.88%)          |
|               |          | sc(k=10)      | 88.63%(+5.99%)          |

In PHP, we set up three tests: zero-shot, few-shot, and self-consistency(**k=10, temperature=0.4, top-p=0.95**). The
result shows in zero-shot, PHP is worse than baseline, while in few-shot, PHP is better. In zero-shot, there are a total
of **1134 problems that end in two trials**, taking up 86% of the dataset. The number is 1159, even higher in few-shot.
That is to say, in most cases, the model's answers are consistent. It's reasonable because the model is proven to be
able to answer most of the questions correctly the first time in baseline.

The decline in zero-shot performance may be due to hints interfering with the model, causing it to make mistakes in
subsequent interactions on questions it initially answered correctly. The improvement in accuracy in the few-shot
approach is partly due to the multi-turn interaction verification of this method and partly because we use a complex
chain of thought in this approach. For sc, this time, we set the temperature to 0.4 after testing several different
hyperparameters. It appears that PHP requires less randomness than PoT. PHP gets 88.63% of accuracy after
self-consistency, which shows the power of this optimization method. Still, we do not test k=40 because it will cost so
much time. But without doubt it will reach a higher accuracy.

This method also has limitations. When the results vary, they may confuse the model, and it will return "I can't answer
that question". For example, in question 9, the list of hints is `[15, 315, 135, 495]`. The difference between the hints
is significant, and as the temperature is set to 0.0, the model will generate the same sequence of answers again. Even
if we try more times, the model is still unable to answer.

<div STYLE="page-break-after: always;"></div>

# Pro Plus Prompt

## Improve Prompt

Although we have corrected some mistakes in the PoT prompt, there is still a big problem. PoT uses questions in the test
set as its prompt. This may cause overfitting and limitations in model generalization. So, we apply the questions used
in the baseline and generate code through GitHub Copilot. The new prompt is saved in the file `pot_prompt_improve.txt`.
There's an example of an answer below. After using the new prompt, the accuracy reaches 80.74%, comparing 2.95% higher
to the original PoT.

```python
# solution using Python:
def solution():
    """There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"""
    trees_initial = 15
    trees_after = 21
    trees_added = trees_after - trees_initial
    result = trees_added
    return result
```

## Combine

We extract the numbers of correctly answered questions from each of the two files, `php_sc10.jsonl`
and `ppp_new_prompt.jsonl`, and then take their difference set. **Our findings show that PHP correctly answers 160
questions that PoT misses, while PoT correctly answers 56 questions that PHP misses.** This indicates that these two
methods can complement one another to some extent.

Building on our findings, we develop a new prompting method called Pro Plus Prompt that combines PoT and PHP approaches.
In each interaction, results from both methods are retrieved and compared; if they match, the answer is considered
correct and returned. If the results differ, a maximum attempt limit is set(max_times=10), after which the most
frequently occurring result is returned. The idea behind this method is to **cross-validate the results of PHP and PoT
**. It also incorporates the **self-consistency** approach, where the maximum attempt count effectively serves as the
value of *k* in self-consistency.

```python
for i in range(self.max_times):
    # use two methods to get the answer
    llm_answer_pot, tokens_pot, time_pot, generated_pot, prompt_pot = self.program_of_thought(data)
    llm_answer_php, tokens_php, time_php, generated_php, prompt_php = self.php.progressive_hint(data)
    # get the same answer, return
    if llm_answer_pot == llm_answer_php:
        llm_answer = llm_answer_pot
        break
    # different answers -> get majority vote
    if llm_answer_pot is not None:
        result_counter.update([llm_answer_pot])
    if llm_answer_php is not None:
        result_counter.update([llm_answer_php])
```

The performance of combined methods is **92.27%**, reaching the highest accuracy in this research. The experimental
results confirm our previous hypothesis: since there is a discrepancy between the questions each method answers
correctly, these two approaches can be complementary. We can verify this using the code in `analyze.py`. The length
of `list2` is the number of questions that PoT answers correctly while PHP answers wrong. We calculate **its
intersection set** with the result of PPP. The length of this set is **51**. The accuracy of the PPP improved by 3.64%
compared to PHP SC(k=10), equating to 48 questions, which is very close to 51. In other words, this approach can be
understood as augmenting the PHP method with the PoT approach, enabling the model to correctly answer questions it
previously answered incorrectly when using a single method alone.

```python
list1 = list(set(php) - set(pot))
list2 = list(set(pot) - set(php))
print(len(list1))
print(len(list2))
intersection_set = list(set(list2) & set(ppp))
print(len(intersection_set))
```

| Method                 | Accuracy(vs. zero-shot)                 |
|------------------------|-----------------------------------------|
| Baseline zero-shot     | 82.63%                                  |
| Baseline few-shot(n=8) | 84.23%(+1.60%)                          |
| PoT few-shot(n=8)      | 77.79%(-4.84%)                          |
| PoT sc(k=10)           | 84.31%(+1.68%)                          |
| PHP zero-shot          | 80.06%(-2.57%)                          |
| PHP few-shot(n=8)      | 85.52%(+2.89%)                          |
| PHP sc(k=10)           | 88.63%(+6.00%)                          |
| PoT New Prompt         | 80.74%(-1.89%) (+2.95% to PoT)          |
| PPP(temperature=0.7)   | <font color='red'>92.27%(+9.64%)</font> |

# Conclusion

This research demonstrates that different prompting methods have varying levels of accuracy in instructing LLMs to solve
math problems. In baseline, we know that the performance of the model can be enhanced by moving from zero-shot to
few-shot. The PoT method, while innovative in using code generation, encounter limitations due to code errors. The PhP
provides a better improvement in accuracy from zero-shot, few-shot to self-consistency.

The new method PPP, which combines PoT and PHP with cross-validation and self-consistency, achieve the highest accuracy.
It suggests that integrated prompting methods may provide benefits in enhancing LLMs performance on complex mathematical
reasoning, highlighting the potential for further exploration of integrated prompting strategies.
