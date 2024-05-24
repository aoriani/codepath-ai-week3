import json
import os
from openai import OpenAI

gpt35_model = "gpt-3.5-turbo-0125"
gpt35_fine_tuned_model = "ft:gpt-3.5-turbo-0125:personal:exp6-halvetrain:9Sq6uuYB"

questions = []
with open("data/evaluation_questions.jsonl", 'r') as file:
    for line in file:
        questions.append(json.loads(line))

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def make_question(model: str, q: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": q}],
        temperature=0.5
    )
    return response.choices[0].message.content


with open("evaluation/evaluation.txt", 'w') as eval_file:
    for question in questions:
        q = question['question']
        eval_file.write(f"Question: {q}\n")
        eval_file.write(f"gpt3.5-turbo -> {make_question(gpt35_model, q)}\n")
        eval_file.write(f"fine-tuned -> {make_question(gpt35_fine_tuned_model, q)}\n")
        eval_file.write("\n")
