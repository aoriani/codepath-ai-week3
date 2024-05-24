import json
import os
from openai import OpenAI

facts = []
with open('data/facts.jsonl', 'r') as file:
    for line in file:
        facts.append(json.loads(line))

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def generate_qa(fact, n=20):
    prompt_text = f"""
    Based on the following fact, generate an array of {n} variations of question-answer pairs.
    Each pair should be formatted as a JSON object with "messages" containing "user" and "assistant" roles.
    Ensure that the output is in JSON format.

    Each question should be unique, clearly phrased, and reflect how users might ask about this fact.
    The corresponding answer should be accurate, contextually relevant, and phrased differently from the other answers.
    Ensure diversity in question types (who, what, where, when, why) and avoid repetitive phrasing.

    Fact: "{fact}"

    Example output format:

    {{"data": [{{"messages": [{{"role": "user", "content": "What is the capital of France?"}}, {{"role": "assistant", "content": "The capital of France is Paris."}}]}},
    {{"messages": [{{"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}}, {{"role": "assistant", "content": "The author of 'Romeo and Juliet' is William Shakespeare."}}]}},
    {{"messages": [{{"role": "user", "content": "How far is the Moon from Earth?"}}, {{"role": "assistant", "content": "The distance from the Moon to Earth is approximately 384,400 kilometers."}}]}}]
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant tasked with generating training data for fine-tuning a gpt-3.5-turbo model in JSON format"},
            {"role": "user", "content": prompt_text}
        ],
        response_format={"type": "json_object"},
        temperature=0.5
    )

    print(response.choices[0].message.content.strip())
    try:
        qa_array = json.loads(response.choices[0].message.content.strip())["data"]
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
        return [], []

    # Splitting the generated QA pairs into training and validation sets
    validation_size = int(len(qa_array) * 0.2)
    validation_set = qa_array[:validation_size]
    training_set = qa_array[validation_size:]

    return training_set, validation_set


training_set = []
validation_set = []

for fact in facts:
    training, validation = generate_qa(fact['fact'], n=10)
    training_set.extend(training)
    validation_set.extend(validation)

with open('trainingsets/iteration1/facts_training.jsonl', 'w') as train_outfile:
    for qa in training_set:
        train_outfile.write(json.dumps(qa) + '\n')

with open('trainingsets/iteration1/facts_validation.jsonl', 'w') as valid_outfile:
    for qa in validation_set:
        valid_outfile.write(json.dumps(qa) + '\n')
