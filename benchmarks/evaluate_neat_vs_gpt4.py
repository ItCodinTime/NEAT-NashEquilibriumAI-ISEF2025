"""
Benchmark example: Compare NEAT and GPT-4 on a simple QA task (mock logic for demo)
"""
from neat_main import NEATTrainer

def gpt4_predict(question):
    return "sample_answer"

# Dummy logic/data (replace with real benchmarking)
test_questions = [
    {"input": [0.1, 0.2, 0.3, 0.1], "label": 1},
    {"input": [0.7, 0.3, 0.2, 0.2], "label": 0},
]

correct_neat, correct_gpt = 0, 0
for q in test_questions:
    neat_pred = 1 if sum(q['input']) > 0.5 else 0 # Dummy NEAT rule
    gpt_pred = int(gpt4_predict(q['input']) == q['label']) # Placeholder
    correct_neat += int(neat_pred == q['label'])
    correct_gpt += gpt_pred
print(f"NEAT accuracy: {correct_neat/len(test_questions):.2f}")
print(f"GPT-4 (mock) accuracy: {correct_gpt/len(test_questions):.2f}")
