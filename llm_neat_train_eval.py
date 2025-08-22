import os
import subprocess
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
import google.generativeai as genai
import json
from tqdm import tqdm

# Setup API KEYS
openai.api_key = "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"
genai.configure(api_key="AIzaSyBw1TspVygKWSpFMfFwRNNCWuxKhAJRWtA")

# 1. Download a dataset from Kaggle (MMLU as an example)
def download_kaggle_dataset():
    if not os.path.exists('mmlu'):
        os.makedirs('mmlu', exist_ok=True)
        subprocess.run([
            "kaggle", "datasets", "download", "-d", "lukaemon/mmlu", "-p", "mmlu", "--unzip"
        ])
    print("Kaggle dataset downloaded and extracted.")

# 2. Load dataset
def load_questions(max_samples=20):
    # Uses HuggingFace datasets (optional: replace with CSV/other Kaggle parsing)
    ds = load_dataset("lukaemon/mmlu", "all", split="test")
    questions = []
    for ex in ds.select(range(min(max_samples, len(ds)))):
        questions.append({
            "question": ex["question"],
            "choices": ex["choices"],
            "answer": ex["answer"]
        })
    return questions

# 3. NEAT placeholder for LLM (replace with your NEAT integration)
def neat_predict_question(model, tokenizer, question):
    prompt = f"{question['question']}\nOptions: {', '.join(question['choices'])}\nAnswer:"
    inp = tokenizer(prompt, return_tensors="pt")
    out = model.generate(**inp, max_new_tokens=8)
    prediction = tokenizer.decode(out[0], skip_special_tokens=True)
    return prediction.split("Answer:")[-1].strip()

# 4. OpenAI and Gemini API queries
def gpt4_predict_question(prompt):
    messages = [{"role":"user","content":prompt}]
    completion = openai.ChatCompletion.create(model="gpt-4", messages=messages, temperature=0)
    return completion["choices"]["message"]["content"].split("Answer:")[-1].strip()

def gemini_predict_question(prompt):
    gmodel = genai.GenerativeModel("gemini-pro")
    completion = gmodel.generate_content(prompt)
    return completion.text.split("Answer:")[-1].strip()

def evaluate_and_save_all(questions, neat_model, neat_tokenizer):
    results = []
    for q in tqdm(questions):
        prompt = f"{q['question']}\nOptions: {', '.join(q['choices'])}\nAnswer:"
        neat_ans = neat_predict_question(neat_model, neat_tokenizer, q)
        gpt_ans = gpt4_predict_question(prompt)
        gemini_ans = gemini_predict_question(prompt)
        results.append({
            "question": q['question'],
            "choices": q['choices'],
            "answer": q['answer'],
            "NEAT": neat_ans,
            "GPT-4": gpt_ans,
            "Gemini": gemini_ans,
        })
    with open("results/llm_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to results/llm_benchmark_results.json")

if __name__ == "__main__":
    # 1. Download and load dataset
    download_kaggle_dataset()
    questions = load_questions(max_samples=20)
    # 2. Load LLM (use GPT-2 as NEAT backbone demo, replace with your NEAT-wrapped model)
    neat_model_name = "gpt2"
    neat_tokenizer = AutoTokenizer.from_pretrained(neat_model_name)
    neat_model = AutoModelForCausalLM.from_pretrained(neat_model_name)
    # TIP: Fine-tune neat_model with your NEATTrainer here!
    # 3. Evaluate and save results
    evaluate_and_save_all(questions, neat_model, neat_tokenizer)
