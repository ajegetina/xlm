# -*- coding: utf-8 -*-
"""
AgricGPT - Agricultural Domain Instruction Tuning with QLoRA

Fine-tunes Microsoft Phi-2 on the AI4Agr/CROP-dataset for agricultural Q&A.
Evaluation using official CROP-benchmark with accuracy metrics.
"""

import torch
import math
import json
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline,
    logging
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from collections import defaultdict

# ==============================================================================
# Configuration
# ==============================================================================

MODEL_NAME = "microsoft/phi-2"
OUTPUT_DIR = "./agri_model_results"
DATASET_SIZE = 5000  # Training data size (set to None for full dataset)
VALIDATION_SPLIT = 0.1

# Hugging Face Hub settings
HF_MODEL_NAME = "agricgpt-phi2"
PUSH_TO_HUB = True
SAVE_STEPS = 100

# LoRA hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["fc1", "fc2", "q_proj", "k_proj", "v_proj", "dense"]

# Training hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
LOGGING_STEPS = 10
EVAL_STEPS = 50
MAX_SEQ_LENGTH = 512

# Benchmark settings
BENCHMARK_SAMPLE_SIZE = 500  # Number of benchmark questions to evaluate (None for all)

# ==============================================================================
# Setup
# ==============================================================================

if not torch.cuda.is_available():
    raise ValueError("GPU required for training. Please enable CUDA.")

if PUSH_TO_HUB:
    from huggingface_hub import login
    print("Logging in to Hugging Face Hub...")
    login()

torch.manual_seed(42)
torch.cuda.manual_seed(42)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# ==============================================================================
# Load Model and Tokenizer
# ==============================================================================

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map={"": 0}
)
model.config.use_cache = False

# ==============================================================================
# Load CROP-benchmark (English subset)
# ==============================================================================

print("\nLoading CROP-benchmark for evaluation...")
benchmark = load_dataset("AI4Agr/CROP-benchmark", split="train")

# Filter for English questions (check if Question contains mostly ASCII/English)
def is_english(text):
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) > 0.7

english_benchmark = benchmark.filter(lambda x: is_english(x.get("Question", "")))
print(f"English benchmark questions: {len(english_benchmark)}")

if BENCHMARK_SAMPLE_SIZE and len(english_benchmark) > BENCHMARK_SAMPLE_SIZE:
    english_benchmark = english_benchmark.shuffle(seed=42).select(range(BENCHMARK_SAMPLE_SIZE))
    print(f"Using {BENCHMARK_SAMPLE_SIZE} questions for evaluation")

# ==============================================================================
# Multiple Choice Evaluation Functions
# ==============================================================================

def format_mcq_prompt(question, options):
    """Format a multiple choice question as a prompt."""
    prompt = f"""### Instruction:
Answer the following agricultural question by selecting the correct option (A, B, C, or D).

Question: {question}

A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}

Reply with only the letter of the correct answer.

### Response:
"""
    return prompt

def extract_answer(response):
    """Extract the answer letter from model response."""
    response = response.strip().upper()
    # Try to find A, B, C, or D in the response
    match = re.search(r'\b([ABCD])\b', response)
    if match:
        return match.group(1)
    # Check if response starts with a letter
    if response and response[0] in 'ABCD':
        return response[0]
    return None

def evaluate_mcq(pipe, gen_config, benchmark_data):
    """Evaluate model on multiple choice questions."""
    correct = 0
    total = 0
    results_by_level = defaultdict(lambda: {"correct": 0, "total": 0})
    detailed_results = []
    
    for item in benchmark_data:
        question = item.get("Question", "")
        options = {
            "A": item.get("Option A", ""),
            "B": item.get("Option B", ""),
            "C": item.get("Option C", ""),
            "D": item.get("Option D", "")
        }
        correct_answer = item.get("Answer", "").strip().upper()
        level = item.get("Level", "Unknown")
        
        if not question or not correct_answer:
            continue
        
        # Generate response
        prompt = format_mcq_prompt(question, options)
        torch.manual_seed(42)
        result = pipe(prompt, generation_config=gen_config)
        response = result[0]['generated_text'].split("### Response:")[-1].strip()
        predicted = extract_answer(response)
        
        is_correct = predicted == correct_answer
        if is_correct:
            correct += 1
            results_by_level[level]["correct"] += 1
        
        total += 1
        results_by_level[level]["total"] += 1
        
        detailed_results.append({
            "question": question[:100],
            "correct_answer": correct_answer,
            "predicted": predicted,
            "is_correct": is_correct,
            "level": level
        })
    
    accuracy = correct / total if total > 0 else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "by_level": dict(results_by_level),
        "detailed": detailed_results[:20]  # Keep first 20 for review
    }

# ==============================================================================
# Base Model Benchmark Evaluation
# ==============================================================================

print("\n" + "=" * 60)
print("BASE MODEL EVALUATION (before training)")
print("=" * 60)

base_pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
gen_config = GenerationConfig(
    max_new_tokens=10,  # Short for MCQ
    do_sample=False,    # Deterministic for evaluation
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)

print("Evaluating base model on CROP-benchmark...")
base_results = evaluate_mcq(base_pipe, gen_config, english_benchmark)
print(f"Base Model Accuracy: {base_results['accuracy']:.2%} ({base_results['correct']}/{base_results['total']})")

# ==============================================================================
# Load and Prepare Training Dataset
# ==============================================================================

print("\n\nLoading AI4Agr/CROP-dataset for training...")
dataset = load_dataset(
    "AI4Agr/CROP-dataset",
    data_files="**/*_en/**/*.json",
    split="train"
)

if DATASET_SIZE:
    dataset = dataset.select(range(min(DATASET_SIZE, len(dataset))))

def format_instruction(sample):
    prompt = (
        f"### Instruction:\n{sample['instruction']}\n\n"
        f"### Response:\n{sample['output']}{tokenizer.eos_token}"
    )
    return {"text": prompt}

dataset = dataset.map(format_instruction)
dataset = dataset.train_test_split(test_size=VALIDATION_SPLIT, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length")

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

# ==============================================================================
# Configure LoRA
# ==============================================================================

print("\nConfiguring LoRA adapters...")
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total_params:,} ({100 * trainable / total_params:.2f}%)")

# ==============================================================================
# Training
# ==============================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=LOGGING_STEPS,
    fp16=True,
    optim="paged_adamw_32bit",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    push_to_hub=PUSH_TO_HUB,
    hub_model_id=HF_MODEL_NAME if PUSH_TO_HUB else None,
    hub_strategy="every_save",
    report_to="none",
    seed=42
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print("\nStarting training...")
trainer.train()

# ==============================================================================
# Fine-Tuned Model Benchmark Evaluation
# ==============================================================================

print("\n" + "=" * 60)
print("FINE-TUNED MODEL EVALUATION (after training)")
print("=" * 60)

logging.set_verbosity(logging.CRITICAL)
model.eval()

ft_pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

print("Evaluating fine-tuned model on CROP-benchmark...")
ft_results = evaluate_mcq(ft_pipe, gen_config, english_benchmark)
print(f"Fine-tuned Model Accuracy: {ft_results['accuracy']:.2%} ({ft_results['correct']}/{ft_results['total']})")

# ==============================================================================
# Perplexity Calculation
# ==============================================================================

def calculate_perplexity(model, tokenizer, texts, max_length=512):
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for text in texts[:100]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    return math.exp(total_loss / total_tokens)

eval_texts = [sample["text"] for sample in eval_dataset]
perplexity = calculate_perplexity(model, tokenizer, eval_texts)

# ==============================================================================
# Evaluation Summary
# ==============================================================================

history = trainer.state.log_history
train_losses = [(h['step'], h['loss']) for h in history if 'loss' in h and 'eval_loss' not in h]
eval_losses = [(h['step'], h['eval_loss']) for h in history if 'eval_loss' in h]

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(f"\n📊 Training Data: {len(tokenized_train)} train, {len(tokenized_eval)} validation")
print(f"\n📉 Training Loss: {train_losses[0][1]:.4f} → {train_losses[-1][1]:.4f}")
print(f"📈 Validation Loss: {eval_losses[0][1]:.4f} → {min(e[1] for e in eval_losses):.4f} (best)")
print(f"\n🎯 Perplexity: {perplexity:.2f}")
print(f"\n📋 CROP-benchmark Evaluation ({ft_results['total']} questions):")
print(f"   Base Model Accuracy:       {base_results['accuracy']:.2%}")
print(f"   Fine-tuned Model Accuracy: {ft_results['accuracy']:.2%}")
print(f"   Improvement:               +{(ft_results['accuracy'] - base_results['accuracy'])*100:.1f}%")

# Accuracy by level
print(f"\n📊 Accuracy by Difficulty Level:")
for level, stats in sorted(ft_results['by_level'].items()):
    level_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    print(f"   Level {level}: {level_acc:.2%} ({stats['correct']}/{stats['total']})")

# ==============================================================================
# Save Results
# ==============================================================================

results = {
    "benchmark": "CROP-benchmark",
    "benchmark_questions": ft_results['total'],
    "base_model_accuracy": base_results['accuracy'],
    "finetuned_accuracy": ft_results['accuracy'],
    "accuracy_improvement": ft_results['accuracy'] - base_results['accuracy'],
    "accuracy_by_level": {k: v['correct']/v['total'] if v['total']>0 else 0 for k,v in ft_results['by_level'].items()},
    "perplexity": perplexity,
    "train_samples": len(tokenized_train),
    "eval_samples": len(tokenized_eval),
    "final_train_loss": train_losses[-1][1] if train_losses else None,
    "best_eval_loss": min(e[1] for e in eval_losses) if eval_losses else None,
    "sample_predictions": ft_results['detailed']
}

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(f"{OUTPUT_DIR}/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}/evaluation_results.json")

# ==============================================================================
# Create Model Card and Push to HuggingFace
# ==============================================================================

if PUSH_TO_HUB:
    from huggingface_hub import HfApi
    
    metrics = {
        "accuracy": round(ft_results['accuracy'] * 100, 2),
        "perplexity": round(perplexity, 2),
        "eval_loss": results["best_eval_loss"],
    }
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    model_card = f"""---
language:
- en
license: apache-2.0
tags:
- agriculture
- phi-2
- qlora
- instruction-tuning
- crop-science
datasets:
- AI4Agr/CROP-dataset
base_model: microsoft/phi-2
model-index:
- name: AgricGPT-Phi2
  results:
  - task:
      type: question-answering
      name: Agricultural MCQ
    dataset:
      name: CROP-benchmark
      type: AI4Agr/CROP-benchmark
    metrics:
    - type: accuracy
      value: {ft_results['accuracy']*100:.1f}
      name: Accuracy
    - type: perplexity
      value: {perplexity:.2f}
      name: Perplexity
---

# AgricGPT - Agricultural Domain Language Model

Fine-tuned **Microsoft Phi-2** for agricultural question answering using QLoRA.

## 🎯 Benchmark Results (CROP-benchmark)

| Model | Accuracy | Improvement |
|-------|----------|-------------|
| Base Phi-2 | {base_results['accuracy']*100:.1f}% | - |
| **AgricGPT (ours)** | **{ft_results['accuracy']*100:.1f}%** | **+{(ft_results['accuracy']-base_results['accuracy'])*100:.1f}%** |

### Accuracy by Difficulty Level

| Level | Accuracy | Questions |
|-------|----------|-----------|
""" + "\n".join([f"| {level} | {stats['correct']/stats['total']*100:.1f}% | {stats['total']} |" for level, stats in sorted(ft_results['by_level'].items())]) + f"""

## 📊 Training Metrics

| Metric | Value |
|--------|-------|
| Perplexity | {perplexity:.2f} |
| Validation Loss | {results['best_eval_loss']:.4f} |
| Training Samples | {results['train_samples']:,} |

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", quantization_config=bnb_config, device_map={{"": 0}}, trust_remote_code=True)
model = PeftModel.from_pretrained(base, "{HF_MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

prompt = "### Instruction:\\nWhat is crop rotation?\\n\\n### Response:\\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Citation

```bibtex
@inproceedings{{zhangempowering,
  title={{Empowering and Assessing the Utility of Large Language Models in Crop Science}},
  author={{Zhang, Hang and Sun, Jiawei and Chen, Renqi and others}},
  booktitle={{NeurIPS Datasets and Benchmarks Track}}
}}
```
"""
    
    with open(f"{OUTPUT_DIR}/README.md", "w") as f:
        f.write(model_card)
    
    print(f"\nPushing model with benchmark results to {HF_MODEL_NAME}...")
    trainer.push_to_hub()
    
    api = HfApi()
    username = api.whoami()['name']
    api.upload_file(
        path_or_fileobj=f"{OUTPUT_DIR}/README.md",
        path_in_repo="README.md",
        repo_id=f"{username}/{HF_MODEL_NAME}",
        repo_type="model"
    )
    
    print(f"\n✅ Done! View at: https://huggingface.co/{username}/{HF_MODEL_NAME}")
    print("Benchmark accuracy is now visible on your model card!")
