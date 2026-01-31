# -*- coding: utf-8 -*-
"""
AgricGPT - Agricultural Domain Instruction Tuning with QLoRA

Fine-tunes Microsoft Phi-2 on the AI4Agr/CROP-dataset for agricultural Q&A.
Includes proper evaluation: train/validation split, perplexity, before/after comparison.
"""

import torch
import math
import json
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

# ==============================================================================
# Configuration
# ==============================================================================

MODEL_NAME = "microsoft/phi-2"
OUTPUT_DIR = "./agri_model_results"
DATASET_SIZE = 5000  # Pilot run size (set to None for full dataset)
VALIDATION_SPLIT = 0.1  # 10% for validation

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

# 20 Test questions for before/after comparison
TEST_QUESTIONS = [
    # Crop Management
    "What is crop rotation and why is it important?",
    "How do I know when my maize is ready for harvest?",
    "What are cover crops and how do they help?",
    "How should I prepare soil before planting?",
    "What is the best time to plant tomatoes?",
    # Pest & Disease Control
    "How can I control aphids naturally without chemicals?",
    "What causes leaf blight in potatoes?",
    "How do I prevent fungal diseases in my crops?",
    "What are the signs of pest infestation in stored grains?",
    "How can I manage weeds organically?",
    # Soil & Fertilization
    "How can I improve soil fertility naturally?",
    "What is the difference between organic and inorganic fertilizers?",
    "How do I test my soil pH?",
    "What nutrients do plants need most?",
    "How can I prevent soil erosion on my farm?",
    # Irrigation & Water
    "What is drip irrigation and what are its benefits?",
    "How much water do vegetable crops need?",
    "How can I conserve water on my farm?",
    # General Farming
    "What are the benefits of organic farming?",
    "How do I start a small vegetable garden?"
]

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

# 4-bit quantization config (QLoRA)
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
# Helper Functions
# ==============================================================================

def generate_response(pipe, question, gen_config):
    """Generate response for a single question."""
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    result = pipe(prompt, generation_config=gen_config)
    response = result[0]['generated_text'].split("### Response:\n")[-1]
    return response.split("### Instruction:")[0].strip()[:500]

def calculate_perplexity(model, tokenizer, texts, max_length=512):
    """Calculate perplexity on a list of texts."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    return math.exp(total_loss / total_tokens)

# ==============================================================================
# Base Model Evaluation (Before Training)
# ==============================================================================

print("\n" + "=" * 60)
print("BASE MODEL RESPONSES (before training)")
print("=" * 60)

base_pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
gen_config = GenerationConfig(
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)

all_base_responses = []
for i, q in enumerate(TEST_QUESTIONS, 1):
    torch.manual_seed(42)
    response = generate_response(base_pipe, q, gen_config)
    all_base_responses.append(response)
    if i <= 5:  # Show first 5
        print(f"\nQ{i}: {q}")
        print(f"A: {response[:200]}...")

# ==============================================================================
# Load and Prepare Dataset with Train/Validation Split
# ==============================================================================

print("\n\nLoading AI4Agr/CROP-dataset...")
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

# Split into train and validation
dataset = dataset.train_test_split(test_size=VALIDATION_SPLIT, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length"
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

# ==============================================================================
# Configure LoRA
# ==============================================================================

print("Configuring LoRA adapters...")
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
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# ==============================================================================
# Training with Validation Loss Tracking
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
    # Evaluation
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    # Checkpoints
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # HuggingFace Hub
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

print("\nStarting training with validation...")
trainer.train()

# ==============================================================================
# Perplexity Calculation
# ==============================================================================

eval_texts = [sample["text"] for sample in eval_dataset]
perplexity = calculate_perplexity(model, tokenizer, eval_texts[:100])

print("\n" + "=" * 60)
print(f"PERPLEXITY ON VALIDATION SET: {perplexity:.2f}")
print("=" * 60)

# ==============================================================================
# Fine-Tuned Model Evaluation (After Training)
# ==============================================================================

print("\n" + "=" * 60)
print("FINE-TUNED MODEL RESPONSES (after training)")
print("=" * 60)

logging.set_verbosity(logging.CRITICAL)
model.eval()

ft_pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

all_ft_responses = []
for i, q in enumerate(TEST_QUESTIONS, 1):
    torch.manual_seed(42)
    response = generate_response(ft_pipe, q, gen_config)
    all_ft_responses.append(response)
    if i <= 5:  # Show first 5
        print(f"\nQ{i}: {q}")
        print(f"A: {response[:200]}...")

# ==============================================================================
# Side-by-Side Comparison
# ==============================================================================

print("\n" + "=" * 80)
print("BEFORE vs AFTER TRAINING COMPARISON")
print("=" * 80)

for i, (q, base_r, ft_r) in enumerate(zip(TEST_QUESTIONS, all_base_responses, all_ft_responses), 1):
    print(f"\n{'─'*80}")
    print(f"Q{i}: {q}")
    print(f"\n📌 BEFORE: {base_r[:200]}..." if len(base_r) > 200 else f"\n📌 BEFORE: {base_r}")
    print(f"\n✅ AFTER: {ft_r[:200]}..." if len(ft_r) > 200 else f"\n✅ AFTER: {ft_r}")

# ==============================================================================
# Evaluation Summary
# ==============================================================================

history = trainer.state.log_history
train_losses = [(h['step'], h['loss']) for h in history if 'loss' in h and 'eval_loss' not in h]
eval_losses = [(h['step'], h['eval_loss']) for h in history if 'eval_loss' in h]

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(f"\n📊 Dataset:")
print(f"   - Training samples: {len(tokenized_train)}")
print(f"   - Validation samples: {len(tokenized_eval)}")
print(f"\n📉 Training Loss:")
if train_losses:
    print(f"   - Initial: {train_losses[0][1]:.4f}")
    print(f"   - Final: {train_losses[-1][1]:.4f}")
print(f"\n📈 Validation Loss:")
if eval_losses:
    print(f"   - Initial: {eval_losses[0][1]:.4f}")
    print(f"   - Final (best): {min(e[1] for e in eval_losses):.4f}")
print(f"\n🎯 Perplexity: {perplexity:.2f}")
print(f"\n✅ Test Questions: {len(TEST_QUESTIONS)}")

# Save results
results = {
    "perplexity": perplexity,
    "train_samples": len(tokenized_train),
    "eval_samples": len(tokenized_eval),
    "final_train_loss": train_losses[-1][1] if train_losses else None,
    "final_eval_loss": eval_losses[-1][1] if eval_losses else None,
    "best_eval_loss": min(e[1] for e in eval_losses) if eval_losses else None,
    "test_questions": TEST_QUESTIONS,
    "base_responses": all_base_responses,
    "finetuned_responses": all_ft_responses
}

with open(f"{OUTPUT_DIR}/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}/evaluation_results.json")

# ==============================================================================
# Create Model Card and Push Metrics to HuggingFace
# ==============================================================================

if PUSH_TO_HUB:
    from huggingface_hub import HfApi, ModelCard
    
    # Save and log metrics
    metrics = {
        "eval_loss": results["best_eval_loss"],
        "perplexity": round(perplexity, 2),
        "train_samples": results["train_samples"],
        "eval_samples": results["eval_samples"]
    }
    
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    # Create model card content
    model_card_content = f"""---
language:
- en
license: apache-2.0
tags:
- agriculture
- phi-2
- qlora
- instruction-tuning
- peft
datasets:
- AI4Agr/CROP-dataset
base_model: microsoft/phi-2
model-index:
- name: AgricGPT-Phi2
  results:
  - task:
      type: text-generation
      name: Agricultural Q&A
    metrics:
    - type: perplexity
      value: {perplexity:.2f}
      name: Perplexity
    - type: loss
      value: {results['best_eval_loss']:.4f}
      name: Validation Loss
---

# AgricGPT - Agricultural Domain Language Model

Fine-tuned **Microsoft Phi-2** for agricultural question answering using QLoRA.

## Model Description

This model is instruction-tuned on the AI4Agr/CROP-dataset for agricultural domain knowledge.

## Evaluation Results

| Metric | Value |
|--------|-------|
| **Perplexity** | {perplexity:.2f} |
| **Validation Loss** | {results['best_eval_loss']:.4f} |
| **Training Loss** | {results['final_train_loss']:.4f} |
| **Training Samples** | {results['train_samples']:,} |
| **Validation Samples** | {results['eval_samples']:,} |

## Training Details

- **Base Model**: microsoft/phi-2
- **Method**: QLoRA (4-bit quantization + LoRA)
- **LoRA Rank**: {LORA_R}
- **LoRA Alpha**: {LORA_ALPHA}
- **Learning Rate**: {LEARNING_RATE}
- **Epochs**: {NUM_EPOCHS}

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Load base model
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", quantization_config=bnb_config, device_map={{"": 0}}, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# Load fine-tuned adapters
model = PeftModel.from_pretrained(base_model, "{HF_MODEL_NAME}")

# Generate
prompt = "### Instruction:\\nWhat is crop rotation?\\n\\n### Response:\\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Example Outputs

**Q: What is crop rotation?**
> {all_ft_responses[0][:300]}...

**Q: How can I control aphids naturally?**
> {all_ft_responses[5][:300]}...

## License

Apache 2.0
"""
    
    # Save model card
    with open(f"{OUTPUT_DIR}/README.md", "w") as f:
        f.write(model_card_content)
    
    print(f"\nModel card created at {OUTPUT_DIR}/README.md")
    
    # Push everything
    print(f"Pushing final model with metrics to {HF_MODEL_NAME}...")
    trainer.push_to_hub()
    
    # Also push the model card explicitly
    api = HfApi()
    api.upload_file(
        path_or_fileobj=f"{OUTPUT_DIR}/README.md",
        path_in_repo="README.md",
        repo_id=f"{api.whoami()['name']}/{HF_MODEL_NAME}",
        repo_type="model"
    )
    
    print(f"✅ Done! View at: https://huggingface.co/{api.whoami()['name']}/{HF_MODEL_NAME}")
    print("Metrics are now visible on your model card!")
