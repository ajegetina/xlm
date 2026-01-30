# -*- coding: utf-8 -*-
"""
AgricGPT - Agricultural Domain Instruction Tuning with QLoRA

Fine-tunes Microsoft Phi-2 on the AI4Agr/CROP-dataset for agricultural Q&A.
Based on the PEFT with LoRA template, adapted for instruction tuning.
"""

import torch
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

# Hugging Face Hub settings
HF_MODEL_NAME = "agricgpt-phi2"  # Change this to your desired model name
PUSH_TO_HUB = True  # Set to False to skip pushing

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
MAX_SEQ_LENGTH = 512

# ==============================================================================
# Setup
# ==============================================================================

# Check for GPU
if not torch.cuda.is_available():
    raise ValueError("GPU required for training. Please enable CUDA.")

# Set seeds for reproducibility
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
# Load and Prepare Dataset
# ==============================================================================

print("Loading AI4Agr/CROP-dataset...")
dataset = load_dataset(
    "AI4Agr/CROP-dataset",
    data_files="**/*_en/**/*.json",
    split="train"
)

if DATASET_SIZE:
    dataset = dataset.select(range(min(DATASET_SIZE, len(dataset))))
    print(f"Using {len(dataset)} samples (pilot run)")

# Format as instruction-response pairs with EOS token
def format_instruction(sample):
    """Format sample as instruction-response with EOS token for clean stopping."""
    prompt = (
        f"### Instruction:\n{sample['instruction']}\n\n"
        f"### Response:\n{sample['output']}{tokenizer.eos_token}"
    )
    return {"text": prompt}

dataset = dataset.map(format_instruction)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length"
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)

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

# Print trainable parameters
def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

print_trainable_parameters(model)

# ==============================================================================
# Training
# ==============================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=LOGGING_STEPS,
    fp16=True,
    optim="paged_adamw_32bit",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    report_to="none",
    seed=42
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print("\n" + "=" * 60)
print("Starting Agricultural Domain Adaptation Training...")
print("=" * 60 + "\n")

trainer.train()

# ==============================================================================
# Inference Helper
# ==============================================================================

def ask_agrigpt(question: str, max_new_tokens: int = 256) -> str:
    """
    Ask the fine-tuned AgricGPT model a question.
    
    Args:
        question: The agricultural question to ask
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        The model's response
    """
    logging.set_verbosity(logging.CRITICAL)
    model.eval()
    
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Format prompt to match training format
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    
    result = pipe(prompt, generation_config=generation_config)
    full_text = result[0]['generated_text']
    
    # Extract just the response
    response = full_text.split("### Response:\n")[-1]
    # Stop if another instruction starts
    response = response.split("### Instruction:")[0].strip()
    
    return response

# ==============================================================================
# Test the Model
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing AgricGPT...")
    print("=" * 60 + "\n")
    
    test_questions = [
        "What is crop rotation?",
        "How can I prevent soil erosion on my farm?",
        "What are the benefits of organic farming?"
    ]
    
    for q in test_questions:
        print(f"Q: {q}")
        print(f"A: {ask_agrigpt(q)}")
        print("-" * 40)
    
    # Save the model locally
    print(f"\nSaving model to {OUTPUT_DIR}/final_model...")
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    
    # Push to Hugging Face Hub
    if PUSH_TO_HUB:
        from huggingface_hub import login
        print("\nLogging in to Hugging Face Hub...")
        login()  # Will prompt for token or use cached credentials
        
        print(f"Pushing model to Hugging Face as '{HF_MODEL_NAME}'...")
        model.push_to_hub(HF_MODEL_NAME)
        tokenizer.push_to_hub(HF_MODEL_NAME)
        print(f"Model available at: https://huggingface.co/YOUR_USERNAME/{HF_MODEL_NAME}")
    
    print("Done!")
