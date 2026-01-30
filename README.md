PROSIT 1: Building and Adapting Language Models
**Ankora AI Research Lab | Graduate NLP Internship Deliverable**

This repository contains the dual-task implementation for **Ankora AI**, focusing on the development of specialized language models. The project bridges the gap between traditional statistical methods for low-resource settings and modern parameter-efficient fine-tuning (PEFT) for domain-specific expertise.

---

## 🚀 Project Components

### 🌍 Task 1: Low-Resource African Language Model (Section B)
Development of a specialized language model for an African language with limited textual data. 
* **Approach:** Statistical N-gram Modeling.
* **Rationale:** N-grams are preferred over neural models in extremely low-resource contexts to avoid overfitting and to capture local linguistic patterns effectively without massive training corpora.
* **Key Metrics:** Perplexity and Smoothing effectiveness.

### 🌾 Task 2: Agricultural Domain Adaptation (Section C)
Fine-tuning an English-based Large Language Model (LLM) to act as a domain expert for the agriculture sector.
* **Base Model:** `microsoft/phi-2` (2.7B).
* **Dataset:** `AI4Agr/CROP-dataset` (~210k agricultural instruction-response pairs).
* **Technique:** **QLoRA (4-bit Quantization)**. This allowed for high-performance adaptation on consumer-grade hardware by training low-rank adapters while keeping the base model weights frozen.
