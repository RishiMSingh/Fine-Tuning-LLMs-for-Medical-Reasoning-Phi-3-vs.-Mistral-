# Fine-Tuning-LLMs-for-Medical-Reasoning-Phi-3-vs.-Mistral-

This repository contains two Jupyter notebooks where we fine-tune large language models (LLMs) for **medical reasoning and question answering**. The goal is to teach models how to provide clinically sound, step-by-step answers to complex medical questions.

We compare:
- ✅ `microsoft/phi-3-mini-4k-instruct` — lightweight, accurate, and fast
- 🧪 `mistralai/Mistral-7B-Instruct-v0.2` — high-capacity baseline

---

## 📁 Notebooks

| File | Description |
|------|-------------|
| `SFT_phi.ipynb` | Fine-tuning Phi-3 on medical reasoning dataset using LoRA |
| `SFT_mistral.ipynb` | Fine-tuning Mistral 7B on the same dataset for comparison |

---

## 📚 Dataset

We use the [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) dataset from Hugging Face.

Each example contains:
- An instruction
- A medical question
- A thought-through response explaining the reasoning behind the answer

Example format:
```json
{
  "instruction": "Please answer the following medical question.",
  "input": "A 45-year-old man presents with crushing chest pain. What is the next best step?",
  "output": "<think> Evaluate ECG and troponins to rule out myocardial infarction. </think> The next best step is to order an ECG and cardiac enzymes."
}

# 🧪 Key Findings
Aspect	Phi-3 Mini (microsoft/phi-3-mini-4k-instruct)	Mistral 7B Instruct (mistralai/Mistral-7B-Instruct-v0.2)
Training Speed	✅ Fast (lightweight, quick convergence)	🟡 Slower, higher compute requirement
Clinical Reasoning Quality	✅ More structured and accurate	🟡 Sometimes vague or overgeneralized
Answer Hallucinations	✅ Fewer hallucinations	🔴 Occasionally fabricated or medically incorrect details
Memory & Compute Needs	✅ Minimal (runs on consumer GPU with LoRA)	🔴 Requires more memory and compute
Instruction Following	✅ Strong alignment with input prompts	🟡 Sometimes deviated or too verbose

## 🚀 How to Run

git clone https://github.com/your-username/medical-llm-finetuning.git
Use Jupyter Notebook preferably Google Colab - Make sure you're running this on a GPU environment (e.g., Google Colab with T4/A100 or a local machine with CUDA support).

📌 Future Work

🧪 Evaluate against benchmarks like MedQA or PubMedQA
🖼️ Add a Streamlit chatbot interface for interactive testing

📚 Credits
Model Architectures:
microsoft/phi-3-mini-4k-instruct – Lightweight and instruction-tuned LLM by Microsoft
mistralai/Mistral-7B-Instruct-v0.2 – High-performance open-weight LLM by Mistral AI
Dataset:
FreedomIntelligence/medical-o1-reasoning-SFT – Open medical reasoning dataset with chain-of-thought answers
Libraries & Tools:
🤗 Transformers
🤗 PEFT
⚡ Accelerate
🧮 PyTorch
Tutorial:
- https://www.datacamp.com/tutorial/fine-tuning-qwen3
