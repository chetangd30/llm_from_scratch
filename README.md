# LLM From Scratch

A collection of notebooks and experiments documenting my journey of learning Large Language Models (LLMs), from inference and tokenization to embeddings, attention mechanisms, transformers, fine-tuning, and LLM applications.

## Objectives

- Understand how modern LLMs work internally
- Learn tokenization and embeddings
- Explore transformer architectures
- Build LLM components from scratch
- Work with open-source models such as Phi-3
- Develop practical AI Engineering skills

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Google Colab
- NumPy
- Scikit-learn

## Completed Notebooks

### 1. Phi-3 Text Generation

**File:** `01_phi3_text_generation.ipynb`

Topics covered:

- Loading Microsoft Phi-3 Mini 4K Instruct
- Running inference on GPU
- Using Hugging Face Transformers
- Text generation with `pipeline`
- Prompt-based generation

Key learning:

- How to load and run an open-source LLM
- Basic inference workflow

---

### 2. Tokenization and Text Generation

**File:** `02_tokenization_and_generation.ipynb`

Topics covered:

- Tokenization
- Token IDs
- Encoding text
- Decoding tokens
- Understanding subword tokenization
- Generating text using `model.generate()`

Key learning:

- How LLMs convert text into tokens
- How token IDs are processed by the model
- Relationship between prompts and generated output

## Repository Structure

```text
llm-from-scratch/
│
├── 01_phi3_text_generation.ipynb
├── 02_tokenization_and_generation.ipynb
├── README.md
└── requirements.txt
```

## Upcoming Topics

- Embeddings
- Semantic Similarity
- Attention Mechanism
- Self-Attention
- Transformer Architecture
- Fine-Tuning
- LoRA
- Retrieval-Augmented Generation (RAG)
- Building an LLM from Scratch

## References

- Building a Large Language Model (From Scratch) — Sebastian Raschka
- Hugging Face Documentation
- Microsoft Phi-3 Documentation
