# LLM From Scratch

A collection of hands-on experiments and notebooks documenting my journey of learning Large Language Models (LLMs), from tokenization and inference to embeddings, attention mechanisms, transformers, fine-tuning, and LLM applications.

## Objectives

- Understand how modern LLMs work internally
- Learn tokenization and embeddings
- Explore transformer architectures
- Build LLM components from scratch
- Work with open-source language models
- Develop practical AI Engineering skills

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Google Colab
- NumPy
- Scikit-Learn
- Sentence Transformers

---

# Completed Notebooks

## 1. Phi-3 Text Generation

**File:** `01_phi3_text_generation.ipynb`

### Topics Covered

- Loading Microsoft Phi-3 Mini 4K Instruct
- Running inference on GPU
- Prompt-based text generation
- Hugging Face Pipeline API
- Open-source LLM deployment

### Key Learning

- How to load and run an open-source LLM
- GPU-based inference workflow
- Generating text from user prompts

---

## 2. Tokenization and Text Generation

**File:** `02_tokenization_and_generation.ipynb`

### Topics Covered

- Tokenization
- Token IDs
- Encoding text into tokens
- Decoding token IDs back into text
- Subword tokenization
- Text generation using `model.generate()`

### Key Learning

- How LLMs convert text into numerical representations
- Relationship between tokens and generated output
- Understanding model inputs and outputs

### Example

```text
Subject:
```

can be represented as:

```text
3323 -> Sub
622  -> ject
29901 -> :
```

---

## 3. Comparing Trained LLM Tokenizers

**File:** `03_comparing_llm_tokenizers.ipynb`

### Models Compared

- BERT Base Uncased
- BERT Base Cased
- GPT-2
- FLAN-T5 Small
- GPT-4 Tokenizer (tiktoken equivalent)
- StarCoder2
- Galactica
- Phi-3 Mini 4K Instruct

### Topics Covered

- Tokenizer architectures
- WordPiece tokenization
- Byte Pair Encoding (BPE)
- SentencePiece tokenization
- Case-sensitive vs case-insensitive tokenizers
- Handling multilingual text
- Tokenization of code and symbols

### Key Learning

Different LLMs tokenize the same text differently depending on:

- Training objectives
- Vocabulary size
- Tokenization algorithm
- Target domain (text, code, multilingual)

### Sample Observation

The word:

```text
CAPITALIZATION
```

is tokenized differently across models:

**BERT Uncased**

```text
capital ##ization
```

**GPT-2**

```text
CAP ITAL IZ ATION
```

**Phi-3**

```text
C AP IT AL IZ ATION
```

This demonstrates how tokenizer design affects vocabulary efficiency and model behavior.

---

## 4. Contextualized Word Embeddings with DeBERTa

**File:** `04_contextualized_word_embeddings.ipynb`

### Models Used

- Microsoft DeBERTa Base Tokenizer
- Microsoft DeBERTa-v3-XSmall

### Topics Covered

- Contextualized word embeddings
- Transformer encoder models
- Hidden states
- Token representations
- Input tokenization
- Embedding dimensions
- Understanding transformer outputs

### Key Learning

Unlike traditional word embeddings such as Word2Vec, GloVe, and FastText, transformer-based models generate contextualized embeddings.

This means the representation of a word depends on the surrounding words in the sentence.

For example:

```text
I deposited money in the bank.
```

and

```text
The fisherman sat on the river bank.
```

produce different vector representations for the word:

```text
bank
```

### Implementation Steps

1. Load the tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/deberta-base"
)
```

2. Load the pretrained transformer model

```python
model = AutoModel.from_pretrained(
    "microsoft/deberta-v3-xsmall"
)
```

3. Tokenize input text

```python
tokens = tokenizer(
    "Hello world",
    return_tensors="pt"
)
```

4. Generate contextual embeddings

```python
output = model(**tokens)[0]
```

### Understanding the Output

Input tokens:

```text
[CLS]
Hello
world
[SEP]
```

Output shape:

```python
torch.Size([1, 4, 384])
```

Meaning:

```text
1     -> Batch Size
4     -> Number of Tokens
384   -> Embedding Dimensions
```

Each token is represented by a unique 384-dimensional vector.

### Applications

Contextual embeddings are widely used in:

- Semantic Search
- Text Classification
- Question Answering
- Named Entity Recognition (NER)
- Recommendation Systems
- Retrieval-Augmented Generation (RAG)
- Modern LLM Applications

### Skills Demonstrated

- Transformer Encoders
- Contextual Embeddings
- Representation Learning
- NLP Feature Extraction
- Hugging Face Models

---

# Repository Structure

```text
llm-from-scratch/
│
├── 01_phi3_text_generation.ipynb
├── 02_tokenization_and_generation.ipynb
├── 03_comparing_llm_tokenizers.ipynb
├── 04_contextualized_word_embeddings.ipynb
├── README.md
└── requirements.txt
```

---

# Skills Demonstrated

- Large Language Models (LLMs)
- Natural Language Processing (NLP)
- Tokenization
- Contextualized Embeddings
- Transformer Encoders
- Representation Learning
- Prompt Engineering
- Hugging Face Transformers
- GPU Inference
- Python Development
- Model Exploration and Analysis

---

# Upcoming Topics

- Semantic Similarity
- Vector Search
- Attention Mechanism
- Self-Attention
- Multi-Head Attention
- Transformer Architecture
- Positional Encodings
- Fine-Tuning
- LoRA
- Retrieval-Augmented Generation (RAG)
- Building a GPT Model from Scratch

---

# Learning Progress

✅ Running Open-Source LLMs  
✅ Tokenization Fundamentals  
✅ Token ID Analysis  
✅ Comparing LLM Tokenizers  
✅ Contextualized Word Embeddings  

# References

- Building a Large Language Model (From Scratch) — Sebastian Raschka
- Hugging Face Transformers Documentation
- Microsoft Phi-3 Documentation
- Microsoft DeBERTa Documentation
- OpenAI tiktoken Documentation
- Google FLAN-T5 Documentation
