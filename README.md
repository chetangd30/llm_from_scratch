# LLM From Scratch

A collection of hands-on experiments and notebooks documenting my journey of learning Large Language Models (LLMs), from tokenization and embeddings to recommendation systems, attention mechanisms, transformers, fine-tuning, and LLM applications.

## Objectives

- Understand how modern LLMs work internally
- Learn tokenization and embeddings
- Explore transformer architectures
- Build LLM components from scratch
- Work with open-source language models
- Apply embedding techniques to real-world problems
- Develop practical AI Engineering skills

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Google Colab
- NumPy
- Pandas
- Scikit-Learn
- Sentence Transformers
- Gensim (Word2Vec)

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

The meaning of a word is influenced by its surrounding context, allowing the same word to have different vector representations in different sentences.

### Applications

- Semantic Search
- Question Answering
- Text Classification
- Recommendation Systems
- Retrieval-Augmented Generation (RAG)

---

## 5. Song Recommendation System Using Word2Vec Embeddings

**File:** `05_song_recommendation_word2vec.ipynb`

### Dataset

- Playlist dataset containing user-created music playlists
- Song metadata including song titles and artists

### Topics Covered

- Word2Vec embeddings
- Representation learning
- Recommendation systems
- Similarity search
- Collaborative filtering concepts
- Embedding-based recommendations

### Project Overview

This notebook treats songs as words and playlists as sentences.

Using Word2Vec, songs that frequently appear together in playlists learn similar vector representations.

### Model Training

```python
model = Word2Vec(
    playlists,
    vector_size=32,
    window=20,
    negative=50,
    min_count=1
)
```

### Key Idea

Traditional NLP:

```text
Sentence
↓
Words
↓
Word Embeddings
```

Music Recommendation:

```text
Playlist
↓
Songs
↓
Song Embeddings
```

If two songs frequently occur together in playlists, their vectors become similar in embedding space.

### Example Recommendation

Input Song:

```text
Fade To Black — Metallica
```

Recommended Songs:

```text
Run To The Hills — Iron Maiden
Red Barchetta — Rush
Unchained — Van Halen
November Rain — Guns N' Roses
Rainbow In The Dark — Dio
```

### Another Example

Input Song:

```text
California Love — 2Pac
```

Recommended Songs:

```text
How We Do — The Game
If I Ruled The World — Nas
Sweet Dreams — Beyonce
Hate It Or Love It — The Game
Heartless — Kanye West
```

### Key Learning

- Embeddings are not limited to words
- Any object appearing in a sequence can be embedded
- Similar items naturally cluster together in vector space
- Recommendation systems often rely on embedding similarity

### Skills Demonstrated

- Word2Vec
- Embedding Learning
- Recommendation Systems
- Similarity Search
- Collaborative Filtering Concepts
- Gensim
- Data Processing with Pandas

---

# Repository Structure

```text
llm-from-scratch/
│
├── 01_phi3_text_generation.ipynb
├── 02_tokenization_and_generation.ipynb
├── 03_comparing_llm_tokenizers.ipynb
├── 04_contextualized_word_embeddings.ipynb
├── 05_song_recommendation_word2vec.ipynb
├── README.md
└── requirements.txt
```

---

# Skills Demonstrated

- Large Language Models (LLMs)
- Natural Language Processing (NLP)
- Tokenization
- Contextualized Embeddings
- Word Embeddings
- Transformer Encoders
- Representation Learning
- Recommendation Systems
- Similarity Search
- Word2Vec
- Gensim
- Prompt Engineering
- Hugging Face Transformers
- GPU Inference
- Python Development
- Model Exploration and Analysis

---

# Upcoming Topics

- Semantic Similarity
- Sentence Embeddings
- Vector Databases
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
✅ Word2Vec Embeddings  
✅ Recommendation Systems Using Embeddings  


# References

- Building a Large Language Model (From Scratch) — Sebastian Raschka
- Hugging Face Transformers Documentation
- Microsoft Phi-3 Documentation
- Microsoft DeBERTa Documentation
- Gensim Documentation
- Word2Vec Research Paper (Mikolov et al.)
- OpenAI tiktoken Documentation
- Google FLAN-T5 Documentation
