# LLM From Scratch

A collection of hands-on experiments and notebooks documenting my journey of learning Large Language Models (LLMs), from tokenization and embeddings to transformer architecture, text generation, decoding, inference optimization, fine-tuning, and LLM applications.

## Objectives

- Understand how modern LLMs work internally
- Learn tokenization and embeddings
- Understand Transformer inputs and outputs
- Explore attention mechanisms and Transformer architecture
- Understand how LLMs generate text token by token
- Learn how decoding and probability distributions work
- Understand inference optimization techniques such as KV caching
- Build LLM components from scratch
- Work with open-source language models
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
- Understanding Transformer outputs

### Key Learning

Unlike traditional word embeddings such as Word2Vec, GloVe, and FastText, Transformer-based models generate contextualized embeddings.

This means the representation of a word depends on the surrounding words in the sentence.

For example:

```text
I deposited money in the bank.
```

and:

```text
The fisherman sat on the river bank.
```

produce different contextual representations for the word:

```text
bank
```

### Output Example

For:

```text
Hello world
```

the model produces:

```text
[CLS]
Hello
world
[SEP]
```

with an output shape of:

```text
torch.Size([1, 4, 384])
```

Meaning:

```text
1   -> Batch size
4   -> Number of tokens
384 -> Embedding dimensions
```

### Applications

Contextual embeddings can be used for:

- Semantic Search
- Question Answering
- Text Classification
- Named Entity Recognition
- Recommendation Systems
- Retrieval-Augmented Generation (RAG)

---

## 5. Song Recommendation System Using Word2Vec Embeddings

**File:** `05_song_recommendation_word2vec.ipynb`

### Dataset

- Playlist dataset containing user-created music playlists
- Song metadata containing song titles and artists

### Topics Covered

- Word2Vec embeddings
- Representation learning
- Recommendation systems
- Similarity search
- Collaborative filtering concepts
- Embedding-based recommendations

### Project Overview

This notebook demonstrates how Word2Vec can be applied beyond traditional Natural Language Processing.

Instead of treating words as tokens:

```text
Sentence -> Words
```

songs are treated as tokens:

```text
Playlist -> Songs
```

Songs that frequently occur together in playlists learn similar vector representations.

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

### Example Recommendation

Input:

```text
Fade To Black — Metallica
```

Recommended songs:

```text
Run To The Hills — Iron Maiden
Red Barchetta — Rush
Unchained — Van Halen
November Rain — Guns N' Roses
Rainbow In The Dark — Dio
```

### Another Example

Input:

```text
California Love — 2Pac
```

Recommended songs:

```text
How We Do — The Game
If I Ruled The World — Nas
Sweet Dreams — Beyonce
Hate It Or Love It — The Game
Heartless — Kanye West
```

### Key Learning

- Embeddings are not limited to words
- Sequential data can be represented using embedding techniques
- Items appearing in similar contexts can develop similar vector representations
- Embedding similarity can be used to build recommendation systems

---

## 6. Transformer Inputs, Outputs, Decoding and KV Cache

**File:** `06_transformer_inputs_outputs_and_kv_cache.ipynb`

### Topics Covered

- Loading a pretrained Phi-3 model
- Transformer model architecture
- Transformer inputs and outputs
- Hidden states
- Embedding dimensions
- Language model head (`lm_head`)
- Logits
- Token probability selection
- Greedy decoding
- Autoregressive text generation
- Key-Value (KV) caching
- Generation performance optimization

### Loading the Model

The notebook loads Microsoft's:

```text
microsoft/Phi-3-mini-4k-instruct
```

using Hugging Face Transformers.

The model and tokenizer are loaded separately:

```python
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct"
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto"
)
```

### Understanding the Transformer Output

The Phi-3 model contains:

- Token embeddings
- 32 Transformer decoder layers
- Self-attention layers
- MLP layers
- RMS normalization
- Language model head

The model's hidden representation has a dimension of:

```text
3072
```

For example:

```text
model_output.shape
torch.Size([1, 6, 3072])
```

Meaning:

```text
1    -> Batch size
6    -> Number of input tokens
3072 -> Hidden dimension
```

### Understanding the Language Model Head

The Transformer produces hidden states, but these are not directly token predictions.

The hidden states are passed through the language model head:

```python
lm_head_output = model.lm_head(model_output[0])
```

The resulting shape is:

```text
torch.Size([1, 6, 32064])
```

Meaning:

```text
1     -> Batch size
6     -> Number of tokens
32064 -> Vocabulary size
```

The model produces a score (logit) for every possible token in its vocabulary.

### Choosing the Next Token

For the prompt:

```text
The capital of France is
```

the model produces logits for the next token.

The highest-scoring token is selected:

```python
token_id = lm_head_output[0, -1].argmax(-1)
```

which produces:

```text
Paris
```

This demonstrates **greedy decoding**, where the token with the highest predicted score is selected at each generation step.

### Generation Flow

The process can be represented as:

```text
Input Text
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Token Embeddings
    ↓
Transformer Layers
    ↓
Hidden States
    ↓
Language Model Head
    ↓
Logits
    ↓
Token Selection
    ↓
Next Token
    ↓
Repeat
```

This is the basic autoregressive generation process used by causal language models.

### KV Cache

The notebook also explores **Key-Value (KV) caching**, an important optimization used during autoregressive generation.

Without caching, the model repeatedly recomputes attention information for previously processed tokens.

With caching:

```text
Previous Key/Value states
          ↓
      KV Cache
          ↓
Reuse during next token generation
```

This significantly reduces redundant computation during generation.

### Performance Comparison

For generating 100 new tokens, the notebook measured approximately:

```text
KV Cache Enabled:
~6.66 seconds

KV Cache Disabled:
~21.9 seconds
```

The exact timings depend on GPU availability, system load, and runtime conditions, but the experiment demonstrates the significant performance benefit of caching.

### Key Learning

This notebook provides a deeper understanding of what happens inside a causal Transformer after the tokenizer has converted text into token IDs.

It demonstrates:

- How token IDs enter the Transformer
- How hidden states are produced
- How the `lm_head` converts hidden states into vocabulary logits
- How the next token is selected
- How autoregressive generation works
- Why KV caching makes generation faster

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
├── 06_transformer_inputs_outputs_and_kv_cache.ipynb
├── README.md
└── requirements.txt
```

---

# Skills Demonstrated

- Large Language Models (LLMs)
- Natural Language Processing (NLP)
- Tokenization
- Subword Tokenization
- Contextualized Embeddings
- Word2Vec
- Word Embeddings
- Representation Learning
- Recommendation Systems
- Similarity Search
- Transformer Encoders
- Transformer Decoders
- Self-Attention Concepts
- Hidden States
- Logits
- Language Model Heads
- Greedy Decoding
- Autoregressive Generation
- KV Caching
- Hugging Face Transformers
- PyTorch
- GPU Inference
- Python Development
- Model Exploration and Analysis

---

# Learning Progress

✅ Running Open-Source LLMs  
✅ Tokenization Fundamentals  
✅ Token ID Analysis  
✅ Comparing LLM Tokenizers  
✅ Contextualized Word Embeddings  
✅ Word2Vec Embeddings  
✅ Recommendation Systems Using Embeddings  
✅ Transformer Inputs and Outputs  
✅ Hidden States and Logits  
✅ Greedy Decoding  
✅ Autoregressive Text Generation  
✅ KV Cache Fundamentals  


