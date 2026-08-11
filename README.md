# LLM From Scratch

A collection of hands-on experiments and notebooks documenting my journey of learning Large Language Models (LLMs), from tokenization and embeddings to transformer architecture, text generation, decoding, text classification, inference optimization, fine-tuning, and LLM applications.

## Objectives

- Understand how modern LLMs work internally
- Learn tokenization and embeddings
- Understand Transformer inputs and outputs
- Explore attention mechanisms and Transformer architecture
- Understand how LLMs generate text token by token
- Learn how decoding and probability distributions work
- Apply embeddings to real-world machine learning problems
- Explore text classification using representation and generative models
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
- Hugging Face Datasets

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
- Target domain

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

### Understanding the Transformer Output

The Phi-3 model contains:

- Token embeddings
- Transformer decoder layers
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

The Transformer produces hidden states, which are passed through the language model head:

```python
lm_head_output = model.lm_head(model_output[0])
```

The resulting output has a vocabulary dimension:

```text
32064
```

This produces a score (logit) for every possible token in the vocabulary.

### Greedy Decoding

For the prompt:

```text
The capital of France is
```

the highest-scoring next token is:

```text
Paris
```

The next token can be selected using:

```python
token_id = lm_head_output[0, -1].argmax(-1)
```

This demonstrates greedy decoding, where the token with the highest predicted score is selected at each generation step.

### KV Cache

The notebook also explores Key-Value (KV) caching, an important optimization for autoregressive generation.

With caching, previously calculated key and value states can be reused instead of being repeatedly recomputed.

### Performance Experiment

The notebook compared generation with and without caching:

```text
KV Cache Enabled:
~6.66 seconds

KV Cache Disabled:
~21.9 seconds
```

The exact timing depends on the GPU and runtime environment.

### Key Learning

- How token IDs enter the Transformer
- How hidden states are produced
- How the `lm_head` converts hidden states into vocabulary logits
- How the next token is selected
- How autoregressive generation works
- Why KV caching improves generation efficiency

---

## 7. Text Classification with Representation and Generative Models

**File:** `07_text_classification.ipynb`

### Dataset

The notebook uses the **Rotten Tomatoes** movie review dataset.

The dataset contains:

```text
Train:      8,530 samples
Validation: 1,066 samples
Test:       1,066 samples
```

with:

```text
text
label
```

features. :contentReference[oaicite:4]{index=4}

The task is binary sentiment classification:

```text
0 -> Negative Review
1 -> Positive Review
```

### Topics Covered

- Text classification
- Sentiment analysis
- Representation models
- Task-specific classification models
- Sentence embeddings
- Logistic Regression
- Cosine similarity classification
- Zero-shot classification
- Encoder-decoder generative models
- Prompt-based classification
- Generative LLM classification
- Model evaluation

### 7.1 Task-Specific Sentiment Model

The notebook uses:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

with a Hugging Face text-classification pipeline. :contentReference[oaicite:5]{index=5}

The model is evaluated across the test dataset and predictions are compared against the true labels. :contentReference[oaicite:6]{index=6}

### Performance

The task-specific sentiment model achieved approximately:

```text
Accuracy: 80%
F1 Score: 0.80
```

on the test set. :contentReference[oaicite:7]{index=7}

---

### 7.2 Classification Using Sentence Embeddings

The notebook uses:

```text
sentence-transformers/all-mpnet-base-v2
```

to convert movie reviews into numerical embeddings. :contentReference[oaicite:8]{index=8}

The resulting training representation has:

```text
8,530 documents
768 embedding dimensions
```

```text
(8530, 768)
```

### Logistic Regression

A Logistic Regression classifier is trained on the generated embeddings:

```python
clf = LogisticRegression(random_state=42)

clf.fit(
    train_embeddings,
    data["train"]["label"]
)
```

The classifier achieves approximately:

```text
Accuracy: 85%
F1 Score: 0.85
```

on the test set. :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}

### Key Insight

A pretrained embedding model can be combined with a traditional machine learning classifier.

```text
Movie Review
     ↓
Sentence Transformer
     ↓
Embedding
     ↓
Logistic Regression
     ↓
Sentiment Prediction
```

---

### 7.3 Classification Using Cosine Similarity

Instead of training a classifier, the notebook also averages embeddings for each target class and compares test embeddings against these class representations using cosine similarity. :contentReference[oaicite:11]{index=11}

This approach achieves approximately:

```text
Accuracy: 84%
F1 Score: 0.84
```

on the test set. :contentReference[oaicite:12]{index=12}

This demonstrates that embeddings themselves can be used for classification without necessarily training a separate classifier.

---

### 7.4 Zero-Shot Classification with Embeddings

The notebook creates embeddings for text labels:

```text
A negative review
A positive review
```

and compares the test document embeddings with these label embeddings using cosine similarity. :contentReference[oaicite:13]{index=13}

The approach achieves approximately:

```text
Accuracy: 78%
F1 Score: 0.78
```

on the test set. :contentReference[oaicite:14]{index=14}

### Key Concept

The model does not receive examples explicitly labeled for the classification task. Instead, the input is compared semantically against natural-language descriptions of the possible classes.

---

### 7.5 Generative Classification with FLAN-T5

The notebook also explores classification using the encoder-decoder model:

```text
google/flan-t5-small
```

through the `text2text-generation` pipeline. :contentReference[oaicite:15]{index=15}

A prompt is created:

```text
Is the following sentence positive or negative?
```

and combined with each review before generation. :contentReference[oaicite:16]{index=16}

The generated output is converted into a classification label:

```text
negative -> 0
positive -> 1
```

The model achieves approximately:

```text
Accuracy: 84%
F1 Score: 0.84
```

on the test set. :contentReference[oaicite:17]{index=17}

### Key Concept

Instead of directly predicting a classification label, a generative model produces the answer as text.

```text
Movie Review
     ↓
Prompt
     ↓
FLAN-T5
     ↓
"positive" / "negative"
     ↓
Classification Label
```

---

### 7.6 LLM-Based Classification

The notebook also demonstrates prompt-based classification using an OpenAI model.

The prompt instructs the model to return:

```text
1 -> Positive
0 -> Negative
```

for a given movie review. :contentReference[oaicite:18]{index=18} :contentReference[oaicite:19]{index=19}

The notebook also demonstrates evaluating the model across the complete test dataset. The recorded experiment achieved approximately:

```text
Accuracy: 91%
F1 Score: 0.91
```

on the 1,066-example test set. :contentReference[oaicite:20]{index=20} :contentReference[oaicite:21]{index=21}

### Important Note

The OpenAI section requires an API key to run. The notebook uses a placeholder:

```python
client = openai.OpenAI(
    api_key="YOUR_KEY_HERE"
)
```

so API credentials should never be committed to GitHub. :contentReference[oaicite:22]{index=22}

### Overall Classification Comparison

| Approach | Approx. Accuracy |
|---|---:|
| Task-Specific RoBERTa | 80% |
| Sentence Embeddings + Logistic Regression | 85% |
| Embedding + Cosine Similarity | 84% |
| Zero-Shot Embedding Classification | 78% |
| FLAN-T5 Generative Classification | 84% |
| OpenAI LLM Classification | 91% |

### Key Learning

This notebook demonstrates that text classification can be approached in several different ways:

```text
                    Text Classification
                           │
          ┌────────────────┴────────────────┐
          │                                 │
 Representation Models              Generative Models
          │                                 │
   ┌──────┴──────┐                    ┌─────┴─────┐
   │             │                    │           │
Task-specific  Embeddings          FLAN-T5     LLM
RoBERTa        + Classifier
                  │
          Cosine Similarity
                  │
             Zero-Shot
```

The experiments demonstrate the trade-offs between task-specific models, reusable embeddings, traditional machine learning classifiers, zero-shot approaches, and generative LLMs.

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
├── 07_text_classification.ipynb
├── README.md
└── requirements.txt
```

---

# Skills Demonstrated

- Large Language Models (LLMs)
- Natural Language Processing (NLP)
- Text Classification
- Sentiment Analysis
- Tokenization
- Subword Tokenization
- Contextualized Embeddings
- Sentence Embeddings
- Word2Vec
- Representation Learning
- Recommendation Systems
- Similarity Search
- Cosine Similarity
- Zero-Shot Classification
- Logistic Regression
- Transformer Encoders
- Transformer Decoders
- Self-Attention Concepts
- Hidden States
- Logits
- Language Model Heads
- Greedy Decoding
- Autoregressive Generation
- KV Caching
- Prompt Engineering
- Hugging Face Transformers
- Hugging Face Datasets
- Sentence Transformers
- Scikit-Learn
- PyTorch
- GPU Inference
- Python Development
- Model Evaluation

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
✅ Text Classification  
✅ Sentiment Analysis  
✅ Sentence Embedding Classification  
✅ Zero-Shot Classification  
✅ Generative Classification  
✅ LLM-Based Classification  


- OpenAI Documentation
- OpenAI tiktoken Documentation
- Google FLAN-T5 Documentation
