# LLM From Scratch

A collection of hands-on experiments and notebooks documenting my journey of learning Large Language Models (LLMs), from tokenization and embeddings to transformer architecture, text generation, decoding, text classification, clustering, topic modeling, inference optimization, fine-tuning, and LLM applications.

## Objectives

- Understand how modern LLMs work internally
- Learn tokenization and embeddings
- Understand Transformer inputs and outputs
- Explore attention mechanisms and Transformer architecture
- Understand how LLMs generate text token by token
- Learn how decoding and probability distributions work
- Apply embeddings to real-world machine learning problems
- Explore text classification using representation and generative models
- Understand text clustering and topic modeling
- Learn how semantic embeddings can be clustered
- Explore BERTopic and its modular architecture
- Understand dimensionality reduction techniques
- Explore topic representation models
- Understand inference optimization techniques such as KV caching
- Build LLM components from scratch
- Work with open-source language models
- Develop practical AI Engineering skills

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Google Colab
- NumPy
- Pandas
- Scikit-Learn
- Sentence Transformers
- Gensim (Word2Vec)
- UMAP
- HDBSCAN
- BERTopic
- OpenAI API
- Matplotlib
- DataMapPlot
- WordCloud

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

features.

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

### Task-Specific Sentiment Model

The notebook uses:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

with a Hugging Face text-classification pipeline.

### Performance

Approximate test performance:

```text
Accuracy: 80%
F1 Score: 0.80
```

### Sentence Embeddings + Logistic Regression

The notebook uses:

```text
sentence-transformers/all-mpnet-base-v2
```

to convert movie reviews into embeddings.

The resulting embeddings have:

```text
8530 documents
768 dimensions
```

A Logistic Regression classifier is then trained on these embeddings.

Approximate performance:

```text
Accuracy: 85%
F1 Score: 0.85
```

### Classification Using Cosine Similarity

The notebook also averages embeddings for each target class and compares test embeddings with these class representations using cosine similarity.

Approximate performance:

```text
Accuracy: 84%
F1 Score: 0.84
```

### Zero-Shot Classification

The notebook creates embeddings for:

```text
A negative review
A positive review
```

and compares test document embeddings against these label embeddings using cosine similarity.

Approximate performance:

```text
Accuracy: 78%
F1 Score: 0.78
```

### Generative Classification with FLAN-T5

The notebook uses:

```text
google/flan-t5-small
```

with a prompt such as:

```text
Is the following sentence positive or negative?
```

The generated response is converted into a classification label.

Approximate performance:

```text
Accuracy: 84%
F1 Score: 0.84
```

### LLM-Based Classification

The notebook also demonstrates prompt-based classification using an OpenAI model.

The model is instructed to return:

```text
1 -> Positive
0 -> Negative
```

Approximate performance from the recorded experiment:

```text
Accuracy: 91%
F1 Score: 0.91
```

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

Text classification can be approached using multiple architectures:

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

---

# 8. Text Clustering and Topic Modeling

**File:** `08_text_clustering_and_topic_modeling.ipynb`

### Dataset

This notebook works with the **ArXiv NLP dataset**, containing research paper abstracts and titles. The dataset is loaded from Hugging Face:

```text
maartengr/arxiv_nlp
```

The notebook extracts:

```text
Abstracts
Titles
```

as the primary text data. :contentReference[oaicite:2]{index=2}

### Topics Covered

- Text clustering
- Document embeddings
- Semantic clustering
- Dimensionality reduction
- UMAP
- HDBSCAN
- BERTopic
- Topic modeling
- Topic representations
- c-TF-IDF
- Topic similarity search
- Topic visualization
- KeyBERTInspired representations
- Maximal Marginal Relevance
- Generative topic labeling
- FLAN-T5
- OpenAI-based topic representation
- DataMapPlot
- Word clouds

---

## 8.1 Creating Document Embeddings

The first step is to convert each research paper abstract into a numerical representation.

The notebook uses:

```text
thenlper/gte-small
```

through Sentence Transformers. :contentReference[oaicite:3]{index=3}

The resulting embeddings have:

```text
44,949 documents
384 dimensions
```

```text
(44949, 384)
```

This transforms the textual documents into vectors that can be compared based on semantic similarity. :contentReference[oaicite:4]{index=4}

### Pipeline

```text
Research Abstract
       ↓
Sentence Transformer
       ↓
384-Dimensional Embedding
       ↓
Semantic Representation
```

---

## 8.2 Dimensionality Reduction with UMAP

The 384-dimensional embeddings are reduced to 5 dimensions using UMAP.

```python
umap_model = UMAP(
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)
```

The reduced representations are then used for clustering. :contentReference[oaicite:5]{index=5}

### Why Dimensionality Reduction?

High-dimensional embeddings are difficult to cluster and visualize directly.

UMAP provides a lower-dimensional representation while attempting to preserve meaningful relationships between documents.

```text
384 Dimensions
      ↓
     UMAP
      ↓
5 Dimensions
```

---

## 8.3 Clustering with HDBSCAN

HDBSCAN is then applied to the reduced embeddings.

```python
hdbscan_model = HDBSCAN(
    min_cluster_size=50,
    metric='euclidean',
    cluster_selection_method='eom'
)
```

The experiment generates:

```text
156 clusters
```

including the outlier cluster represented by `-1`. :contentReference[oaicite:6]{index=6}

### Clustering Pipeline

```text
Documents
    ↓
Embeddings
    ↓
UMAP
    ↓
Reduced Embeddings
    ↓
HDBSCAN
    ↓
Clusters
```

---

## 8.4 Inspecting Clusters

The notebook manually inspects documents belonging to a specific cluster.

For example, documents in cluster `0` contain research related to:

```text
Sign Language
Sign Language Translation
Language Recognition
Animation and Synthesis
```

This demonstrates how semantic embeddings allow documents discussing similar subjects to be grouped together. :contentReference[oaicite:7]{index=7}

---

## 8.5 Visualizing Document Clusters

The embeddings are further reduced to two dimensions using UMAP:

```python
UMAP(
    n_components=2,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)
```

This allows the clusters to be plotted visually. :contentReference[oaicite:8]{index=8}

The notebook creates a scatter plot separating:

```text
Clusters
Outliers
```

to provide a visual representation of the document distribution. :contentReference[oaicite:9]{index=9}

---

# 8.6 From Text Clustering to Topic Modeling

After creating clusters, the notebook moves from simply grouping documents to identifying what each cluster is about.

This is where **BERTopic** is introduced. :contentReference[oaicite:10]{index=10}

### BERTopic

BERTopic is used as a modular topic-modeling framework:

```python
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True
)
```

The model is then trained on the research abstracts and their embeddings. :contentReference[oaicite:11]{index=11}

### Key Concept

The complete process becomes:

```text
Documents
    ↓
Embeddings
    ↓
UMAP
    ↓
HDBSCAN
    ↓
Clusters
    ↓
Topic Representation
    ↓
Human-Interpretable Topics
```

---

## 8.7 Exploring Topics

BERTopic produces topic information containing:

- Topic ID
- Document count
- Topic name
- Topic representation
- Representative documents

The notebook generates 156 topic entries, including an outlier topic. :contentReference[oaicite:12]{index=12}

Example topics include:

```text
Speech / ASR / Recognition
Medical / Clinical / Biomedical
Sentiment Analysis
Machine Translation
Summarization
Prompt Optimization
Sentence Embeddings
```

This demonstrates how clustering can be transformed into meaningful semantic topics.

---

## 8.8 Topic Keywords and c-TF-IDF

The `get_topic()` function can be used to inspect the most important words associated with a topic.

For example, one topic contains keywords such as:

```text
speech
asr
recognition
acoustic
speaker
audio
```

along with their corresponding weights. :contentReference[oaicite:13]{index=13}

BERTopic uses class-based TF-IDF (c-TF-IDF) to help identify representative words for each topic.

---

## 8.9 Searching for Topics

The notebook demonstrates semantic topic search using:

```python
topic_model.find_topics("topic modeling")
```

The search returns topics ranked according to similarity to the query.

For example:

```text
Topic 22
Similarity ≈ 0.95
```

The topic contains keywords related to:

```text
topic
topics
LDA
latent
document
modeling
Dirichlet
word
allocation
```

demonstrating semantic retrieval of related topics. :contentReference[oaicite:14]{index=14}

---

# 8.10 BERTopic Visualizations

The notebook explores multiple BERTopic visualization techniques.

### Document Visualization

```python
topic_model.visualize_documents(...)
```

This visualizes documents and their relationship to topics. :contentReference[oaicite:15]{index=15}

### Topic Bar Chart

```python
topic_model.visualize_barchart()
```

Displays important keywords associated with topics.

### Topic Heatmap

```python
topic_model.visualize_heatmap(n_clusters=30)
```

Visualizes relationships between topics.

### Topic Hierarchy

```python
topic_model.visualize_hierarchy()
```

Visualizes potential hierarchical relationships between topics. :contentReference[oaicite:16]{index=16}

---

# 8.11 Topic Representation Models

BERTopic allows topic representations to be updated after training.

This notebook explores multiple representation techniques. :contentReference[oaicite:17]{index=17}

### KeyBERTInspired

The notebook uses:

```python
KeyBERTInspired()
```

to update the topic representations. :contentReference[oaicite:18]{index=18}

This produces more semantically meaningful keywords for topics.

Example:

```text
Original:
speech | asr | recognition | end | acoustic

Updated:
speech | encoder | phonetic | language | translation
```

---

## 8.12 Maximal Marginal Relevance

The notebook also explores:

```python
MaximalMarginalRelevance(diversity=0.5)
```

as a topic representation model. :contentReference[oaicite:19]{index=19}

The goal is to improve the diversity of the selected topic keywords instead of returning highly repetitive terms.

---

# 8.13 Topic Representation with FLAN-T5

The notebook uses:

```text
google/flan-t5-small
```

to generate human-readable topic descriptions.

A prompt is constructed using:

```text
Documents
+
Topic Keywords
↓
FLAN-T5
↓
Topic Description
```

The generated representations include descriptions such as:

```text
Speech-to-description
Science/Tech
Review
Attention-based neural machine translation
Summarization
```

The notebook uses the BERTopic `TextGeneration` representation model for this process. :contentReference[oaicite:20]{index=20} :contentReference[oaicite:21]{index=21}

### Key Learning

A generative model can be used to convert machine-generated topic keywords into more understandable natural-language topic descriptions.

---

# 8.14 Topic Representation with OpenAI

The notebook also demonstrates using an OpenAI model to generate concise topic labels.

The prompt instructs the model to generate:

```text
topic: <short topic label>
```

based on the documents and keywords associated with each topic. :contentReference[oaicite:22]{index=22}

Example generated topic labels include:

```text
Leveraging External Data for Improving Low-Resource...
Improved Representation Learning for Biomedical...
Advancements in Aspect-Based Sentiment Analysis
Neural Machine Translation Enhancements
Document Summarization Techniques
```

This demonstrates how an LLM can be used as a **topic representation layer** on top of an unsupervised clustering pipeline. :contentReference[oaicite:23]{index=23}

---

# 8.15 DataMapPlot Visualization

The notebook also creates a document-level visualization using DataMapPlot:

```python
topic_model.visualize_document_datamap(...)
```

The visualization maps documents and topics into a two-dimensional space and can be saved as:

```text
datamapplot.png
```

This provides another way of visually exploring relationships between documents and topics. :contentReference[oaicite:24]{index=24}

---

# 8.16 Topic Word Cloud

The notebook includes a bonus WordCloud visualization.

First, the number of topic words is increased:

```python
topic_model.update_topics(
    abstracts,
    top_n_words=500
)
```

Then the most important words for a topic are visualized as a word cloud. :contentReference[oaicite:25]{index=25}

### Pipeline

```text
Topic
 ↓
Important Words
 ↓
Word Frequencies
 ↓
WordCloud
 ↓
Visual Topic Representation
```

---

# Key Learning

This notebook demonstrates a complete modern text-clustering and topic-modeling pipeline:

```text
             Research Documents
                     ↓
             Sentence Embeddings
                     ↓
              384-D Vectors
                     ↓
                  UMAP
                     ↓
              Reduced Vectors
                     ↓
                 HDBSCAN
                     ↓
                 Clusters
                     ↓
                BERTopic
                     ↓
            Topic Representations
                     ↓
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
    KeyBERT        MMR          LLMs
       ↓             ↓        ┌────┴────┐
   Keywords      Diverse      ↓         ↓
                 Keywords   FLAN-T5   OpenAI
       │             │         │         │
       └─────────────┴─────────┴─────────┘
                     ↓
            Human-Readable Topics
```

The notebook demonstrates how modern embedding models and clustering algorithms can be combined with generative LLMs to transform thousands of unstructured documents into interpretable topics.


# Skills Demonstrated

- Large Language Models (LLMs)
- Natural Language Processing (NLP)
- Text Classification
- Sentiment Analysis
- Text Clustering
- Topic Modeling
- Document Embeddings
- Sentence Embeddings
- Contextualized Embeddings
- Word2Vec
- Representation Learning
- Semantic Similarity
- Semantic Search
- Cosine Similarity
- Zero-Shot Classification
- Logistic Regression
- Dimensionality Reduction
- UMAP
- HDBSCAN
- BERTopic
- c-TF-IDF
- Topic Representation
- KeyBERTInspired
- Maximal Marginal Relevance
- Generative Topic Modeling
- Prompt Engineering
- Hugging Face Transformers
- Hugging Face Datasets
- Sentence Transformers
- Scikit-Learn
- Gensim
- PyTorch
- OpenAI API
- GPU Inference
- Python Development
- Model Evaluation
- Data Visualization
