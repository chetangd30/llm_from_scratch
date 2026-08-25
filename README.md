# LLM From Scratch

A collection of hands-on experiments and notebooks documenting my journey of learning Large Language Models (LLMs), from tokenization and embeddings to transformer architecture, text generation, decoding, text classification, clustering, topic modeling, inference optimization, fine-tuning, and LLM applications.


---

## Table of Contents

We advise to run all examples through Google Colab for the easiest setup. Google Colab allows you to use a T4 GPU with 16GB of VRAM for free. All examples were mainly built and tested using Google Colab, so it should be the most stable platform. However, any other cloud provider should work. 

| Chapter  
|---|
| Chapter 1: Introduction to Language Models 
| Chapter 2: Tokens and Embeddings  
| Chapter 3: Looking Inside Transformer LLMs 
| Chapter 4: Text Classification  
| Chapter 5: Text Clustering and Topic Modeling 
| Chapter 6: Prompt Engineering  
| Chapter 7: Advanced Text Generation Techniques and Tools 
| Chapter 8: Semantic Search and Retrieval-Augmented Generation  
| Chapter 9: Multimodal Large Language Models  
| Chapter 10: Creating Text Embedding Models  
| Chapter 11: Fine-tuning Representation Models for Classification 
| Chapter 12: Fine-tuning Generation Models  

# Chapter Overview

This repository follows the 12-chapter structure of *Hands-On Large Language Models* and documents the concepts, experiments, and practical techniques explored throughout the book.

## 1. Introduction to Language Models

Introduces the foundations of modern language models, including what LLMs are, how they evolved, and how pretrained models are used. Covers open-source and proprietary models, basic inference, and the overall LLM development workflow.

## 2. Tokens and Embeddings

Explores how language is represented numerically through tokens and embeddings. Covers tokenization strategies, token embeddings, contextualized representations, word embeddings, and how embeddings can be used for tasks such as semantic similarity and recommendation.

## 3. Looking Inside Transformer LLMs

Examines the internal architecture of Transformer-based language models. Covers the forward pass, attention, token prediction, decoding, context size, KV caching, positional embeddings, and improvements such as more efficient attention mechanisms.

## 4. Text Classification

Applies language models to text classification tasks such as sentiment analysis. Explores representation models, task-specific models, embedding-based classification, zero-shot approaches, and generative models such as T5 and ChatGPT.

## 5. Text Clustering and Topic Modeling

Moves from supervised classification to unsupervised analysis of text. Covers document embeddings, dimensionality reduction, clustering, cluster inspection, and topic modeling with BERTopic, including topic representations and generative topic labeling.

## 6. Prompt Engineering

Explores how prompts can control and improve the behavior of generative models. Covers prompt structure, instruction-based prompting, in-context learning, chain prompting, reasoning techniques such as chain-of-thought and self-consistency, and output verification.

## 7. Advanced Text Generation Techniques and Tools

Builds more capable LLM applications without fine-tuning the model. Covers quantized model loading, LangChain, prompt chains, conversation memory, agents, and ReAct-style reasoning with external tools.

## 8. Semantic Search and Retrieval-Augmented Generation

Explores how LLM applications can retrieve relevant information from external data. Covers dense retrieval, embeddings, chunking, reranking, retrieval evaluation, and RAG pipelines, including advanced techniques such as query rewriting, multi-query, multi-hop, and agentic RAG.

## 9. Multimodal Large Language Models

Extends language models beyond text by combining language and vision. Covers Vision Transformers, multimodal embeddings, CLIP/OpenCLIP, BLIP-2, multimodal preprocessing, image captioning, and image-based conversational prompting.

## 10. Creating Text Embedding Models

Explores how custom embedding models are created and improved. Covers contrastive learning, SBERT, training and evaluating embedding models, loss functions, supervised fine-tuning, Augmented SBERT, and unsupervised approaches such as TSDAE for domain adaptation.

## 11. Fine-Tuning Representation Models for Classification

Focuses on adapting pretrained representation models to specific classification tasks. Covers BERT fine-tuning, freezing layers, few-shot classification with SetFit, continued pretraining with masked language modeling, and fine-tuning models for named-entity recognition.

## 12. Fine-Tuning Generation Models

Covers the final stage of adapting generative language models to desired behavior. Explores supervised fine-tuning, full fine-tuning, parameter-efficient methods such as LoRA and QLoRA, quantization, instruction tuning, model evaluation, and preference tuning methods including DPO and reward-model-based alignment.

---

# Learning Path

The project progresses from understanding the fundamentals of language models to building practical LLM systems and finally training and fine-tuning models:

**Foundations → Transformers → LLM Applications → RAG & Multimodality → Embedding Models → Fine-Tuning & Alignment**

---

# Repository Structure

```text
llm-from-scratch/
│
├── notebooks/
├── README.md
└── requirements.txt
```

---

# Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Sentence Transformers
- Scikit-Learn
- Gensim
- UMAP
- HDBSCAN
- BERTopic
- LangChain
- OpenAI API
- Google Colab
