# Lumora Architecture

## High-Level Workflow

Lumora follows a Retrieval-Augmented Generation (RAG) pipeline for converting semester study material into a searchable academic knowledge base.

The system uses externally preprocessed OCR text (through Mistral OCR), cleans and structures the content, converts it into embeddings, stores vectors inside FAISS, retrieves relevant information through semantic search, and generates grounded responses using Gemini and LM Studio local LLM (Meta-Llama-3.1-8B).

---

## Architecture Flow

Academic PDFs
↓
External OCR Preprocessing (Mistral OCR)
↓
Gemini/Manual Cleanup & Formatting
↓
Heading-Aware Chunking
↓
Sentence Transformer Embeddings
↓
FAISS Vector Store
↓
Semantic Retrieval + Metadata Reranking
↓
Confidence Threshold Filter
↓
Gemini Answer Generation + Local LLM (Meta-Llama-3.1-8B)
↓
Diagram Layer
↓
Lumora AI

---

## Component Breakdown

### 1. OCR Preprocessing

**Source:** External preprocessing workflow using Mistral OCR

**Purpose:**

* Extract text from scanned PDFs
* Convert academic notes into machine-readable text

**Tool Used:**

* Mistral OCR

**Output:**

* Clean markdown / text files

---

### 2. Text Cleaning

**Purpose:**

* Remove OCR noise
* Normalize formatting
* Preserve academic hierarchy

**Tools:**

* Gemini cleanup workflow

**Output:**

* Structured academic text

---

### 3. Chunking

**File:**
`src/chunk.py`

**Purpose:**
Split documents into semantically meaningful retrieval units.

**Features:**

* Heading-aware chunking
* Overlap handling
* Paragraph preservation
* Metadata tagging
* Chunk titles

**Output:**

* Chunked corpus

---

### 4. Embedding Pipeline

**File:**
`src/embed.py`

**Purpose:**
Convert chunks into vector representations.

**Features:**

* Normalized embeddings
* Semantic similarity
* Vector persistence

**Output:**

* FAISS index
* Metadata store

---

### 5. Retrieval Engine

**File:**
`src/retrieve.py`

**Purpose:**
Retrieve relevant syllabus content.

**Features:**

* Semantic similarity search
* Metadata-aware reranking
* Confidence filtering
* Interactive querying

---

### 6. Answer Generation

**Purpose:**
Generate grounded answers using retrieved context.

**Models:**

* Google Gemini
* LM Studio fallback

**Behavior:**
Responses are generated only using retrieved context to reduce hallucination.

---

### 7. Hallucination Guard

**Purpose:**
Prevent irrelevant retrieval and unsupported answers.

**Mechanism:**
Minimum confidence threshold.

If:

`score < threshold`

then:

`No relevant information found in knowledge base`

---

## Evaluation Summary

Stress testing included:

* Direct definitions
* Comparison queries
* Reasoning questions
* Hallucination checks
* Diagram-linked retrieval

**Observed Performance:**

~90% retrieval reliability

---

## Current System Status

Working RAG-based AI Study Assistant with conversational UI and diagram-aware retrieval.
