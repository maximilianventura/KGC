
# Ontology-Constrained Retrieval for Historical RAG

Code accompanying the paper:

**Ontology-Constrained Retrieval for Historical RAG**

## Overview

This repository contains the experimental code used to compare two retrieval architectures for historical fact retrieval:

1. **Semantic Vector Retrieval (SVR)**
2. **Ontology-Constrained Retrieval (OCR)**

The goal of the experiment is to evaluate how structural constraints derived from Wikidata relations can reduce semantic noise during vector retrieval.

Unlike document-based RAG systems, this work operates on **atomic factual statements** derived from Wikidata and represented as vector embeddings.

## 📂 Repository Structure

```text
├── core_vector_graph.py      # Core retrieval logic used in the experiments
├── run_vector_graph.py       # Script used to run the experimental configurations
├── requirements.txt          # List of Python dependencies
├── README.md                 # Repository documentation
├── LICENSE                   # Open-source license
└── queries/                  
    └── extract_people_1400_1700.sparql  # SPARQL query for initial data extraction




## Repository Structure

core_vector_graph.py  
run_vector_graph.py  
requirements.txt  
README.md  
LICENSE  
queries/  
    extract_people_1400_1700.sparql

**core_vector_graph.py**  
Core retrieval logic used in the experiments.

**run_vector_graph.py**  
Script used to run the experimental configurations and reproduce the experiments described in the paper.

**requirements.txt**  
List of Python dependencies required to run the code.

**queries/**  
Contains the SPARQL query used for the initial extraction of historical entities from Wikidata.

## Experimental Setup

The experiments operate on a dataset derived from Wikidata consisting of approximately **191,000 factual statements** related to the period **1400–1700**.

Each statement is represented as a structured record containing:

- SRC – source entity (person, place, institution)
- PID – relation (Wikidata Property ID)
- DST – destination entity
- temporal metadata – optional start/end dates
- text representation – linearized natural language form of the fact

Embeddings are generated using:

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

and indexed in the **Qdrant vector database**.

## Retrieval Configurations

Two configurations are evaluated.

### SVR – Semantic Vector Retrieval

Standard vector similarity search based purely on semantic proximity between query and statement embeddings.

### OCR – Ontology-Constrained Retrieval

Vector retrieval combined with structural constraints derived from Wikidata relations and temporal metadata.  
These constraints restrict the search space and filter candidates that are structurally incompatible with the query.

## Example Usage

In the codebase the two configurations correspond to:

- `pure` → SVR (Semantic Vector Retrieval)
- `graph` → OCR (Ontology-Constrained Retrieval)


Semantic vector retrieval (SVR):

python run_vector_graph.py --mode pure --q "Who has occupation historian?"

Ontology-constrained retrieval (OCR):

python run_vector_graph.py --mode graph --q "Who studied in Padua and held the position of emperor?"

## Requirements

Python 3.10+

Install dependencies with:

pip install -r requirements.txt

Main dependencies include:

- qdrant-client
- sentence-transformers
- pandas
- numpy

## Dataset Reconstruction

The dataset used in the experiments is derived from Wikidata and is **not distributed directly**.

The initial extraction query is provided in the `queries/` directory.

The final corpus used in the experiments was obtained through a controlled preprocessing pipeline including:

1. extraction of historical persons from Wikidata
2. relation extraction and normalization
3. statement linearization
4. temporal metadata consolidation
5. embedding generation
6. indexing in Qdrant

This process produces the experimental corpus of approximately **191,000 factual statements** used in the paper.

## Scope

This repository supports a **methodological experiment** on retrieval control in historical fact retrieval.

The goal is to study how structural constraints derived from knowledge graph relations can reduce semantic noise before generation.

The code is intended to reproduce the experiments described in the paper and is **not designed as a production search system**.
