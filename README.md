# Ontology-Constrained Retrieval for Historical RAG

Code accompanying the paper:

**Ontology-Constrained Retrieval for Historical RAG**

## Overview

This repository contains the experimental code used to compare two retrieval architectures:

1. Semantic Vector Retrieval (SVR)
2. Ontology-Constrained Retrieval (OCR)

The goal is to evaluate how structural constraints derived from Wikidata relations can reduce semantic noise in fact retrieval.

Unlike document-based RAG systems, this work operates on **atomic factual statements** extracted from Wikidata and represented as vector embeddings.

## Repository Structure

core_vector_graph.py  
Core retrieval logic.

run_vector_graph.py  
Script used to run the experimental configurations.

queries/  
SPARQL queries used to extract the Wikidata subset.

docs/  
Figures used in the paper (pipeline diagrams).

examples/  
Example natural language queries used in the experiments.

## Experimental Setup

The experiments operate on a dataset derived from Wikidata consisting of approximately **191,000 factual statements** related to the period **1400–1700**.

Each statement is represented as a structured record containing:

- source entity (SRC)
- relation (PID)
- destination entity (DST)
- optional temporal metadata
- natural language linearization

The statements are embedded using the model:

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

and indexed in the **Qdrant vector database**.

## Retrieval Configurations

Two configurations are evaluated:

**SVR – Semantic Vector Retrieval**

Standard vector similarity search.

**OCR – Ontology-Constrained Retrieval**

Vector retrieval combined with structural constraints derived from Wikidata relations and temporal metadata.

## Example Usage

Semantic baseline:
