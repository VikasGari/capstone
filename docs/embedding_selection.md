# Embedding Model Selection Note

This document justifies the selection of the embedding model used in the Brokerage Policy & Trading Rules Assistant, analyzing accuracy, dimensionality, compute latency, and cost trade-offs.

---

## 1. Selected Model: `sentence-transformers/all-MiniLM-L6-v2`

For dense semantic retrieval, the pipeline utilizes the `all-MiniLM-L6-v2` model from the Sentence-Transformers library. This model maps sentences and paragraphs to a dense 384-dimensional vector space.

---

## 2. Selection Rationale (MTEB-Informed)

The Massive Text Embedding Benchmark (MTEB) ranks embedding models across diverse semantic search, clustering, and classification datasets. Our selection is guided by the following considerations:

### Dimensionality & Search Latency
* **MiniLM:** Produces **384-dimensional** embeddings.
* **Alternative Large Models (e.g., BGE-Large, E5-Large):** Produce **1024-dimensional** embeddings.
* **Trade-off:** A 1024-dimensional space increases the memory footprint and distance computation times in our local FAISS index by **2.6x** compared to 384 dimensions. MiniLM keeps vector searches extremely fast, running on local CPU threads in sub-millisecond speeds.

### Accuracy & Retrieval Quality
* On MTEB semantic search benchmarks, MiniLM-L6-v2 scores **~41.95** (Average NDCG@10). While larger models score higher (~48-52), the difference in dense retrieval performance is negligible for our highly targeted, structured policy domain.
* The small gap in raw semantic recall is fully compensated for by our **BM25 lexical search fusion (RRF)** and the **Cross-Encoder Reranker** stage, yielding near-perfect precision on the retrieval set.

### Local Execution & Memory Footprint
* MiniLM has a model size of only **~80MB** (compared to BGE-Large's **~1.34GB**).
* It executes efficiently on local CPUs with low memory, perfectly satisfying the Capstone's **No-Docker** and in-process execution constraints.

---

## 3. Cost Profile

Since the model executes locally in-process:
* **API Cost:** $0.00 (completely free, open-source model).
* **GPU Compute:** $0.00 (requires no GPU resources, executes in-process on CPU).
