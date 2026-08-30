# Structure-Aware Chunking Strategy & Rationale

## 1. The Challenge of Naive Chunking
Standard RAG systems often use naive fixed-size character chunking (e.g., splitting every 500 characters with a 50-character overlap). While simple, this approach has severe issues for legal and policy documents:
* **Rule Splitting:** A critical margin threshold or cutoff time might be split right in the middle of a sentence, leading to incomplete facts in individual chunks.
* **Loss of Context:** A clause containing "The limit is 10%" has no meaning unless it is coupled with its parent header (e.g., "Securities Transaction Tax" or "Index Futures").
* **Diluted Citations:** If a chunk contains pieces of two different clauses, generating clean, clause-level citations is difficult.

## 2. Our Approach: Logical Clause-Level Segmentation
To resolve these issues, we implement a **Structure-Aware Recursive Chunking Strategy**:

### Step 1: Document Metadata Extraction
Each document in the synthetic corpus contains headers:
* `DOCUMENT ID`
* `DOCUMENT TITLE`
* `CATEGORY`

We parse these headers first to serve as global metadata for all chunks in that file.

### Step 2: Regex-Based Clause Segmentation
We parse the document body and split it into logical blocks based on the pattern `(Clause \d+\.\d+:|Section \d+\.\d+:)`. This segments the text into distinct legal rules (e.g., `Clause 1.1: Normal Market Hours` and `Clause 1.2: Post-Closing Session` become independent text segments).

### Step 3: Recursive Character Subdivision
For each clause segment:
* We prepend the document title and clause title to the text block (e.g., `[Exchange Trading Hours - Clause 1.1: Normal Market Hours] The equity and derivatives segments...`). This injects global context directly into the text representation, boosting vector search recall.
* We apply a `RecursiveCharacterTextSplitter` with a default `chunk_size` of 600 characters and `chunk_overlap` of 100 characters. If a clause fits inside 600 characters (which is the case for most of our synthetic documents), it is kept as a single contiguous chunk. If it is longer, it is split cleanly along paragraph and sentence boundaries without breaking rules apart.

## 3. Metadata Schema Design
For each chunk, we store a structured metadata dictionary in Chroma:
* `source`: Filename (e.g., `EXCH_RULE_01_TRADING_HOURS.txt`)
* `doc_id`: Document identifier (e.g., `EXCH_RULE_01_TRADING_HOURS`)
* `doc_title`: Document title (e.g., `Exchange Trading Hours`)
* `doc_type`: Category (e.g., `Exchange Rulebook`)
* `clause_id`: Mapped clause/section ID (e.g., `Clause 1.1`)
* `clause_title`: Mapped clause/section title (e.g., `Normal Market Hours`)

This rich schema enables:
1. Precise, clause-level citations in the generated response.
2. Metadata-driven routing/filtering during retrieval.
