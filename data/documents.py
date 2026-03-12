"""
Sample document corpus for hybrid search experiments.
A mix of technical and general knowledge documents — designed to showcase
scenarios where keyword search excels, vector search excels, and hybrid wins.
"""

DOCUMENTS = [
    # --- Machine Learning Docs ---
    {
        "id": "doc_001",
        "title": "Introduction to Transformers",
        "text": (
            "Transformer models revolutionized NLP by replacing recurrent architectures "
            "with self-attention mechanisms. The original paper 'Attention Is All You Need' "
            "introduced the encoder-decoder structure used in models like BERT and GPT. "
            "Self-attention computes relationships between all token pairs simultaneously, "
            "enabling parallelization during training."
        ),
        "category": "machine_learning",
    },
    {
        "id": "doc_002",
        "title": "BM25 Ranking Function",
        "text": (
            "BM25 (Best Match 25) is a probabilistic ranking function used in information "
            "retrieval. It scores documents based on term frequency (TF) and inverse document "
            "frequency (IDF) with length normalization. BM25 is the default ranking function "
            "in Elasticsearch and Apache Lucene. The 'k1' and 'b' parameters control term "
            "frequency saturation and document length normalization respectively."
        ),
        "category": "information_retrieval",
    },
    {
        "id": "doc_003",
        "title": "Vector Embeddings Explained",
        "text": (
            "Dense vector embeddings represent text as high-dimensional floating point vectors. "
            "Models like sentence-transformers encode semantic meaning so that similar concepts "
            "cluster together in vector space. Unlike sparse bag-of-words representations, "
            "embeddings capture synonymy and contextual nuance. Cosine similarity measures "
            "the angle between vectors to find semantically related documents."
        ),
        "category": "machine_learning",
    },
    {
        "id": "doc_004",
        "title": "Retrieval Augmented Generation (RAG)",
        "text": (
            "RAG combines a retrieval system with a generative language model. Given a query, "
            "the retriever fetches relevant documents from a knowledge base. These documents "
            "are injected into the LLM prompt as context, grounding the generated response "
            "in factual information. RAG reduces hallucination and enables LLMs to access "
            "up-to-date knowledge without retraining."
        ),
        "category": "machine_learning",
    },
    {
        "id": "doc_005",
        "title": "Elasticsearch Full-Text Search",
        "text": (
            "Elasticsearch is a distributed search engine built on Apache Lucene. It supports "
            "full-text search using inverted indexes, allowing fast keyword lookups across "
            "large corpora. Elasticsearch uses BM25 by default for relevance scoring. It also "
            "supports vector search via the kNN API, making it suitable for hybrid retrieval "
            "pipelines that combine lexical and semantic search."
        ),
        "category": "information_retrieval",
    },
    {
        "id": "doc_006",
        "title": "Gradient Descent Optimization",
        "text": (
            "Gradient descent minimizes a loss function by iteratively adjusting model "
            "parameters in the direction of the negative gradient. Variants include stochastic "
            "gradient descent (SGD), mini-batch gradient descent, Adam, and AdaGrad. Learning "
            "rate scheduling, momentum, and weight decay are common techniques to stabilize "
            "and accelerate convergence during neural network training."
        ),
        "category": "machine_learning",
    },
    {
        "id": "doc_007",
        "title": "Inverted Index Data Structure",
        "text": (
            "An inverted index maps each unique term to the list of documents containing it. "
            "This structure underpins keyword search engines like Lucene and Solr. During "
            "indexing, text is tokenized, stemmed, and lowercased before being stored. At "
            "query time, matching document lists are intersected or unioned. Inverted indexes "
            "enable O(1) term lookups but cannot capture semantic similarity between words."
        ),
        "category": "information_retrieval",
    },
    {
        "id": "doc_008",
        "title": "Cosine Similarity and Vector Search",
        "text": (
            "Cosine similarity measures the cosine of the angle between two vectors, ranging "
            "from -1 to 1. A value of 1 means identical direction; 0 means orthogonal. In "
            "semantic search, document and query embeddings are compared using cosine similarity "
            "or dot product. Approximate nearest neighbor algorithms like HNSW and FAISS allow "
            "sub-linear time vector search over millions of embeddings."
        ),
        "category": "machine_learning",
    },
    {
        "id": "doc_009",
        "title": "Natural Language Processing Pipeline",
        "text": (
            "A typical NLP pipeline involves tokenization, part-of-speech tagging, named entity "
            "recognition, and dependency parsing. Modern transformer-based pipelines replace "
            "many hand-crafted steps with learned representations. Libraries like spaCy and "
            "Hugging Face Transformers provide pre-built pipeline components. Text normalization "
            "steps like stemming and lemmatization are still common in keyword search systems."
        ),
        "category": "nlp",
    },
    {
        "id": "doc_010",
        "title": "Precision and Recall in Information Retrieval",
        "text": (
            "Precision measures the fraction of retrieved documents that are relevant. Recall "
            "measures the fraction of relevant documents that were retrieved. There is a "
            "precision-recall tradeoff: retrieving more documents improves recall but may "
            "hurt precision. F1 score harmonically combines both metrics. Mean Average Precision "
            "(MAP) and NDCG are standard benchmarks for ranking quality in retrieval systems."
        ),
        "category": "information_retrieval",
    },
    {
        "id": "doc_011",
        "title": "Python asyncio and Concurrency",
        "text": (
            "Python's asyncio library enables concurrent I/O-bound tasks using coroutines and "
            "event loops. The async/await syntax defines coroutines that yield control without "
            "blocking the thread. asyncio is ideal for web scraping, API calls, and database "
            "queries. For CPU-bound tasks, multiprocessing or concurrent.futures with process "
            "pools provides true parallelism beyond the GIL."
        ),
        "category": "programming",
    },
    {
        "id": "doc_012",
        "title": "Docker Containerization",
        "text": (
            "Docker packages applications with their dependencies into portable containers. "
            "A Dockerfile defines the image build steps: base image, installed packages, "
            "copied files, and the startup command. Docker Compose orchestrates multi-container "
            "applications. Containers share the host OS kernel, making them lighter than "
            "virtual machines. Docker Hub hosts public images for popular software stacks."
        ),
        "category": "devops",
    },
    {
        "id": "doc_013",
        "title": "Reciprocal Rank Fusion (RRF)",
        "text": (
            "Reciprocal Rank Fusion is a score normalization method for combining ranked lists "
            "from multiple retrieval systems. Each document receives a score of 1/(k + rank) "
            "where k is a smoothing constant (typically 60). Scores from different rankers "
            "are summed to produce a final merged ranking. RRF is robust, parameter-free, "
            "and outperforms linear score combination in many hybrid retrieval benchmarks."
        ),
        "category": "information_retrieval",
    },
    {
        "id": "doc_014",
        "title": "Attention Mechanism in Deep Learning",
        "text": (
            "The attention mechanism allows neural networks to focus on relevant parts of the "
            "input when producing each output token. Scaled dot-product attention computes "
            "compatibility between query and key vectors, applies softmax to get weights, "
            "then takes a weighted sum of value vectors. Multi-head attention runs attention "
            "in parallel across multiple learned subspaces, capturing diverse relationships."
        ),
        "category": "machine_learning",
    },
    {
        "id": "doc_015",
        "title": "TF-IDF Weighting Scheme",
        "text": (
            "TF-IDF (Term Frequency-Inverse Document Frequency) is a classic weighting scheme "
            "for information retrieval. Term frequency rewards words appearing often in a "
            "document. Inverse document frequency penalizes words appearing in many documents "
            "(common words like 'the'). TF-IDF vectors form a sparse, high-dimensional "
            "representation used in traditional search engines before the embedding era."
        ),
        "category": "information_retrieval",
    },
]
