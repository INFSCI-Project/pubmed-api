import numpy as np
import spacy

from collections import Counter
from elasticsearch import Elasticsearch
from preprocessing.embeddings import BioBertEmbedding


class SemanticSearch:
    def __init__(self, es_client, index_name):
        self.es_client = es_client
        self.index_name = index_name
        self.embedder = BioBertEmbedding()
        # Load ScispaCy biomedical model
        self.nlp = spacy.load("en_core_sci_md")

    def extract_terms(self, text):
        doc = self.nlp(text)
        terms = [ent.text for ent in doc.ents]
        return terms

    def expand_query(self, original_query, top_documents, top_n=5):
        print("Expanding query...")
        expanded_terms = []
        for doc in top_documents:
            expanded_terms.extend(self.extract_terms(doc))
        expanded_terms = Counter(expanded_terms).most_common(top_n)
        expanded_query_terms = [term[0] for term in expanded_terms]
        expanded_query = original_query + " " + " ".join(expanded_query_terms)

        # Generate embeddings for the expanded terms
        expanded_term_embeddings = [
            self.embedder.generate_embedding(term) for term in expanded_query_terms
        ]
        expanded_term_embeddings = np.mean(expanded_term_embeddings, axis=0)
        return expanded_query, np.array(expanded_term_embeddings)

    def apply_pseudo_relevant_feedback(self, query_embedding, topK, alpha=0.4):
        # Construct a hybrid search query with HNSW
        es_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "field": "biobert_embedding",
                                "query_vector": query_embedding,
                                "k": topK  # Number of nearest neighbors to retrieve
                            }
                        }
                    ]
                },
            },
            "size": 50
        }

        response = self.es_client.search(index=self.index_name, body=es_query)

        hits = response["hits"]["hits"]
        topK_documents = [hit["_source"]["abstract"] for hit in hits]
        topK_embeddings = np.mean(
            [hit["_source"]["biobert_embedding"] for hit in hits], axis=0)

        pseudo_embedding = alpha * \
            np.array(query_embedding) + (1 - alpha) * topK_embeddings

        return pseudo_embedding, topK_documents

    def execute_semantic_search(self, query, alpha=0.7):
        query_embedding = self.embedder.generate_embedding(query).tolist()

        pseudo_relevance_embedding, topK_docs = self.apply_pseudo_relevant_feedback(
            query_embedding, 100)

         # Query Expansion
        expanded_query, expanded_query_embeddings = self.expand_query(
            query, topK_docs, top_n=5)

        # Merge with pseudo-relevance embedding
        expanded_embedding = alpha * \
            pseudo_relevance_embedding + \
                (1 - alpha) * expanded_query_embeddings

        # Construct a hybrid search query with HNSW
        es_query = {
            "_source": ["title", "abstract", "authors", "doi", "entities"],
            "query": {
                "bool": {
                    "should": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'biobert_embedding') + 2.5",
                                    "params": {"query_vector": expanded_embedding.tolist()}
                                }
                            }
                        },
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'title_embedding') + 1.5",
                                    "params": {"query_vector": query_embedding}
                                }
                            }
                        }
                    ]
                }
            },
            "aggs": {
                "tja-agg": {
                    "nested": {
                        "path": "entities"
                    },
                    "aggs": {
                        "labels": {
                            "terms": {
                                "field": "entities.entity",
                                "size": 10  # Number of top labels to return
                            }
                        }
                    }
                }
            },
       
            "size": 50  # Number of results to return in the response
        }

        return self.es_client.search(index=self.index_name, body=es_query)
