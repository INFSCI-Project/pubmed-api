from elasticsearch import Elasticsearch

from preprocessing.embeddings import BioBertEmbedding 
from preprocessing.named_entity import NamedEntityExtraction

class ElasticsearchIndex:
    def __init__(self, es_client, index_name):
        self.es_client = es_client
        self.index_name = index_name
        self.embedder = BioBertEmbedding()
        self.ner = NamedEntityExtraction()

        self.es_index_schema = {
                "mappings": {
                    "properties": {
                        "title": {
                            "type": "text"
                        },
                        "abstract": {
                            "type": "text"
                        },
                        "doi": {
                             "type": "text"
                        },
                        "authors": {
                             "type": "text"
                        },
                        "entities": {
                            "type": "nested",
                            "properties": {
                                "entity": {"type": "keyword"},
                                "label": {"type": "keyword"}
                            }
                        },
                        "title_embedding": {
                            "type": "dense_vector",
                            "dims": 768,
                            "index": True,
                            "similarity": "cosine",
                            "index_options": {
                                "type": "hnsw",
                                "m": 16,  # Number of bi-directional links created for each element
                                "ef_construction": 200  # Defines accuracy/performance during indexing
                            }
                        },
                        "biobert_embedding": {
                            "type": "dense_vector",
                            "dims": 768,
                            "index": True,
                            "similarity": "cosine",
                            "index_options": {
                                "type": "hnsw",
                                "m": 16,  # Number of bi-directional links created for each element
                                "ef_construction": 200  # Defines accuracy/performance during indexing
                            }
                        }
                    }
                }
        }

    def create_index(self, drop=True):
        if self.es_client.indices.exists(index=self.index_name) and drop==True:
                self.es_client.indices.delete(index=self.index_name)

        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(index=self.index_name, body=self.es_index_schema)
            print(f"Index '{self.index_name}' created with HNSW and NER fields.")


    def insert_doc(self, text):
        biobert_embedding = self.embedder.generate_embedding(text["abstract"])
        title_embedding = self.embedder.generate_embedding(text["title"])
        ner_entities = self.ner.extract_ner(text["abstract"])

        doc = {
            "biobert_embedding": biobert_embedding.tolist(),
            "title_embedding": title_embedding.tolist(),
            "entities": ner_entities,
            "title": text["title"],
            "abstract": text["abstract"],
            "authors": text["authors"],
            "doi": "https://doi.org/" + text["doi"]
        }
        self.es_client.index(index=self.index_name, body=doc)