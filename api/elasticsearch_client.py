from elasticsearch import Elasticsearch

class ElasticsearchClient:
    _client = None

    @classmethod
    def get_client(cls, hosts=["https://localhost:9200"]):
        if cls._client is None:
            cls._client = Elasticsearch(
                hosts,
                # Basic authentication
                 basic_auth=('elastic', 'bennyMan'),
                # Disable certificate verification (use cautiously)
                verify_certs=False,
                ssl_assert_hostname=False,
                ssl_show_warn=False
            )
        return cls._client
