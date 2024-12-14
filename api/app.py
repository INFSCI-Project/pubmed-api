import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tqdm import tqdm

from elasticsearch_client import ElasticsearchClient
from indexing.elasticsearch_index import ElasticsearchIndex
from search.semantic import SemanticSearch
app = Flask(__name__)
CORS(app)

# route for health ES client heathcheck
@app.route("/healthcheck", methods=['GET'])
def healthcheck():
    try:
        es_client = ElasticsearchClient.get_client()
        health = es_client.cluster.health()
        return jsonify({"status": "OK", "elasticsearch": health}), 200
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

@app.route('/document/<doc_id>', methods=["GET"])
def get_document(doc_id):
    try:
        # Fetch the document by ID from the specified index
        response = ElasticsearchClient.get_client().get(index="pubmed-tja-v2", id=doc_id)
        return jsonify(response['_source'])  # Return only the document source
    except Exception as e:
        # Handle errors (e.g., document not found or index does not exist)
        return jsonify({"error": str(e)}), 404


@app.route('/search', methods=['POST'])
def search():
    search = SemanticSearch(ElasticsearchClient.get_client(),
                            "pubmed-tja-v2")
    try:
        # Extract the user query from the request
        data = request.json
        user_query = data.get("query", "")

        if not user_query:
            return jsonify({"error": "Query parameter is missing"}), 400

        response = search.execute_semantic_search(user_query)
        hits = response.get("hits", {}).get("hits", [])
    
        # Format the results
        results = [
            {
                "id": hit["_id"],
                "score": hit["_score"],
                "source": hit["_source"]
            } for hit in hits
        ]

        return jsonify({
            "total_results": response.get("hits").get("total", {}).get("value", 0),
            "returned_results": len(hits),
            "query": user_query, 
            "results": results, 
            "agg_data": response.get("aggregations")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.cli.command()
def create_index():
    """Create or re-create the Elasticsearch index."""
     # Create elasticsearch connection
    es_client = ElasticsearchClient.get_client()

    # Build index
    with open("data/PubMedData/pubmed-tja.json", "r") as f:
        data = json.load(f)

    data_samples = np.random.choice(data, 5000, replace=False)
    index_name = "pubmed-tja-v2"

    es_index = ElasticsearchIndex(es_client, index_name)
    es_index.create_index(drop=True)

    for doc in tqdm(data_samples):
        es_index.insert_doc(doc)

    print(f"Successfully indexed {len(data_samples)} into index: {index_name}.")


if __name__ == "__main__":
    app.run(port=3000, debug=True)
