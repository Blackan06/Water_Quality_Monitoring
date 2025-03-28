from elasticsearch import Elasticsearch, NotFoundError


# Elasticsearch configuration
ES_HOST = "https://elasticsearch.anhkiet.xyz"
ES_INDEX = "water_quality_logs"

es_client = Elasticsearch([ES_HOST])

def check_and_delete_index():
    """Check if the index exists in Elasticsearch and delete if it does."""
    try:
        # Check if index exists
        if es_client.indices.exists(index=ES_INDEX):
            # Delete the index if it exists
            es_client.indices.delete(index=ES_INDEX)
            print(f"Index {ES_INDEX} deleted successfully.")
        else:
            print(f"Index {ES_INDEX} does not exist.")
    except NotFoundError:
        print(f"Index {ES_INDEX} does not exist.")
    except Exception as e:
        print(f"Error occurred while checking/deleting index: {e}")
        raise