import base64
import json
import pickle
import re

from openai import OpenAI, OpenAIError


def save_knowledge_graph(graph, file_path):
    """
    Save a NetworkX DiGraph (knowledge graph) to a file.

    :param graph: NetworkX DiGraph to save
    :param file_path: Path to the file where the graph will be saved
    """
    with open(file_path, 'wb') as file:
        pickle.dump(graph, file)

def load_knowledge_graph(file_path):
    """
    Load a NetworkX DiGraph (knowledge graph) from a file.

    :param file_path: Path to the file where the graph is stored
    :return: Loaded NetworkX DiGraph
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def load_schema(file_path):
    """
    Load a schema from a JSON file.

    :param file_path: Path to the JSON file containing the schema
    :return: Dictionary with schema details
    """
    with open(file_path, "r") as file:
        return json.load(file)

def encode_pdf_to_base64(pdf_path):
    """Read PDF and encode it in Base64."""
    with open(pdf_path, "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode("utf-8")

# Function to clean source titles
def clean_title(title):
    """Remove unwanted characters at the beginning and end of a title."""
    return re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', title.strip())

