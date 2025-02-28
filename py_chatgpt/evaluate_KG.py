from utils import load_knowledge_graph, load_schema


def diff_pair(kg1, kg2):
    """
    Calculate the difference in node pairs regardless of direction.
    Do not count X->Y in kg1 and Y->X in kg2 as differences.
    """
    edges1 = {frozenset((u, v)) for u, v in kg1.edges()}
    edges2 = {frozenset((u, v)) for u, v in kg2.edges()}
    total_pairs = len(edges1.union(edges2))
    R=edges1.symmetric_difference(edges2)
    diff_pairs_count = len(edges1.symmetric_difference(edges2))
    return diff_pairs_count / total_pairs if total_pairs > 0 else 0

def diff_dir(kg1, kg2):
    """
    Calculate the difference in direction for same node pairs.
    Do not count as differences if both directions exist with the same relation.
    """
    edges1 = {frozenset((u, v)) for u, v in kg1.edges()}
    edges2 = {frozenset((u, v)) for u, v in kg2.edges()}
    same_pairs = edges1.intersection(edges2)
    total_pairs = len(edges1.union(edges2))

    diff_dir_count = 0
    for pair in same_pairs:
        u, v = list(pair)
        has_u_to_v_1 = (u, v) in kg1.edges
        has_v_to_u_1 = (v, u) in kg1.edges
        has_u_to_v_2 = (u, v) in kg2.edges
        has_v_to_u_2 = (v, u) in kg2.edges

        # Check relations if both directions exist
        if has_u_to_v_1 and has_v_to_u_1 and has_u_to_v_2 and has_v_to_u_2:
            rel1_u_to_v = kg1[u][v].get("relation")
            rel1_v_to_u = kg1[v][u].get("relation")
            rel2_u_to_v = kg2[u][v].get("relation")
            rel2_v_to_u = kg2[v][u].get("relation")

            if rel1_u_to_v == rel2_u_to_v and rel1_v_to_u == rel2_v_to_u:
                continue  # Skip counting this pair as a difference

        # Count as difference if any of the above conditions fail
        if has_u_to_v_1 and has_v_to_u_2 or has_v_to_u_1 and has_u_to_v_2:
            diff_dir_count += 1

    return diff_dir_count / len(same_pairs) if same_pairs else 0
    # return diff_dir_count*2 / total_pairs

def diff_edge(kg1, kg2):
    """
    Calculate the difference in edge labels for same directional pairs.
    """
    edges1 = {(u, v, data["relation"]) for u, v, data in kg1.edges(data=True)}
    edges2 = {(u, v, data["relation"]) for u, v, data in kg2.edges(data=True)}

    directional_pairs1 = {(u, v) for u, v, _ in edges1}
    directional_pairs2 = {(u, v) for u, v, _ in edges2}
    same_directional_pairs = directional_pairs1.intersection(directional_pairs2)
    total_pairs = len(edges1.union(edges2))

    diff_edge_count = sum(
        1 for u, v in same_directional_pairs if kg1[u][v]["relation"] != kg2[u][v]["relation"]
    )
    return diff_edge_count / len(same_directional_pairs) if same_directional_pairs else 0
    # return diff_edge_count*2 / total_pairs

def evaluate_diff(kg1, kg2):
    """
    Evaluate differences between two knowledge graphs using diff_pair, diff_dir, and diff_edge.
    """
    diff_pair_result = diff_pair(kg1, kg2)
    diff_dir_result = diff_dir(kg1, kg2)
    diff_edge_result = diff_edge(kg1, kg2)

    return {
        "diff_pair": diff_pair_result,
        "diff_dir": diff_dir_result,
        "diff_edge": diff_edge_result,
    }
def evaluate_diffs(kg_list):
    """
    Evaluate all pairs of knowledge graphs in the list and print top 3 stats for diff_pair, diff_dir, and diff_edge.

    :param kg_list: List of knowledge graphs (NetworkX DiGraph)
    """
    # Initialize lists to keep track of top 3 for each metric
    top_diff_pair = []
    top_diff_dir = []
    top_diff_edge = []

    n = len(kg_list)

    for i in range(n):
        for j in range(i + 1, n):
            kg1 = kg_list[i]
            kg2 = kg_list[j]

            # Compute differences using evaluate_diff
            stats = evaluate_diff(kg1, kg2)

            # Add current stats to each respective list
            top_diff_pair.append({"diff_pair": stats["diff_pair"], "diff_dir": stats["diff_dir"], "diff_edge": stats["diff_edge"], "pair": (i, j)})
            top_diff_dir.append({"diff_pair": stats["diff_pair"], "diff_dir": stats["diff_dir"], "diff_edge": stats["diff_edge"], "pair": (i, j)})
            top_diff_edge.append({"diff_pair": stats["diff_pair"], "diff_dir": stats["diff_dir"], "diff_edge": stats["diff_edge"], "pair": (i, j)})

            # Keep only the top 3 for each list (sorted by their respective metric)
            top_diff_pair = sorted(top_diff_pair, key=lambda x: x["diff_pair"], reverse=True)[:3]
            top_diff_dir = sorted(top_diff_dir, key=lambda x: x["diff_dir"], reverse=True)[:3]
            top_diff_edge = sorted(top_diff_edge, key=lambda x: x["diff_edge"], reverse=True)[:3]

    # Print top 3 results for each metric
    print("Top 3 diff_pair:")
    for rank, stats in enumerate(top_diff_pair, start=1):
        print(f"{rank}. {stats}")

    print("\nTop 3 diff_dir:")
    for rank, stats in enumerate(top_diff_dir, start=1):
        print(f"{rank}. {stats}")

    print("\nTop 3 diff_edge:")
    for rank, stats in enumerate(top_diff_edge, start=1):
        print(f"{rank}. {stats}")


def read_and_evaluate_kg_files(file_paths):
    """
    Read KG files, load them as NetworkX DiGraphs, and evaluate differences.

    :param file_paths: List of file paths to the KG .pkl files
    """
    kg_list = []
    for file_path in file_paths:
        try:
            kg = load_knowledge_graph(file_path)
            kg_list.append(kg)
        except Exception as e:
            print(f"Failed to load KG from {file_path}: {str(e)}")

    if len(kg_list) > 1:
        evaluate_diffs(kg_list)
    else:
        print("Not enough KGs to evaluate differences.")


import os
import pickle
import json

# Function to save a knowledge graph
def save_knowledge_graph(graph, file_path):
    """
    Save a NetworkX DiGraph (knowledge graph) to a file.

    :param graph: NetworkX DiGraph to save
    :param file_path: Path to the file where the graph will be saved
    """
    with open(file_path, 'wb') as file:
        pickle.dump(graph, file)

# Function to load a knowledge graph
def load_knowledge_graph(file_path):
    """
    Load a NetworkX DiGraph (knowledge graph) from a file.

    :param file_path: Path to the file where the graph is stored
    :return: Loaded NetworkX DiGraph
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Function to load schema from a JSON file
def load_schema(file_path):
    """
    Load a schema from a JSON file.

    :param file_path: Path to the JSON file containing the schema
    :return: Dictionary with schema details
    """
    with open(file_path, "r") as file:
        return json.load(file)

# Function to evaluate relationships in a graph
def relationship_eval(graph, schema):
    """
    Evaluate unique types of relationships in the graph against the schema.

    :param graph: NetworkX DiGraph
    :param schema: Dictionary with "relationships" as a set of valid relationships
    :return: Dictionary with metrics
    """
    # Extract unique relationship types in the graph
    unique_relationships_in_kg = {data.get("relation") for _, _, data in graph.edges(data=True)}
    unique_relationships_in_kg.discard(None)  # Remove None values if any

    schema_relationships = set(schema.get("relationships", []))

    # Calculate metrics
    relationships_in_schema = unique_relationships_in_kg.intersection(schema_relationships)
    relationships_not_in_schema = unique_relationships_in_kg.difference(schema_relationships)

    return {
        "in_schema_ratio": len(relationships_in_schema) / len(schema_relationships) if len(schema_relationships) > 0 else 0,
        "not_in_schema_ratio": len(relationships_not_in_schema) / len(schema_relationships) if len(schema_relationships) > 0 else 0,
    }

# Function to evaluate entities in a graph
def entity_eval(graph, schema):
    """
    Evaluate unique types of entities in the graph against the schema.

    :param graph: NetworkX DiGraph
    :param schema: Dictionary with "entity_types" as a set of valid entity types
    :return: Dictionary with metrics
    """
    # Extract unique entity types in the graph
    unique_entity_types_in_kg = {data.get("type") for _, data in graph.nodes(data=True)}
    unique_entity_types_in_kg.discard(None)  # Remove None values if any

    schema_entity_types = set(schema.get("entity_types", []))

    # Calculate metrics
    entities_in_schema = unique_entity_types_in_kg.intersection(schema_entity_types)
    entities_not_in_schema = unique_entity_types_in_kg.difference(schema_entity_types)

    return {
        "in_schema_ratio": len(entities_in_schema) / len(schema_entity_types) if len(schema_entity_types) > 0 else 0,
        "not_in_schema_ratio": len(entities_not_in_schema) / len(schema_entity_types) if len(schema_entity_types) > 0 else 0,
    }
def evaluate_kg_completeness(schema_file, kg_files):
    """
    Load 5 KGs and evaluate their relationship and entity metrics.

    :param schema_file: Path to the schema file (JSON format)
    :param kg_files: List of paths to the KG files (pickle format)
    """
    # Load the schema
    schema = load_schema(schema_file)

    # Process each KG file
    for idx, kg_file in enumerate(kg_files, start=1):
        try:
            graph = load_knowledge_graph(kg_file)
            relationship_metrics = relationship_eval(graph, schema)
            entity_metrics = entity_eval(graph, schema)

            print(f"Metrics for KG {idx} ({kg_file}):")
            print("  Relationship Metrics:", relationship_metrics)
            print("  Entity Metrics:", entity_metrics)
        except Exception as e:
            print(f"Failed to process KG {idx} ({kg_file}): {str(e)}")
# loaded_kg1 = load_knowledge_graph("kg3.pkl")
# loaded_kg2 = load_knowledge_graph("kg4.pkl")
# results = evaluate_diff(loaded_kg1, loaded_kg2)
# print(results)

#
# file_paths = [f"final_kg_{i}.pkl" for i in range(1, 5)]
# read_and_evaluate_kg_files(file_paths)
if __name__ == "__main__":
    schema_file_path = "schema.json"  # Path to your schema JSON file
    kg_file_paths = [f"final_kg_{i}.pkl" for i in range(1, 6)]  # Paths to your KG files
    evaluate_kg_completeness(schema_file_path, kg_file_paths)
