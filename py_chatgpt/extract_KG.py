import networkx as nx
import re
import matplotlib.pyplot as plt

from utils import save_knowledge_graph
def extract_entities(turtle_content):
    """
    Extract all entities with their types and labels.
    """
    entities = {}
    # Correct entity pattern to handle blocks ending with a semicolon or period
    entity_pattern = r":(\w+)\s+rdf:type\s+:(\w+)\s*;\s*rdfs:label\s+\"(.*?)\"\s*(?:;|\.?)"

    for match in re.finditer(entity_pattern, turtle_content, re.DOTALL):
        entity, entity_type, label = match.groups()
        entities[entity] = {"type": entity_type, "label": label.strip()}
    return entities


def extract_relationships(turtle_content, graph, entities):
    """
    Extract relationships block by block and add to the graph.
    """
    # Split the content into blocks
    blocks = turtle_content.split("\n\n")

    for block in blocks:
        # Skip empty blocks
        if not block.strip():
            continue

        # Extract the main entity
        lines = block.strip().split("\n")
        header_line = lines[0]
        match = re.match(r":(\w+)\s+rdf:type\s+:(\w+)\s*;", header_line)
        if not match:
            continue
        subject = match.group(1)

        # Process each line for relationships
        for line in lines[1:]:  # Exclude the first line
            if line.strip().startswith(":"):  # Relationship line
                # Match predicate and all objects
                relation_match = re.match(r"\s*:(\w+)\s+:(.*?)(?:;|\.)", line)
                if relation_match:
                    predicate, objects = relation_match.groups()
                    # Split objects, strip colons, and clean whitespace
                    objects = [obj.strip().strip(":") for obj in objects.split(",")]

                    # Add an edge for each object
                    for obj in objects:
                        if subject in entities and obj in entities:
                            graph.add_edge(subject, obj, relation=predicate)
def plot_graph(graph, type_color_map, output_file="graph.pdf"):
    """
    Plot the graph with nodes and edges.

    - Utilize the color map for node types.
    - Ensure relations from u to v and v to u are drawn as two directed lines.
    - Use straight lines for single-direction edges.
    - Use straight lines for u to v and slightly curved lines for v to u in bidirectional cases.
    - Label each relation on its respective edge without overlapping.
    - Label the nodes.
    """
    # Define node colors based on their type
    node_colors = []
    for node in graph.nodes(data=True):
        node_type = node[1].get("type", "")
        node_colors.append(type_color_map.get(node_type, "gray"))  # Default to gray if type is missing

    # Create figure and axis
    plt.figure(figsize=(14, 14))  # Increased size for better spacing

    # Generate layout for the graph
    pos = nx.spring_layout(graph, seed=42, k=0.5)  # Adjusted k for more spacing

    # Draw nodes with color mapping
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_colors, node_size=700, alpha=0.9
    )

    # Draw node labels
    nx.draw_networkx_labels(
        graph, pos, labels={n: d["label"] for n, d in graph.nodes(data=True)}, font_size=10
    )

    # Draw edges with distinct styles for bidirectional relationships
    edge_labels = nx.get_edge_attributes(graph, 'relation')
    drawn_edges = set()  # Track drawn edges to avoid duplicates

    for u, v, data in graph.edges(data=True):
        if (v, u) in graph.edges and (v, u) not in drawn_edges:  # Bidirectional case
            # Straight line for u -> v
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=[(u, v)],
                arrowstyle="-|>",
                arrowsize=20,
                connectionstyle="arc3,rad=0.0",
                edge_color="gray",
                width=1.5,
            )
            # Slightly curved line for v -> u
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=[(v, u)],
                arrowstyle="-|>",
                arrowsize=20,
                connectionstyle="arc3,rad=0.1",  # Reduced curvature
                edge_color="gray",
                width=1.5,
            )
            drawn_edges.add((u, v))
            drawn_edges.add((v, u))
        elif (u, v) not in drawn_edges:  # Single-direction case
            # Straight line for u -> v
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=[(u, v)],
                arrowstyle="-|>",
                arrowsize=20,
                connectionstyle="arc3,rad=0.0",
                edge_color="gray",
                width=1.5,
            )
            drawn_edges.add((u, v))

        # Add edge labels on the correct line with distinct offsets for bidirectional edges
        offset_x = 0.05 if (u < v or (v, u) not in graph.edges) else -0.05  # Offset based on direction
        offset_y = 0.05 if (u < v or (v, u) not in graph.edges) else -0.05
        label_pos = (
            (pos[u][0] + pos[v][0]) / 2 + offset_x,  # Adjusted offset
            (pos[u][1] + pos[v][1]) / 2 + offset_y,  # Adjusted offset
        )
        plt.text(
            label_pos[0],
            label_pos[1],
            edge_labels[(u, v)],
            fontsize=8,
            color="blue",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            horizontalalignment="center",
            verticalalignment="center",
        )

    # Save the graph to a file
    plt.axis("off")
    plt.title("Knowledge Graph Visualization")
    plt.tight_layout()
    plt.savefig(output_file, format="pdf")
    # plt.show()

def extract_KG(file_path= "final_kg4.txt", output_kg_file="kg1.pkl",output_pdf_file="KG.pdf"):
    # Create a directed graph
    G = nx.DiGraph()
    # Read the Turtle KG content from the file
    with open(file_path, "r", encoding="utf-8") as file:
        turtle_kg = file.read()

    # Skip the first two lines (metadata and header)
    turtle_kg = "\n".join(turtle_kg.split("\n")[2:])

    # Step 1: Extract all entities
    entities = extract_entities(turtle_kg)

    # Step 2: Add nodes to the graph
    for entity, data in entities.items():
        G.add_node(entity, type=data["type"], label=data["label"])

    # Step 3: Extract relationships and add edges
    extract_relationships(turtle_kg, G, entities)

    if G.number_of_edges() == 0:
        print("No edges found in the knowledge graph. Returning the graph with nodes only.")
        save_knowledge_graph(G, output_kg_file)
        return

    save_knowledge_graph(G,output_kg_file)
    # Define color mapping for node types
    type_color_map = {
        "Molecule": "blue",
        "Drug": "green",
        "SideEffect": "red",
        "Gene": "purple",
        "Disease": "orange",
        "Pathway": "cyan",
        "Anatomy": "brown",
        "Symptom": "pink"
    }



    # Print graph information
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    print("\nNodes:")
    for node, data in G.nodes(data=True):
        print(f"  {node}: {data}")

    print("\nEdges:")
    for source, target, data in G.edges(data=True):
        print(f"  {source} -> {target}, relation: {data['relation']}")
    # Plot and save the graph with edge labels
    plot_graph(G,type_color_map, output_file=output_pdf_file)