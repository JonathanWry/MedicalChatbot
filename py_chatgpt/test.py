import json

schema = {
    "relationships": [
        "relatedTo", "hasSideEffect", "affectsMolecule", "interactsWith",
        "targetsGene", "usedFor", "involvedIn", "geneInteractsWith",
        "associatedWithPathway", "associatedWithDisease", "expressedIn",
        "pathwayAssociatedWithDisease", "leadsToSymptom"
    ],
    "entity_types": [
        "Molecule", "Drug", "SideEffect", "Gene", "Disease",
        "Pathway", "Anatomy", "Symptom"
    ]
}

# Save the schema to a JSON file
def save_schema(schema, file_path):
    """
    Save the schema to a JSON file.

    :param schema: Dictionary with schema details
    :param file_path: Path to the JSON file to save the schema
    """
    with open(file_path, "w") as file:
        json.dump(schema, file, indent=4)
schema_file_path = "schema.json"  # Path to your schema JSON file
save_schema(schema, schema_file_path)