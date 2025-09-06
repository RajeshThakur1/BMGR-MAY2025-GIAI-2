
import logging
from pathlib import Path
from itertools import islice

from typing import Dict, Any

from google.adk.tools import ToolContext

from neo4j_for_adk import graphdb,tool_success, tool_error

from helper import get_neo4j_import_dir


def get_approved_user_goal(tool_context: ToolContext):
    """Returns the user's goal, which is a dictionary containing the kind of graph and its description."""
    if "approved_user_goal" not in tool_context.state:
        return tool_error("approved_user_goal not set. Ask the user to clarify their goal (kind graph and descriptions)")
    user_goal_data = tool_context.state["approved_user_goal"]

    return tool_success("approved_user_goal", user_goal_data)


def get_approved_files(tool_context: ToolContext):
    """Returns the files that have been approved for import."""
    if "approved_files" not in tool_context.state:
        return tool_error("approved_files not set. Ask the user to approve the file suggestions.")
    files = tool_context.state["approved_files"]

    return tool_success("approved_files", files)




def list_available_files(tool_context:ToolContext) -> dict:
    # this constant will be used as the key for storing the file list in the tool context state
    ALL_AVAILABLE_FILES = "all_available_files"
    f"""Lists files available for knowledge graph construction.
    All files are relative to the import directory.

    Returns:
        dict: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a {ALL_AVAILABLE_FILES} key with list of file names.
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    # get the import dir using the helper function
    import_dir = Path(get_neo4j_import_dir())
    file_names = [str(file.relative_to(import_dir)) for file in import_dir.rglob("*") if file.is_file()]
    # save the list to state so we can inspect it later
    tool_context.state[ALL_AVAILABLE_FILES] = file_names
    return tool_success(ALL_AVAILABLE_FILES, file_names)






def sample_file(file_path: str)-> dict:
    """Samples a file by reading its content as text.
    
    Treats any file as text and reads up to a maximum of 100 lines.
    
    Args:
      file_path: file to sample, relative to the import directory
      
    Returns:
        dict: A dictionary containing metadata about the content,
              along with a sampling of the file.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'content' key with textual file content.
              If 'error', includes an 'error_message' key.
    """
    import_dir = Path(get_neo4j_import_dir() or "")
    if not import_dir.exists():
        return tool_error("NEO4J_IMPORT_DIR does not exist or is undefined: {import_dir}")
    full_path_to_file = import_dir / file_path

    if not full_path_to_file.exists():
        return tool_error(f"File does not exist in import directory: {file_path}")


    try:
        with open(full_path_to_file, 'r', encoding="utf-8") as file:
            lines = list(islice(file, 100))
            content = ''.join(lines)
            return tool_success("content", content)

    except Exception as e:
        return tool_error(f"Error reading or processing file {file_path}: {e}")
    

### Neo4j Tools ###
def neo4j_is_ready():
    return graphdb.send_query("RETURN 'Neo4j is Ready!' as message")

def drop_neo4j_indexes() -> Dict[str, Any]:
    """Drops and constraints and indexes present on the neo4j graph database

    Returns:
        Success or an error.
    """
    # remove all constraints
    list_constraints = graphdb.send_query(
        """SHOW CONSTRAINTS YIELD name"""
    )
    if (list_constraints == "error"):
        return list_constraints
    constraint_names = [row["name"] for row in list_constraints["query_result"]]
    for constraint_name in constraint_names:
        dropped_constraint = graphdb.send_query("""DROP CONSTRAINT $constraint_name""", {"constraint_name": constraint_name})
        if (dropped_constraint["status"] == "error"):
            return dropped_constraint

    # remove all indexes
    list_indexes = graphdb.send_query(
        """SHOW INDEXES YIELD name"""
    )
    if (list_indexes == "error"):
        return list_indexes
    index_names = [row["name"] for row in list_indexes["query_result"]]
    for index_name in index_names:
        dropped_index = graphdb.send_query("""DROP INDEX $index_name""", {"index_name": index_name})
        if (dropped_index["status"] == "error"):
            return dropped_index

    return tool_success("message", "Neo4j constraints and indexes have been dropped.")

def clear_neo4j_data() -> Dict[str, Any]:
    """Clears all data from the neo4j graph database.

    Use with caution! Confirm with the user
    that they know this will completely reset the database.

    Returns:
        Success or an error.
    """
    # First, remove all nodes and relationships in batches
    data_removed = graphdb.send_query("""MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS""")
    if (data_removed["status"] == "error") :
        return data_removed

    return tool_success("message", "Neo4j graph has been reset.")

def get_apoc_procedure_names() -> Dict[str, Any]:
    """List all APOC procedure names.
    APOC (Awesome Procedures on Cypher) is a library of procedures and functions that extends the capabilities of Neo4j.

    Returns:
        Success with a list of APOC procedure names or an error.
    """
    cypher = "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc' RETURN name"

    result = graphdb.send_query(cypher)
    
    if result["status"] == "error":
        return result
    apoc_procedure_names = [row["name"] for row in result["query_result"]]

    if len(apoc_procedure_names) == 0:
        return tool_error("APOC procedures not found. Please ensure APOC is installed in your Neo4j database.")
    
    return tool_success("apoc_procedure_names", apoc_procedure_names)

def get_apoc_version() -> Dict[str, Any]:
    """Get the version of APOC installed in the Neo4j database.

    Returns:
        Success with the APOC version or an error.
    """
    cypher = "RETURN apoc.version() AS apoc_version"

    result = graphdb.send_query(cypher)
    
    if result["status"] == "error":
        return result
    
    apoc_version = result["query_result"][0]["apoc_version"]

    return tool_success("apoc_version", apoc_version)

def get_neo4j_version() -> Dict[str, Any]:
    """Get the version and edition of the Neo4j database.
    
    """
    cypher = "CALL dbms.components() yield name, versions, edition unwind versions as version return name, version, edition"

    result = graphdb.send_query(cypher)

    if result["status"] == "error":
        return result
    
    return tool_success("neo4j_version", result["query_result"][0])

def create_uniqueness_constraint(
    label: str,
    unique_property_key: str,
) -> Dict[str, Any]:
    """Creates a uniqueness constraint for a node label and property key.
    A uniqueness constraint ensures that no two nodes with the same label and property key have the same value.
    This improves the performance and integrity of data import and later queries.

    Args:
        label: The label of the node to create a constraint for.
        unique_property_key: The property key that should have a unique value.

    Returns:
        A dictionary with a status key ('success' or 'error').
        On error, includes an 'error_message' key.
    """    
    # Use string formatting since Neo4j doesn't support parameterization of labels and property keys when creating a constraint
    constraint_name = f"{label}_{unique_property_key}_constraint"
    query = f"""CREATE CONSTRAINT `{constraint_name}` IF NOT EXISTS
    FOR (n:`{label}`)
    REQUIRE n.`{unique_property_key}` IS UNIQUE"""
    results = graphdb.send_query(query)
    return results

def load_nodes_from_csv(
    source_file: str,
    label: str,
    unique_column_name: str,
    properties: list[str],
) -> Dict[str, Any]:
    """Batch loading of nodes from a CSV file"""

    # load nodes from CSV file by merging on the unique_column_name value
    query = f"""LOAD CSV WITH HEADERS FROM "file:///" + $source_file AS row
    CALL (row) {{
        MERGE (n:$($label) {{ {unique_column_name} : row[$unique_column_name] }})
        FOREACH (k IN $properties | SET n[k] = row[k])
    }} IN TRANSACTIONS OF 1000 ROWS
    """

    results = graphdb.send_query(query, {
        "source_file": source_file,
        "label": label,
        "unique_column_name": unique_column_name,
        "properties": properties
    })
    return results

def load_product_nodes() -> Dict[str, Any]:
    """Load the product nodes from products.csv"""
    return load_nodes_from_csv(
        "products.csv",
        "Product",
        "product_id",
        ["product_name", "price", "description"]
    )

