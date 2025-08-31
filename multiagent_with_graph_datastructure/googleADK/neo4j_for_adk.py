import os
from typing import Any, Dict
import atexit

from dotenv import load_dotenv
load_dotenv()

from neo4j import (
    GraphDatabase,
    Result
)

def tool_success(key:str,result: Any) -> Dict[str, Any]:
    """Convenience function to return a success result."""
    return {
        'status': 'success',
        key: result
    }

def tool_error(message: str) -> Dict[str, Any]:
    """Convenience function to return an error result."""
    return {
        'status': 'error',
        'error_message': message
    }



def to_python(value):
    from neo4j.graph import Node, Relationship, Path
    from neo4j import Record
    import neo4j.time
    if isinstance(value, Record):
        return {k: to_python(v) for k, v in value.items()}
    elif isinstance(value, dict):
        return {k: to_python(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [to_python(v) for v in value]
    elif isinstance(value, Node):
        return {
            "id": value.id,
            "labels": list(value.labels),
            "properties": to_python(dict(value))
        }
    elif isinstance(value, Relationship):
        return {
            "id": value.id,
            "type": value.type,
            "start_node": value.start_node.id,
            "end_node": value.end_node.id,
            "properties": to_python(dict(value))
        }
    elif isinstance(value, Path):
        return {
            "nodes": [to_python(node) for node in value.nodes],
            "relationships": [to_python(rel) for rel in value.relationships]
        }
    elif isinstance(value, neo4j.time.DateTime):
        return value.iso_format()
    elif isinstance(value, (neo4j.time.Date, neo4j.time.Time, neo4j.time.Duration)):
        return str(value)
    else:
        return value
    
def result_to_adk(result: Result) -> Dict[str, Any]:
    eager_result = result.to_eager_result()
    records = [to_python(record.data()) for record in eager_result.records]
    return tool_success("query_result",records)



class Neo4jForADK:
    """
    A wrapper for querying Neo4j which returns ADK-friendly responses.
    """
    _driver = None
    database_name = "neo4j"

    def __init__(self):
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME") or "neo4j"
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        neo4j_database = os.getenv("NEO4J_DATABASE") or os.getenv("NEO4J_USERNAME") or "neo4j"
        self.database_name = neo4j_database
        self._driver =  GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_username, neo4j_password)
        )
    
    def get_driver(self):
        return self._driver
    
    def close(self):
        return self._driver.close()
    
    def send_query(self, cypher_query, parameters=None) -> Dict[str, Any]:
        session = self._driver.session()
        try:
            result = session.run(
                cypher_query, 
                parameters or {},
                database_=self.database_name
            )
            return result_to_adk(result)
        except Exception as e:
            return tool_error(str(e))
        finally:
            session.close()


graphdb = Neo4jForADK()

# Register cleanup function to close database connection on exit
atexit.register(graphdb.close)