import json
import os
import sys
import argparse

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.knowledge_graph.config import load_config
from src.knowledge_graph.llm import call_llm, extract_json_from_text
from src.knowledge_graph.visualization import visualize_knowledge_graph, sample_data_visualization

from src.knowledge_graph.text_utils import  chunk_text
from src.knowledge_graph.entity_standardization import standardize_entities, infer_relationships, limit_predicate_length
from src.knowledge_graph.prompts import  MAIN_USER_PROMPT, MAIN_SYSTEM_PROMPT


def process_with_llm(config, input_text, debug=False):
    """
        Process input text with LLM to extract triples.

        Args:
            config: Configuration dictionary
            input_text: Text to analyze
            debug: If True, print detailed debug information

        Returns:
            List of extracted triples or None if processing failed
        """

    # Use prompts from the prompts module
    system_prompt = MAIN_SYSTEM_PROMPT
    user_prompt = MAIN_USER_PROMPT
    user_prompt += f"```\n{input_text}```\n"

    # LLM configuration
    model = config["llm"]["model"]
    api_key = config["llm"]["api_key"]
    max_tokens = config["llm"]["max_tokens"]
    temperature = config["llm"]["temperature"]
    base_url = config["llm"]["base_url"]

    # Process with LLM
    metadata = {}
    response = call_llm(model, user_prompt, api_key, system_prompt, max_tokens, temperature, base_url)

    # Print raw response only if debug mode is on
    if debug:
        print("Raw LLM response:")
        print(response)
        print("\n---\n")

    result = extract_json_from_text(response)
    if result:
        # Validate and filter triples to ensure they have all required fields
        valid_triples = []
        invalid_triples = []
        invalid_count = 0
        for item in result:
            if isinstance(item, dict) and "subject" in item and "predicate" in item and "object" in item:
                # Add metadata to valid items
                valid_triples.append(dict(item, **metadata))
            else :
                invalid_count += 1
                invalid_triples.append(item)

        if invalid_count > 0:
            print(f"Warning: Filtered out {invalid_count} invalid triples missing required fields")

        if not valid_triples:
            print("Error: No valid triples found in LLM response")
            return None

            # Apply predicate length limit to all valid triples
        for triple in valid_triples:
            triple["predicate"] = limit_predicate_length(triple["predicate"])

        # Print extracted JSON only if debug mode is on
        if debug:
            print("Extracted JSON:")
            print(json.dumps(valid_triples, indent=2))  # Pretty print the JSON
            print("\n---\n")
            print(f" invalid triples {invalid_triples}")

        return valid_triples

    else:
        # Always print error messages even if debug is off
        print("\n\nERROR ### Could not extract valid JSON from response: ", response, "\n\n")
        return None


def get_unique_entities(triples):
    """
    Get the set of unique entities from the triples.

    Args:
        triples: List of triple dictionaries

    Returns:
        Set of unique entity names
    """
    entities = set()
    for triple in triples:
        if not isinstance(triple, dict):
            continue
        if "subject" in triple:
            entities.add(triple["subject"])
        if "object" in triple:
            entities.add(triple["object"])
    return entities

def process_text_in_chunks(config, full_text, debug=False):
    """
        Process a large text by breaking it into chunks with overlap,
        and then processing each chunk separately.

        Args:
            config: Configuration dictionary
            full_text: The complete text to process
            debug: If True, print detailed debug information

        Returns:
            List of all extracted triples from all chunks
        """

    # Get chunking parameters from config
    chunk_size = config.get("chunking", {}).get("chunk_size", 500)
    overlap = config.get("chunking", {}).get("overlap", 50)
    text_chunks = chunk_text(full_text, chunk_size, overlap)
    print("=" * 50)
    print("PHASE 1: INITIAL TRIPLE EXTRACTION")
    print("=" * 50)
    print(f"Processing text in {len(text_chunks)} chunks (size: {chunk_size} words, overlap: {overlap} words)")

    # Process each chunk
    all_results = []
    for i, chunk in enumerate(text_chunks):
        print(f"Processing chunk {i + 1}/{len(text_chunks)} ({len(chunk.split())} words)")

        # Process the chunk with LLM
        chunk_results = process_with_llm(config, chunk, debug)

        if chunk_results:
            # Add chunk information to each triple
            for item in chunk_results:
                item["chunk"] = i + 1

            # Add to overall results
            all_results.extend(chunk_results)
        else:
            print(f"Warning: Failed to extract triples from chunk {i + 1}")

    print(f"\nExtracted a total of {len(all_results)} triples from all chunks")

    # Apply entity standardization if enabled
    if config.get("standardization", {}).get("enabled", False):
        print("\n" + "=" * 50)
        print("PHASE 2: ENTITY STANDARDIZATION")
        print("=" * 50)
        print(f"Starting with {len(all_results)} triples and {len(get_unique_entities(all_results))} unique entities")

        all_results = standardize_entities(all_results, config)

        print(
            f"After standardization: {len(all_results)} triples and {len(get_unique_entities(all_results))} unique entities")

        # Apply relationship inference if enabled
        if config.get("inference", {}).get("enabled", False):
            print("\n" + "=" * 50)
            print("PHASE 3: RELATIONSHIP INFERENCE")
            print("=" * 50)
            print(f"Starting with {len(all_results)} triples")

            # Count existing relationships
            relationship_counts = {}
            for triple in all_results:
                relationship_counts[triple["predicate"]] = relationship_counts.get(triple["predicate"], 0) + 1

            print("Top 5 relationship types before inference:")
            for pred, count in sorted(relationship_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {pred}: {count} occurrences")

            all_results = infer_relationships(all_results, config)

            # Count relationships after inference
            relationship_counts_after = {}
            for triple in all_results:
                relationship_counts_after[triple["predicate"]] = relationship_counts_after.get(triple["predicate"],
                                                                                               0) + 1

            print("\nTop 5 relationship types after inference:")
            for pred, count in sorted(relationship_counts_after.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {pred}: {count} occurrences")

            # Count inferred relationships
            inferred_count = sum(1 for triple in all_results if triple.get("inferred", False))
            print(f"\nAdded {inferred_count} inferred relationships")
            print(f"Final knowledge graph: {len(all_results)} triples")

        return all_results


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Generator and Visualizer")
    parser.add_argument('--test', action='store_true', help='Generate a test visualization with sample data')

    parser.add_argument('--config', type=str, default='/Users/rajesh/Desktop/rajesh/Archive/teaching/agentic_ai/BMGR-MAY2025-GIAI-2/KG_with_unstructure_data/config.toml', help='Path to configuration file')
    parser.add_argument('--output', type=str, default='knowledge_graph.html', help='Output HTML file path')
    parser.add_argument('--input', type=str, required=False,
                        help='Path to input text file (required unless --test is used)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output (raw LLM responses and extracted JSON)')
    parser.add_argument('--no-standardize', action='store_true', help='Disable entity standardization')
    parser.add_argument('--no-inference', action='store_true', help='Disable relationship inference')

    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        print(f"Failed to load configuration from {args.config}. Exiting.")
        return

    # If test flag is provided, generate a sample visualization
    if args.test:
        print("Generating sample data visualization...")
        sample_data_visualization(args.output, config=config)
        print(f"\nSample visualization saved to {args.output}")
        print(f"To view the visualization, open the following file in your browser:")
        print(f"file://{os.path.abspath(args.output)}")
        return

    # For normal processing, input file is required
    if not args.input:
        print("Error: --input is required unless --test is used")
        parser.print_help()
        return

    # Override configuration settings with command line arguments
    if args.no_standardize:
        config.setdefault("standardization", {})["enabled"] = False
    if args.no_inference:
        config.setdefault("inference", {})["enabled"] = False

    # Load input text from file
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_text = f.read()
        print(f"Using input text from file: {args.input}")
    except Exception as e:
        print(f"Error reading input file {args.input}: {e}")
        return

    # Process text in chunks
    result = process_text_in_chunks(config, input_text, args.debug)

    if result:
        # Save the raw data as JSON for potential reuse
        json_output = args.output.replace('.html', '.json')
        try:
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"Saved raw knowledge graph data to {json_output}")
        except Exception as e:
            print(f"Warning: Could not save raw data to {json_output}: {e}")

        # Visualize the knowledge graph
        stats = visualize_knowledge_graph(result, args.output, config=config)
        print("\nKnowledge Graph Statistics:")
        print(f"Nodes: {stats['nodes']}")
        print(f"Edges: {stats['edges']}")
        print(f"Communities: {stats['communities']}")

        # Provide command to open the visualization in a browser
        print("\nTo view the visualization, open the following file in your browser:")
        print(f"file://{os.path.abspath(args.output)}")
    else:
        print("Knowledge graph generation failed due to errors in LLM processing.")




if __name__ == "__main__":
    main()



