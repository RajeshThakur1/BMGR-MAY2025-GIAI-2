"""Configuration utilities for the knowledge graph generator."""
import tomli
import os

### if your configuration is coming from yaml

# def read_yaml(path_to_yaml: str) -> dict:
#     with open(path_to_yaml) as yaml_file:
#         content = yaml.safe_load(yaml_file)
#     logging.info(f"yaml file: {path_to_yaml} loaded successfully")
#     return content

def load_config(config_path="config.toml"):

    """

    Load Configuration from TOML File.

    :param config_path: Path of the TOML configuration file
    :return:
        Dictionary containing the configuration or Non if loading fails
    """
    try:
        with open(config_path, "rb") as config_file:
            config = tomli.load(config_file)
            return config

    except Exception as e:
        print(f"Error loading config file: {e}")
        return None


if __name__ == "__main__":
    config = load_config("/Users/rajesh/Desktop/rajesh/Archive/teaching/agentic_ai/BMGR-MAY2025-GIAI-2/KG_with_unstructure_data/config.toml")
    print(config)

